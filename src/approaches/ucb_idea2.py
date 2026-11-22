import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from .buffer import BalancedReplayBuffer
from metrics import compute_cl_metrics, learning_curve_auc


def split_ucb_params(model, lr_dict):
    mu_params, mu_lrs, rho_params, rho_lrs = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'mu' in name:
            mu_params.append(param)
            mu_lrs.append(lr_dict[name])
        elif 'rho' in name:
            rho_params.append(param)
            rho_lrs.append(lr_dict[name])
    return mu_params, mu_lrs, rho_params, rho_lrs


def mc_nll_loss(model, x, y, n_samples=5, t=0, sample=True):
    if not sample or n_samples <= 1:
        logits = model.forward_logits(x, t=t, sample=sample, calculate_log_probs=False)
        return F.cross_entropy(logits, y, reduction='mean')
    loss = 0.0
    for _ in range(n_samples):
        logits = model.forward_logits(x, t=t, sample=True, calculate_log_probs=False)
        loss = loss + F.cross_entropy(logits, y, reduction='mean')
    return loss / float(n_samples)


def _fork_devices_for(model):
    try:
        p = next(model.parameters())
        if p.is_cuda:
            return [p.device]
    except StopIteration:
        pass
    return []


def train_step_idea2(
    model,
    lr_dict,
    x_curr,
    y_curr,
    n_train_examples,
    buffer,
    mode='ucb_d_proj',
    base_lr=0.01,
    n_samples=5,
    t=0,
    replay_alpha=0.5,
    diag_buffer=None,
    measure_diagnostics=False,
    measure_scale_interference=False,
    step_idx=0,
    eps=1e-12,
):
    """
    Unified training step aligned with Algorithm 1 (BBB loss) & Idea 2 proposal:
      Tier 1: 'ucb', 'ucb_replay', 'agem', 'ucb_d_proj'
      Tier 2: 'ucb_raw_proj', 'bft_replay'

    Returns dict of step metrics:
      loss, cos_sim, conflict, has_diag, post_hd, post_violation, has_post_check,
      scale_delta_loss, has_scale_check, diag_overhead_sec
    """
    mu_params, mu_lrs, rho_params, rho_lrs = split_ucb_params(model, lr_dict)
    has_buffer_data = (buffer is not None) and (not buffer.is_empty())
    use_buffer = has_buffer_data and (mode != 'ucb')
    fork_devices = _fork_devices_for(model)

    diag_overhead_sec = 0.0

    # 1. Compute reference buffer likelihood gradient h with respect to posterior means mu
    h_list = None
    buf_nll_val = 0.0
    if use_buffer:
        model.zero_grad()
        x_buf, y_buf = buffer.get_all()
        if mode == 'ucb_d_proj_2nd':
            crn_seed = 987654 + int(step_idx)
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(crn_seed)
                buf_nll = mc_nll_loss(model, x_buf, y_buf, n_samples=n_samples, t=t, sample=True)
                buf_nll_val = float(buf_nll.item())
                buf_nll.backward()
        else:
            buf_nll = mc_nll_loss(model, x_buf, y_buf, n_samples=n_samples, t=t, sample=True)
            buf_nll_val = float(buf_nll.item())
            buf_nll.backward()
        h_list = [p.grad.detach().clone() for p in mu_params]
    elif mode == 'ucb' and measure_diagnostics and (diag_buffer is not None) and (not diag_buffer.is_empty()):
        # Read-only diagnostic probe for replay-free UCB: forked RNG preserves exact UCB trajectory
        t_d0 = time.perf_counter()
        with torch.random.fork_rng(devices=fork_devices):
            model.zero_grad()
            x_buf, y_buf = diag_buffer.get_all()
            buf_nll = mc_nll_loss(model, x_buf, y_buf, n_samples=n_samples, t=t, sample=True)
            buf_nll.backward()
            h_list = [p.grad.detach().clone() for p in mu_params]
        model.zero_grad()
        diag_overhead_sec += time.perf_counter() - t_d0

    # 2. Fast 2-stage Exact Algorithm 1 ELBO gradient:
    #    Stage A: NLL backward -> gives pure likelihood gradient g_lik for diagnostics
    model.zero_grad()
    curr_nll = mc_nll_loss(model, x_curr, y_curr, n_samples=n_samples, t=t, sample=True)
    curr_nll.backward()
    g_lik_list = [p.grad.detach().clone() for p in mu_params]

    #    Stage B: Accumulate exact Algorithm 1 KL / N_train backward WITHOUT zero_grad()
    kl_sum = 0.0
    for _ in range(n_samples):
        kl_sum = kl_sum + model.kl_loss(t=t)
    kl_loss = (kl_sum / float(n_samples)) / float(max(1, int(n_train_examples)))
    kl_loss.backward()

    g_mu_list = [p.grad.detach().clone() for p in mu_params]
    g_rho_list = [p.grad.detach().clone() for p in rho_params]
    g_kl_mu_list = [gm - gl for gm, gl in zip(g_mu_list, g_lik_list)]
    total_loss = float(curr_nll.item() + kl_loss.item())

    # For replay baselines, mix only likelihood gradients while preserving exact KL gradient
    if mode in ('ucb_replay', 'bft_replay') and use_buffer:
        g_mu_list = [
            g_kl + (1.0 - replay_alpha) * g_lik + replay_alpha * h
            for g_kl, g_lik, h in zip(g_kl_mu_list, g_lik_list, h_list)
        ]
        total_loss = (1.0 - replay_alpha) * float(curr_nll.item()) + replay_alpha * buf_nll_val + float(kl_loss.item())

    # 3. Compute diagnostics & apply mean parameter update mu <- mu - d*
    cos_sim = 0.0
    conflict = 0
    has_diag = (h_list is not None)
    post_hd = 0.0
    post_violation = 0
    has_post_check = False
    scale_delta_loss = 0.0
    has_scale_check = False
    curvature_margin = 0.0
    finite_step_conflict = 0
    has_2nd_order_check = False

    with torch.no_grad():
        if has_diag:
            dot_lik = sum((gl * h).sum(dtype=torch.float64) for gl, h in zip(g_lik_list, h_list))
            norm_gl = torch.sqrt(sum((gl * gl).sum(dtype=torch.float64) for gl in g_lik_list) + eps)
            norm_h = torch.sqrt(sum((h * h).sum(dtype=torch.float64) for h in h_list) + eps)
            cos_sim = float((dot_lik / (norm_gl * norm_h)).item())

        proj_triggered = False

        if mode == 'ucb_d_proj' and use_buffer:
            hDg = sum((h * a * g).sum(dtype=torch.float64) for h, a, g in zip(h_list, mu_lrs, g_mu_list))
            hDh = sum((h * a * h).sum(dtype=torch.float64) for h, a in zip(h_list, mu_lrs))
            if hDg.item() < 0.0:
                conflict = 1
                if hDh.item() > eps:
                    proj_triggered = True
                    coeff = float((hDg / hDh).item())
                    d_list = [(a * g) - coeff * (a * h) for a, g, h in zip(mu_lrs, g_mu_list, h_list)]
                else:
                    d_list = [a * g for a, g in zip(mu_lrs, g_mu_list)]
            else:
                d_list = [a * g for a, g in zip(mu_lrs, g_mu_list)]

        elif mode == 'ucb_d_proj_2nd' and use_buffer:
            d0 = [a * g for a, g in zip(mu_lrs, g_mu_list)]
            crn_seed = 987654 + int(step_idx)
            for p, d in zip(mu_params, d0):
                p.sub_(d)
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(crn_seed)
                buf_loss_after = float(mc_nll_loss(model, x_buf, y_buf, n_samples=n_samples, t=t, sample=True).item())
            for p, d in zip(mu_params, d0):
                p.add_(d)

            delta_L = buf_loss_after - buf_nll_val
            hDg = sum((h * a * g).sum(dtype=torch.float64) for h, a, g in zip(h_list, mu_lrs, g_mu_list))
            hDh = sum((h * a * h).sum(dtype=torch.float64) for h, a in zip(h_list, mu_lrs))

            has_2nd_order_check = True
            curvature_margin = float(delta_L + hDg.item())
            finite_step_conflict = 1 if delta_L > 0.0 else 0

            if (hDg.item() < 0.0) or (delta_L > 0.0):
                conflict = 1
                if hDh.item() > eps:
                    proj_triggered = True
                    lam_1st = max(0.0, -float(hDg.item()) / float(hDh.item()))
                    lam_2nd = max(0.0, float(delta_L) / float(hDh.item()))
                    lam = max(lam_1st, min(lam_2nd, lam_1st + 5.0))
                    d_list = [d + lam * (a * h) for d, a, h in zip(d0, mu_lrs, h_list)]
                else:
                    d_list = d0
            else:
                d_list = d0

        elif mode == 'agem' and use_buffer:
            hg = sum((h * g).sum(dtype=torch.float64) for h, g in zip(h_list, g_mu_list))
            hh = sum((h * h).sum(dtype=torch.float64) for h in h_list)
            if hg.item() < 0.0:
                conflict = 1
                if hh.item() > eps:
                    proj_triggered = True
                    coeff = float((hg / hh).item())
                    d_list = [base_lr * (g - coeff * h) for g, h in zip(g_mu_list, h_list)]
                else:
                    d_list = [base_lr * g for g in g_mu_list]
            else:
                d_list = [base_lr * g for g in g_mu_list]

        elif mode == 'ucb_raw_proj' and use_buffer:
            hDg = sum((h * a * g).sum(dtype=torch.float64) for h, a, g in zip(h_list, mu_lrs, g_mu_list))
            if hDg.item() < 0.0:
                conflict = 1
            hg = sum((h * g).sum(dtype=torch.float64) for h, g in zip(h_list, g_mu_list))
            hh = sum((h * h).sum(dtype=torch.float64) for h in h_list)
            if hg.item() < 0.0 and hh.item() > eps:
                proj_triggered = True
                coeff = float((hg / hh).item())
                d_list = [a * (g - coeff * h) for a, g, h in zip(mu_lrs, g_mu_list, h_list)]
            else:
                d_list = [a * g for a, g in zip(mu_lrs, g_mu_list)]

        elif mode == 'bft_replay':
            if has_diag:
                hg = sum((h * g).sum(dtype=torch.float64) for h, g in zip(h_list, g_mu_list))
                if hg.item() < 0.0:
                    conflict = 1
            d_list = [base_lr * g for g in g_mu_list]

        else:  # 'ucb', 'ucb_replay', or Task 0
            if has_diag:
                hDg = sum((h * a * g).sum(dtype=torch.float64) for h, a, g in zip(h_list, mu_lrs, g_mu_list))
                if hDg.item() < 0.0:
                    conflict = 1
            d_list = [a * g for a, g in zip(mu_lrs, g_mu_list)]

        if has_diag and (conflict == 1 or proj_triggered):
            has_post_check = True
            post_hd = float(sum((h * d).sum(dtype=torch.float64) for h, d in zip(h_list, d_list)).item())
            post_violation = 1 if post_hd < -1e-7 else 0

        for p, d in zip(mu_params, d_list):
            p.sub_(d)

        # 4. Measure scale-update interference on buffer loss before and after updating rho (common random numbers)
        ref_buf = buffer if use_buffer else (diag_buffer if mode == 'ucb' else None)
        do_scale_check = bool(measure_scale_interference and (ref_buf is not None) and (not ref_buf.is_empty()))
        if do_scale_check:
            t_s0 = time.perf_counter()
            x_buf_si, y_buf_si = ref_buf.get_all()
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(12345 + int(step_idx))
                loss_before_rho = mc_nll_loss(model, x_buf_si, y_buf_si, n_samples=3, t=t, sample=True)
            diag_overhead_sec += time.perf_counter() - t_s0

        # Exact Algorithm 1, Line 13: rho <- rho - alpha_rho * grad_rho L_BBB
        for p_rho, a_rho, g_rho in zip(rho_params, rho_lrs, g_rho_list):
            p_rho.sub_(a_rho * g_rho)
            p_rho.clamp_(min=-10.0, max=2.0)

        if do_scale_check:
            t_s1 = time.perf_counter()
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(12345 + int(step_idx))
                loss_after_rho = mc_nll_loss(model, x_buf_si, y_buf_si, n_samples=3, t=t, sample=True)
            scale_delta_loss = float(loss_after_rho.item() - loss_before_rho.item())
            has_scale_check = True
            diag_overhead_sec += time.perf_counter() - t_s1

    return {
        'loss': total_loss,
        'cos_sim': cos_sim,
        'conflict': conflict,
        'has_diag': has_diag,
        'post_hd': post_hd,
        'post_violation': post_violation,
        'has_post_check': has_post_check,
        'scale_delta_loss': scale_delta_loss,
        'has_scale_check': has_scale_check,
        'curvature_margin': curvature_margin,
        'finite_step_conflict': finite_step_conflict,
        'has_2nd_order_check': has_2nd_order_check,
        'diag_overhead_sec': diag_overhead_sec,
    }


class Appr(object):
    """
    Continual Learning Trainer for Idea 2: Interference Between Similar Tasks.
    Supports Tier-1 ('ucb', 'ucb_replay', 'agem', 'ucb_d_proj')
    and Tier-2 ('ucb_raw_proj', 'bft_replay') modes.
    """

    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = args.device
        self.init_lr = float(args.lr)
        self.sbatch = int(args.sbatch)
        self.nepochs = int(args.nepochs)
        self.samples = int(args.samples)
        self.eval_samples = int(getattr(args, 'eval_samples', 10))
        self.mode = getattr(args, 'mode', 'ucb_d_proj')
        self.replay_alpha = float(getattr(args, 'replay_alpha', 0.5))
        self.buffer_size = int(getattr(args, 'buffer_size', 200))
        self.checkpoint = args.checkpoint
        self.experiment = args.experiment
        self.seed = int(args.seed)
        self.init_sigma = float(math.log1p(math.exp(float(args.rho))))

        # Replay buffer is strictly empty for 'ucb'; diag_buffer is read-only for logging UCB conflict/scale interference
        self.buffer = BalancedReplayBuffer(capacity=self.buffer_size, device=self.device)
        self.diag_buffer = BalancedReplayBuffer(capacity=self.buffer_size, device=self.device) if self.mode == 'ucb' else None
        self.lr_dict = self._init_lr_dict()

        self.task_diagnostics = {}
        self.learning_curves = {}
        self.total_diag_overhead_sec = 0.0

    def _init_lr_dict(self):
        lr_dict = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'mu' in name:
                lr_dict[name] = torch.full_like(param.data, self.init_lr)
            elif 'rho' in name:
                lr_dict[name] = self.init_lr
        return lr_dict

    def _compute_lr_d_spread(self):
        weight_lrs = [
            v.detach().view(-1) for k, v in self.lr_dict.items()
            if 'weight_mu' in k and isinstance(v, torch.Tensor)
        ]
        if not weight_lrs:
            return {'min': self.init_lr, 'median': self.init_lr, 'max': self.init_lr, 'max_min_ratio': 1.0}
        cat = torch.cat(weight_lrs)
        l_min = float(cat.min().item())
        l_med = float(cat.median().item())
        l_max = float(cat.max().item())
        return {
            'min': l_min,
            'median': l_med,
            'max': l_max,
            'max_min_ratio': float(l_max / max(1e-12, l_min)),
        }

    def update_lr_at_boundary(self):
        """
        Algorithm 2 boundary learning-rate update for UCB modes:
          a_j = init_lr * (sigma_j / init_sigma)
        Strictly linear in sigma_j = log(1 + exp(rho_j)), calibrated so initial scale equals init_lr.
        """
        if self.mode not in ('ucb', 'ucb_replay', 'ucb_d_proj', 'ucb_raw_proj', 'ucb_d_proj_2nd'):
            return

        with torch.no_grad():
            for module_name, module in self.model.named_modules():
                if hasattr(module, 'weight_mu') and hasattr(module, 'weight_rho'):
                    w_name = f"{module_name}.weight_mu" if module_name else "weight_mu"
                    if w_name in self.lr_dict:
                        w_sig = F.softplus(module.weight_rho.data)
                        self.lr_dict[w_name] = self.init_lr * (w_sig / self.init_sigma)
                if getattr(module, 'use_bias', False) and hasattr(module, 'bias_mu'):
                    b_name = f"{module_name}.bias_mu" if module_name else "bias_mu"
                    if b_name in self.lr_dict:
                        b_sig = F.softplus(module.bias_rho.data)
                        self.lr_dict[b_name] = self.init_lr * (b_sig / self.init_sigma)

    def train(self, t, xtrain, ytrain, xvalid, yvalid):
        self.model.train()
        n_train_examples = int(xtrain.size(0))

        epoch_valid_accs = []
        diag_steps = 0
        total_conflicts = 0
        cos_sim_sum = 0.0
        post_check_count = 0
        total_post_violations = 0
        post_hd_sum = 0.0
        scale_check_count = 0
        scale_delta_sum = 0.0
        scale_interference_count = 0
        second_order_check_count = 0
        curvature_margin_sum = 0.0
        finite_step_conflict_count = 0
        task_diag_overhead_sec = 0.0
        task_train_t0 = time.perf_counter()

        global_step = 0
        for e in range(self.nepochs):
            t0 = time.time()
            self.model.train()
            perm = torch.randperm(n_train_examples, device=self.device)

            ep_loss = 0.0
            ep_steps = 0
            ep_diag_steps = 0
            ep_conflicts = 0
            ep_cos_sum = 0.0

            for batch_idx, i in enumerate(range(0, n_train_examples, self.sbatch)):
                b = perm[i:i + self.sbatch]
                xb, yb = xtrain[b], ytrain[b]

                measure_diag = (t > 0) and (self.mode != 'ucb' or (batch_idx % 10 == 0))
                measure_scale = (t > 0) and (batch_idx % 25 == 0)

                step_res = train_step_idea2(
                    model=self.model,
                    lr_dict=self.lr_dict,
                    x_curr=xb,
                    y_curr=yb,
                    n_train_examples=n_train_examples,
                    buffer=self.buffer,
                    mode=self.mode,
                    base_lr=self.init_lr,
                    n_samples=self.samples,
                    t=t,
                    replay_alpha=self.replay_alpha,
                    diag_buffer=self.diag_buffer,
                    measure_diagnostics=measure_diag,
                    measure_scale_interference=measure_scale,
                    step_idx=int(t) * 100000 + global_step,
                )
                global_step += 1
                ep_loss += step_res['loss']
                ep_steps += 1
                task_diag_overhead_sec += step_res['diag_overhead_sec']

                if step_res['has_diag']:
                    ep_diag_steps += 1
                    ep_conflicts += step_res['conflict']
                    ep_cos_sum += step_res['cos_sim']
                    diag_steps += 1
                    total_conflicts += step_res['conflict']
                    cos_sim_sum += step_res['cos_sim']

                if step_res['has_post_check']:
                    post_check_count += 1
                    total_post_violations += step_res['post_violation']
                    post_hd_sum += step_res['post_hd']

                if step_res['has_scale_check']:
                    scale_check_count += 1
                    scale_delta_sum += step_res['scale_delta_loss']
                    if step_res['scale_delta_loss'] > 1e-6:
                        scale_interference_count += 1

                if step_res['has_2nd_order_check']:
                    second_order_check_count += 1
                    curvature_margin_sum += step_res['curvature_margin']
                    finite_step_conflict_count += step_res['finite_step_conflict']

            valid_loss, valid_acc = self.eval(t, xvalid, yvalid, n_samples=1)
            epoch_valid_accs.append(float(valid_acc))
            mean_ep_cos = ep_cos_sum / max(1, ep_diag_steps) if ep_diag_steps > 0 else 0.0
            ep_conf_rate = ep_conflicts / max(1, ep_diag_steps) if ep_diag_steps > 0 else 0.0

            dt = time.time() - t0
            print(
                f"| Task {t:2d} | Epoch {e + 1:3d}/{self.nepochs:3d} ({dt:4.1f}s) "
                f"| Train Loss={ep_loss / max(1, ep_steps):.4f} "
                f"| Valid Acc={100.0 * valid_acc:5.2f}% "
                f"| cos(g,h)={mean_ep_cos:+.4f} | conflict={100.0 * ep_conf_rate:5.1f}% |"
            )

        task_total_sec = time.perf_counter() - task_train_t0
        self.total_diag_overhead_sec += task_diag_overhead_sec

        # Task-boundary hooks:
        # 1. Update replay buffer (200 examples balanced across observed tasks/classes)
        if self.mode != 'ucb':
            self.buffer.update_after_task(t, xtrain, ytrain, seed=self.seed)
        elif self.diag_buffer is not None:
            self.diag_buffer.update_after_task(t, xtrain, ytrain, seed=self.seed)

        # 2. Update per-parameter learning rates D = diag(a_j) for UCB modes (Algorithm 2)
        self.update_lr_at_boundary()

        # 3. Record diagnostics & learning curve AUC for task t
        self.learning_curves[int(t)] = epoch_valid_accs
        self.task_diagnostics[int(t)] = {
            'mean_cos_sim': float(cos_sim_sum / max(1, diag_steps)) if diag_steps > 0 else 0.0,
            'conflict_rate': float(total_conflicts / max(1, diag_steps)) if diag_steps > 0 else 0.0,
            'post_proj_violation_rate': float(total_post_violations / max(1, post_check_count)) if post_check_count > 0 else 0.0,
            'mean_post_proj_hd': float(post_hd_sum / max(1, post_check_count)) if post_check_count > 0 else 0.0,
            'post_check_count': int(post_check_count),
            'mean_scale_interference_delta_loss': float(scale_delta_sum / max(1, scale_check_count)) if scale_check_count > 0 else 0.0,
            'scale_interference_rate': float(scale_interference_count / max(1, scale_check_count)) if scale_check_count > 0 else 0.0,
            'scale_check_count': int(scale_check_count),
            'mean_curvature_margin': float(curvature_margin_sum / max(1, second_order_check_count)) if second_order_check_count > 0 else 0.0,
            'finite_step_conflict_rate': float(finite_step_conflict_count / max(1, second_order_check_count)) if second_order_check_count > 0 else 0.0,
            'second_order_check_count': int(second_order_check_count),
            'learning_curve_auc': learning_curve_auc(epoch_valid_accs),
            'buffer_size_after_task': len(self.buffer),
            'lr_d_spread': self._compute_lr_d_spread(),
            'train_time_sec': float(max(0.0, task_total_sec - task_diag_overhead_sec)),
            'diag_overhead_sec': float(task_diag_overhead_sec),
        }
        self.save_model(t)

    def eval(self, t, x, y, debug=False, n_samples=None):
        if n_samples is None:
            n_samples = self.eval_samples
        self.model.eval()
        total_loss = 0.0
        total_acc = 0
        total_num = 0

        with torch.no_grad():
            for i in range(0, x.size(0), self.sbatch):
                xb = x[i:i + self.sbatch].to(self.device)
                yb = y[i:i + self.sbatch].to(self.device)

                if n_samples <= 1:
                    logits = self.model.forward_logits(xb, t=t, sample=False, calculate_log_probs=False)
                    probs = F.softmax(logits, dim=1)
                else:
                    probs_sum = 0.0
                    for _ in range(n_samples):
                        logits = self.model.forward_logits(xb, t=t, sample=True, calculate_log_probs=False)
                        probs_sum = probs_sum + F.softmax(logits, dim=1)
                    probs = probs_sum / float(n_samples)

                log_probs = torch.log(probs.clamp(min=1e-12))
                loss = F.nll_loss(log_probs, yb, reduction='sum')
                pred = log_probs.argmax(dim=1)

                total_loss += float(loss.item())
                total_acc += int((pred == yb).sum().item())
                total_num += int(xb.size(0))

        return total_loss / max(1, total_num), total_acc / max(1, total_num)

    def save_model(self, t):
        os.makedirs(self.checkpoint, exist_ok=True)
        torch.save(
            {'model_state_dict': self.model.state_dict()},
            os.path.join(self.checkpoint, f'model_{t}.pth.tar'),
        )
