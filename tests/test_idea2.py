import copy
import os
import sys
import unittest
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from approaches.buffer import BalancedReplayBuffer
from approaches.ucb_idea2 import Appr, train_step_idea2, split_ucb_params
from metrics import compute_cl_metrics, get_peak_cpu_memory_mb
from networks.distributions import Prior
from networks.mlp_ucb import Net


class Idea2Test(unittest.TestCase):
    def _make_args(self, mode='ucb_d_proj', nepochs=1):
        return SimpleNamespace(
            device='cpu',
            inputsize=[1, 4, 4],
            taskcla=[(0, 4), (1, 4)],
            samples=2,
            eval_samples=2,
            sbatch=8,
            lr=0.05,
            nhid=16,
            nlayers=2,
            shared_head=True,
            rho=-3.0,
            sig1=0.0,
            sig2=6.0,
            pi=0.25,
            arch='mlp',
            nepochs=nepochs,
            output='',
            checkpoint=os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'test_tmp'),
            experiment='rmnist10',
            num_tasks=2,
            mode=mode,
            buffer_size=20,
            replay_alpha=0.5,
            seed=0,
        )

    def test_prior_log_prob_finite_at_large_weight(self):
        args = self._make_args()
        prior = Prior(args)
        w = torch.tensor([10.0, -10.0, 0.0], dtype=torch.float32)
        lp = prior.log_prob(w)
        self.assertTrue(torch.isfinite(lp).item())

    def test_balanced_replay_buffer_stays_at_capacity_and_balanced(self):
        buf = BalancedReplayBuffer(capacity=200, device='cpu')
        self.assertTrue(buf.is_empty())

        # Simulate 5 tasks with 10 classes each
        for t in range(5):
            x = torch.randn(300, 1, 28, 28)
            y = torch.arange(300) % 10
            buf.update_after_task(t, x, y, seed=0)
            self.assertEqual(len(buf), 200)

        # After 5 tasks x 10 classes = 50 buckets, each bucket should have exactly 200 / 50 = 4 examples
        self.assertEqual(len(buf.storage), 50)
        for key, (bx, by) in buf.storage.items():
            self.assertEqual(bx.size(0), 4)
            self.assertEqual(by.size(0), 4)

    def test_d_projection_vs_raw_projection_in_train_step(self):
        """
        Verify on a real train_step_idea2 call with heterogeneous D = diag(lr) that:
          - ucb_d_proj triggers projection on conflict and achieves |h^T d*| < 1e-6 (0 violations),
          - ucb_raw_proj (Euclidean projection before applying D) violates h^T d >= 0 (post_violation == 1).
        """
        torch.manual_seed(0)
        args = self._make_args(mode='ucb_d_proj')
        args.rho = -8.0  # small posterior noise so MC gradients are nearly deterministic
        model_d = Net(args).to('cpu')
        model_raw = copy.deepcopy(model_d)

        # Build conflicting buffer vs current batch with opposite targets on identical inputs
        x = torch.randn(8, 1, 4, 4)
        y_buf = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
        y_curr = torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], dtype=torch.long)

        buf = BalancedReplayBuffer(capacity=8, device='cpu')
        buf.update_after_task(0, x, y_buf, seed=0)

        # Probe pure h and g_lik on a copy to assign a heterogeneous D where raw proj violates h^T d >= 0
        probe = copy.deepcopy(model_d)
        mu_params_p = [p for n, p in probe.named_parameters() if 'mu' in n]
        torch.manual_seed(7)
        probe.zero_grad()
        loss_b = torch.nn.functional.cross_entropy(probe.forward_logits(x, t=1, sample=False), y_buf)
        loss_b.backward()
        h_p = [p.grad.detach().clone() for p in mu_params_p]

        probe.zero_grad()
        loss_c = torch.nn.functional.cross_entropy(probe.forward_logits(x, t=1, sample=False), y_curr)
        loss_c.backward()
        g_p = [p.grad.detach().clone() for p in mu_params_p]

        hg = sum((hi * gi).sum() for hi, gi in zip(h_p, g_p))
        hh = sum((hi * hi).sum() for hi in h_p)
        g_raw_proj = [gi - (hg / hh) * hi for hi, gi in zip(h_p, g_p)]

        # Assign heterogeneous D: large lr where h * g_raw_proj < 0 and small lr where >= 0
        lr_dict = {}
        mu_names = [n for n, _ in model_d.named_parameters() if 'mu' in n]
        rho_names = [n for n, _ in model_d.named_parameters() if 'rho' in n]
        for name, hi, gi, gp in zip(mu_names, h_p, g_p, g_raw_proj):
            neg_mask = (hi * gp) < 0.0
            a = torch.where(neg_mask, torch.full_like(hi, 0.2), torch.full_like(hi, 1e-4))
            lr_dict[name] = a
        for name in rho_names:
            lr_dict[name] = 0.05

        torch.manual_seed(123)
        res_d = train_step_idea2(
            model=model_d,
            lr_dict=lr_dict,
            x_curr=x,
            y_curr=y_curr,
            n_train_examples=50000,
            buffer=buf,
            mode='ucb_d_proj',
            base_lr=0.05,
            n_samples=1,
            t=1,
        )

        torch.manual_seed(123)
        res_raw = train_step_idea2(
            model=model_raw,
            lr_dict=lr_dict,
            x_curr=x,
            y_curr=y_curr,
            n_train_examples=50000,
            buffer=buf,
            mode='ucb_raw_proj',
            base_lr=0.05,
            n_samples=1,
            t=1,
        )

        self.assertEqual(res_d['conflict'], 1)
        self.assertTrue(res_d['has_post_check'])
        self.assertLess(abs(res_d['post_hd']), 1e-6)
        self.assertEqual(res_d['post_violation'], 0)

        self.assertTrue(res_raw['has_post_check'])
        self.assertLess(res_raw['post_hd'], -1e-7)
        self.assertEqual(res_raw['post_violation'], 1)

    def test_ucb_identical_with_or_without_diag_buffer(self):
        torch.manual_seed(42)
        args = self._make_args(mode='ucb')
        m1 = Net(args).to('cpu')
        m2 = copy.deepcopy(m1)
        appr1 = Appr(m1, args)
        appr2 = Appr(m2, args)

        x0 = torch.randn(16, 1, 4, 4)
        y0 = torch.arange(16) % 4
        x1 = torch.randn(16, 1, 4, 4)
        y1 = (torch.arange(16) + 1) % 4

        diag_buf = BalancedReplayBuffer(capacity=16, device='cpu')
        diag_buf.update_after_task(0, x0, y0, seed=0)

        torch.manual_seed(99)
        res_no_diag = train_step_idea2(
            model=m1,
            lr_dict=appr1.lr_dict,
            x_curr=x1,
            y_curr=y1,
            n_train_examples=16,
            buffer=appr1.buffer,
            mode='ucb',
            base_lr=args.lr,
            n_samples=2,
            t=1,
            diag_buffer=None,
            measure_diagnostics=False,
            measure_scale_interference=False,
        )

        torch.manual_seed(99)
        res_with_diag = train_step_idea2(
            model=m2,
            lr_dict=appr2.lr_dict,
            x_curr=x1,
            y_curr=y1,
            n_train_examples=16,
            buffer=appr2.buffer,
            mode='ucb',
            base_lr=args.lr,
            n_samples=2,
            t=1,
            diag_buffer=diag_buf,
            measure_diagnostics=True,
            measure_scale_interference=True,
            step_idx=0,
        )

        self.assertTrue(res_with_diag['has_diag'])
        self.assertTrue(res_with_diag['has_scale_check'])
        self.assertFalse(res_no_diag['has_diag'])
        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            self.assertTrue(torch.equal(p1, p2), msg=f"Parameter {n1} differed when diag_buffer was provided!")

    def test_2nd_order_d_projection_reduces_buffer_loss_drift(self):
        """
        Verify ucb_d_proj_2nd triggers projection on a conflicting finite step, achieves
        0 post-step violations (h^T d* >= -1e-7), and leaves lower buffer loss than unprojected ucb.
        """
        torch.manual_seed(0)
        args = self._make_args(mode='ucb_d_proj_2nd')
        args.rho = -8.0
        model_2nd = Net(args).to('cpu')
        model_ucb = copy.deepcopy(model_2nd)
        appr_2nd = Appr(model_2nd, args)
        appr_ucb = Appr(model_ucb, args)

        x = torch.randn(8, 1, 4, 4)
        y_buf = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
        y_curr = torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], dtype=torch.long)

        buf = BalancedReplayBuffer(capacity=8, device='cpu')
        buf.update_after_task(0, x, y_buf, seed=0)

        torch.manual_seed(77)
        res_2nd = train_step_idea2(
            model=model_2nd,
            lr_dict=appr_2nd.lr_dict,
            x_curr=x,
            y_curr=y_curr,
            n_train_examples=50000,
            buffer=buf,
            mode='ucb_d_proj_2nd',
            base_lr=args.lr,
            n_samples=1,
            t=1,
            step_idx=5,
        )

        torch.manual_seed(77)
        train_step_idea2(
            model=model_ucb,
            lr_dict=appr_ucb.lr_dict,
            x_curr=x,
            y_curr=y_curr,
            n_train_examples=50000,
            buffer=None,
            mode='ucb',
            base_lr=args.lr,
            n_samples=1,
            t=1,
            step_idx=5,
        )

        self.assertEqual(res_2nd['conflict'], 1)
        self.assertTrue(res_2nd['has_post_check'])
        self.assertTrue(res_2nd['has_2nd_order_check'])
        self.assertEqual(res_2nd['post_violation'], 0)
        self.assertGreaterEqual(res_2nd['post_hd'], -1e-7)

        with torch.no_grad():
            loss_2nd = torch.nn.functional.cross_entropy(model_2nd.forward_logits(x, t=1, sample=False), y_buf).item()
            loss_ucb = torch.nn.functional.cross_entropy(model_ucb.forward_logits(x, t=1, sample=False), y_buf).item()
        self.assertLess(loss_2nd, loss_ucb)

    def test_all_seven_modes_run_on_cpu_and_report_memory(self):
        self.assertGreater(get_peak_cpu_memory_mb(), 0.0)
        for mode in ['ucb', 'ucb_replay', 'bft_replay', 'agem', 'ucb_raw_proj', 'ucb_d_proj', 'ucb_d_proj_2nd']:
            args = self._make_args(mode=mode)
            model = Net(args).to(args.device)
            appr = Appr(model, args)

            x0 = torch.randn(16, 1, 4, 4)
            y0 = torch.arange(16) % 4
            appr.train(0, x0, y0, x0, y0)

            if mode != 'ucb':
                self.assertEqual(len(appr.buffer), 16)
            else:
                self.assertEqual(len(appr.buffer), 0)

            x1 = torch.randn(16, 1, 4, 4)
            y1 = torch.arange(16) % 4
            appr.train(1, x1, y1, x1, y1)

            loss, acc = appr.eval(0, x0, y0)
            self.assertTrue(np.isfinite(loss))
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)
            self.assertIn('lr_d_spread', appr.task_diagnostics[1])

    def test_cl_metrics_formulas(self):
        R = np.array([
            [0.90, 0.50, 0.40],
            [0.80, 0.92, 0.45],
            [0.75, 0.86, 0.94],
        ])
        m = compute_cl_metrics(R)
        self.assertAlmostEqual(m['A_T'], (0.75 + 0.86 + 0.94) / 3.0, places=8)
        self.assertAlmostEqual(m['BWT'], ((0.75 - 0.90) + (0.86 - 0.92)) / 2.0, places=8)
        self.assertAlmostEqual(m['Forgetting'], ((0.90 - 0.75) + (0.92 - 0.86)) / 2.0, places=8)


if __name__ == '__main__':
    unittest.main()
