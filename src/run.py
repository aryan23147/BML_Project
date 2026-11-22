import sys, os, argparse, time, json
import numpy as np
import torch
import utils
from datetime import datetime
from metrics import compute_cl_metrics, get_peak_gpu_memory_mb, get_peak_cpu_memory_mb

tstart = time.time()

# Arguments
parser = argparse.ArgumentParser(description='UCB & Idea 2 Continual Learning')
parser.add_argument('--seed',               default=0,              type=int,   help='(default=%(default)d)')
parser.add_argument('--device',             default='cuda:0',       type=str,   help='gpu id or cpu')
parser.add_argument('--experiment',         default='mnist5',       type=str,   required=True,
                                            choices=['mnist2', 'mnist5', 'pmnist', 'rmnist10', 'rmnist30', 'cifar', 'mixture'])
parser.add_argument('--approach',           default='ucb',          type=str,   help='ucb or ucb_idea2')
parser.add_argument('--mode',               default='ucb_d_proj',   type=str,
                                            choices=['ucb', 'ucb_replay', 'agem', 'ucb_d_proj', 'ucb_d_proj_2nd', 'ucb_raw_proj', 'bft_replay'],
                                            help='Idea 2 method mode when --approach ucb_idea2')
parser.add_argument('--reverse',            action='store_true',                help='Reverse task stream order')
parser.add_argument('--buffer_size',        default=200,            type=int,   help='Replay buffer capacity (default=200)')
parser.add_argument('--replay_alpha',       default=0.5,            type=float, help='Replay gradient weight for replay baselines')
parser.add_argument('--num_tasks',          default=0,              type=int,   help='Override number of tasks (0 = default)')
parser.add_argument('--data_path',          default='../data/',     type=str,   help='data directory')

# Training parameters
parser.add_argument('--output',             default='',             type=str,   help='')
parser.add_argument('--checkpoint_dir',     default='../checkpoints/', type=str, help='')
parser.add_argument('--nepochs',            default=200,            type=int,   help='')
parser.add_argument('--sbatch',             default=64,             type=int,   help='')
parser.add_argument('--lr',                 default=0.01,           type=float, help='')
parser.add_argument('--nlayers',            default=2,              type=int,   help='')
parser.add_argument('--nhid',               default=200,            type=int,   help='')
parser.add_argument('--shared_head',        action='store_true',                help='Use single shared output head')
parser.add_argument('--parameter',          default='',             type=str,   help='')

# UCB HYPER-PARAMETERS
parser.add_argument('--samples',            default=5,              type=int,   help='Number of Monte Carlo samples in training')
parser.add_argument('--eval_samples',       default=10,             type=int,   help='Number of Monte Carlo samples in evaluation')
parser.add_argument('--rho',                default=-3.0,           type=float, help='Initial rho')
parser.add_argument('--sig1',               default=0.0,            type=float, help='STD for the 1st prior pdf in scaled mixture Gaussian')
parser.add_argument('--sig2',               default=6.0,            type=float, help='STD for the 2nd prior pdf in scaled mixture Gaussian')
parser.add_argument('--pi',                 default=0.25,           type=float, help='weighting factor for prior')
parser.add_argument('--arch',               default='mlp',          type=str,   help='Bayesian Neural Network architecture')

parser.add_argument('--resume',             default='no',           type=str,   help='resume?')
parser.add_argument('--sti',                default=0,              type=int,   help='starting task?')

args = parser.parse_args()

# Default to shared 10-class head for Domain-IL benchmarks in Idea 2
if args.experiment in ('rmnist10', 'rmnist30', 'pmnist') and args.approach == 'ucb_idea2':
    args.shared_head = True

if args.output == '' and args.approach == 'ucb_idea2':
    order_tag = 'rev' if args.reverse else 'fwd'
    args.output = f'{args.experiment}_{order_tag}_{args.mode}'

utils.print_arguments(args)

########################################################################################################################

# Seed
if 'UCB_THREADS' in os.environ:
    torch.set_num_threads(int(os.environ['UCB_THREADS']))
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and 'cuda' in args.device:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print('Using device:', args.device)
checkpoint = utils.make_directories(args)
args.checkpoint = checkpoint
print()

# Args -- Experiment
if args.experiment == 'mnist2':
    from dataloaders import mnist2 as dataloader
    data, taskcla, inputsize = dataloader.get(data_path=args.data_path, seed=args.seed)
elif args.experiment == 'mnist5':
    from dataloaders import mnist5 as dataloader
    data, taskcla, inputsize = dataloader.get(data_path=args.data_path, seed=args.seed)
elif args.experiment == 'pmnist':
    from dataloaders import pmnist as dataloader
    nperm = args.num_tasks if args.num_tasks > 0 else 10
    data, taskcla, inputsize = dataloader.get(
        data_path=args.data_path, seed=args.seed, fixed_order=True, pc_valid=0.1, nperm=nperm, reverse=args.reverse
    )
elif args.experiment == 'rmnist10':
    from dataloaders import rmnist as dataloader
    ntasks = args.num_tasks if args.num_tasks > 0 else 5
    data, taskcla, inputsize = dataloader.get(
        data_path=args.data_path, seed=args.seed, angle_gap=10, reverse=args.reverse, num_tasks=ntasks
    )
elif args.experiment == 'rmnist30':
    from dataloaders import rmnist as dataloader
    ntasks = args.num_tasks if args.num_tasks > 0 else 5
    data, taskcla, inputsize = dataloader.get(
        data_path=args.data_path, seed=args.seed, angle_gap=30, reverse=args.reverse, num_tasks=ntasks
    )
elif args.experiment == 'cifar':
    from dataloaders import cifar as dataloader
    data, taskcla, inputsize = dataloader.get(data_path=args.data_path, seed=args.seed)
elif args.experiment == 'mixture':
    from dataloaders import mixture as dataloader
    data, taskcla, inputsize = dataloader.get(data_path=args.data_path, seed=args.seed)

# Args -- Approach
if args.approach == 'ucb':
    from approaches import ucb as approach
elif args.approach == 'ucb_idea2':
    from approaches import ucb_idea2 as approach
else:
    raise ValueError(f"Unknown approach: {args.approach}")

# Args -- Network
if args.experiment in ('mnist2', 'pmnist', 'mnist5', 'rmnist10', 'rmnist30'):
    from networks import mlp_ucb as network
else:
    from networks import resnet_ucb as network

########################################################################################################################
print()
print("Starting this run on :")
print(datetime.now().strftime("%Y-%m-%d %H:%M"))

print('Input size =', inputsize, '\nTask info =', taskcla)
args.num_tasks = len(taskcla)
args.inputsize, args.taskcla = inputsize, taskcla

# Inits
print('Inits...')
model = network.Net(args).to(args.device)

print('-' * 100)
appr = approach.Appr(model, args=args)
print('-' * 100)

if args.resume == 'yes':
    ckpt = torch.load(os.path.join(args.checkpoint, 'model_{}.pth.tar'.format(args.sti)))
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device=args.device)
else:
    args.sti = 0

# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
for t, ncla in taskcla[args.sti:]:
    print('*' * 100)
    print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    xtrain = data[t]['train']['x'].to(args.device)
    ytrain = data[t]['train']['y'].to(args.device)
    xvalid = data[t]['valid']['x'].to(args.device)
    yvalid = data[t]['valid']['y'].to(args.device)

    # Train
    appr.train(t, xtrain, ytrain, xvalid, yvalid)
    print('-' * 100)

    # Test on all tasks seen so far
    for u in range(t + 1):
        xtest = data[u]['test']['x'].to(args.device)
        ytest = data[u]['test']['y'].to(args.device)
        test_loss, test_acc = appr.eval(u, xtest, ytest, debug=True)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.2f}% <<<'.format(
            u, data[u]['name'], test_loss, 100 * test_acc
        ))
        acc[t, u] = test_acc
        lss[t, u] = test_loss

    # Save accuracy matrix
    print('Save at ' + args.checkpoint)
    tag = f"{args.experiment}_{args.mode}_{args.seed}" if args.approach == 'ucb_idea2' else f"{args.experiment}_{args.approach}_{args.seed}"
    np.savetxt(os.path.join(args.checkpoint, f'{tag}.txt'), acc, '%.5f')

utils.print_log_acc_bwt(args, acc, lss)
elapsed_sec = time.time() - tstart

if args.approach == 'ucb_idea2':
    cl_metrics = compute_cl_metrics(acc)
    later_tasks = [k for k in appr.task_diagnostics.keys() if k > 0]
    mean_cos = float(np.mean([appr.task_diagnostics[k]['mean_cos_sim'] for k in later_tasks])) if later_tasks else 0.0
    mean_conf = float(np.mean([appr.task_diagnostics[k]['conflict_rate'] for k in later_tasks])) if later_tasks else 0.0
    post_tasks = [k for k in later_tasks if appr.task_diagnostics[k]['post_check_count'] > 0]
    mean_post_viol = float(np.mean([appr.task_diagnostics[k]['post_proj_violation_rate'] for k in post_tasks])) if post_tasks else 0.0
    mean_post_hd = float(np.mean([appr.task_diagnostics[k]['mean_post_proj_hd'] for k in post_tasks])) if post_tasks else 0.0
    scale_tasks = [k for k in later_tasks if appr.task_diagnostics[k]['scale_check_count'] > 0]
    mean_scale_delta = float(np.mean([appr.task_diagnostics[k]['mean_scale_interference_delta_loss'] for k in scale_tasks])) if scale_tasks else 0.0
    mean_scale_rate = float(np.mean([appr.task_diagnostics[k]['scale_interference_rate'] for k in scale_tasks])) if scale_tasks else 0.0
    second_tasks = [k for k in later_tasks if appr.task_diagnostics[k].get('second_order_check_count', 0) > 0]
    mean_curv_margin = float(np.mean([appr.task_diagnostics[k]['mean_curvature_margin'] for k in second_tasks])) if second_tasks else 0.0
    mean_finite_conf = float(np.mean([appr.task_diagnostics[k]['finite_step_conflict_rate'] for k in second_tasks])) if second_tasks else 0.0
    mean_auc = float(np.mean([appr.task_diagnostics[k]['learning_curve_auc'] for k in appr.task_diagnostics]))
    train_runtime_sec = float(max(0.0, elapsed_sec - getattr(appr, 'total_diag_overhead_sec', 0.0)))

    summary = {
        'experiment': args.experiment,
        'reverse': bool(args.reverse),
        'mode': args.mode,
        'seed': int(args.seed),
        'nepochs': int(args.nepochs),
        'lr': float(args.lr),
        'buffer_size': int(args.buffer_size),
        'replay_alpha': float(args.replay_alpha),
        'A_T': cl_metrics['A_T'],
        'BWT': cl_metrics['BWT'],
        'Forgetting': cl_metrics['Forgetting'],
        'mean_cos_sim_tasks_1_to_T': mean_cos,
        'mean_conflict_rate_tasks_1_to_T': mean_conf,
        'mean_post_proj_violation_rate': mean_post_viol,
        'mean_post_proj_hd': mean_post_hd,
        'mean_scale_interference_delta_loss': mean_scale_delta,
        'scale_interference_rate': mean_scale_rate,
        'mean_curvature_margin': mean_curv_margin,
        'mean_finite_step_conflict_rate': mean_finite_conf,
        'mean_learning_curve_auc': mean_auc,
        'runtime_sec': float(elapsed_sec),
        'train_runtime_sec': train_runtime_sec,
        'diag_overhead_sec': float(getattr(appr, 'total_diag_overhead_sec', 0.0)),
        'peak_cpu_memory_mb': get_peak_cpu_memory_mb(),
        'peak_gpu_memory_mb': get_peak_gpu_memory_mb(),
        'R_matrix': acc.tolist(),
        'task_diagnostics': appr.task_diagnostics,
    }
    json_path = os.path.join(args.checkpoint, f'metrics_{args.experiment}_{args.mode}_seed{args.seed}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Idea 2 JSON summary saved to: {json_path}")

print('[Elapsed time = {:.1f} s ({:.2f} h)]'.format(elapsed_sec, elapsed_sec / 3600.0))
