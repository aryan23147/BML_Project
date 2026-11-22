import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from approaches.ucb_idea2 import Appr
from metrics import get_peak_cpu_memory_mb, get_peak_gpu_memory_mb
from networks.mlp_ucb import Net as BayesianMLPNet


def load_dataset(args):
    if args.experiment == 'rmnist10':
        from dataloaders import rmnist as dataloader
        ntasks = args.num_tasks if args.num_tasks > 0 else 5
        return dataloader.get(
            data_path=args.data_path,
            seed=args.seed,
            angle_gap=10,
            reverse=args.reverse,
            num_tasks=ntasks,
        )
    if args.experiment == 'rmnist30':
        from dataloaders import rmnist as dataloader
        ntasks = args.num_tasks if args.num_tasks > 0 else 5
        return dataloader.get(
            data_path=args.data_path,
            seed=args.seed,
            angle_gap=30,
            reverse=args.reverse,
            num_tasks=ntasks,
        )
    if args.experiment == 'pmnist':
        from dataloaders import pmnist as dataloader
        nperm = args.num_tasks if args.num_tasks > 0 else 10
        return dataloader.get(
            data_path=args.data_path,
            seed=args.seed,
            fixed_order=True,
            pc_valid=0.1,
            nperm=nperm,
            reverse=args.reverse,
        )
    raise ValueError(f"Unsupported experiment: {args.experiment}")


def main():
    parser = argparse.ArgumentParser(description='Experiment 2: From-Scratch Single-Task Transfer Reference')
    parser.add_argument('--experiment', required=True, choices=['rmnist10', 'rmnist30', 'pmnist'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--num_tasks', type=int, default=0)
    parser.add_argument('--tag', type=str, default='', help='Optional filename tag to prevent colliding with default outputs')
    parser.add_argument('--nepochs', type=int, default=5)
    parser.add_argument('--sbatch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--nhid', type=int, default=200)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--eval_samples', type=int, default=5)
    parser.add_argument('--rho', type=float, default=-3.0)
    parser.add_argument('--sig1', type=float, default=0.0)
    parser.add_argument('--sig2', type=float, default=6.0)
    parser.add_argument('--pi', type=float, default=0.25)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--data_path', type=str, default=os.path.join(os.path.dirname(__file__), 'data'))
    parser.add_argument(
        '--out_dir',
        type=str,
        default=os.path.join(os.path.dirname(__file__), '..', 'results', 'idea2_reports', 'json'),
    )
    args = parser.parse_args()

    if 'UCB_THREADS' in os.environ:
        torch.set_num_threads(int(os.environ['UCB_THREADS']))

    os.makedirs(args.out_dir, exist_ok=True)
    t_total_0 = time.perf_counter()

    data, taskcla, inputsize = load_dataset(args)
    order_tag = 'rev' if args.reverse else 'fwd'

    print('=' * 96)
    print(f"FROM-SCRATCH REFERENCE: experiment={args.experiment} ({order_tag}) | seed={args.seed} | nepochs={args.nepochs}")
    print('=' * 96)

    tasks_out = {}
    for t, ncla in taskcla:
        t_task_0 = time.perf_counter()

        # Re-seed identically before each standalone domain so initialization matches Task 0 of continual runs
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available() and 'cuda' in args.device:
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        run_args = SimpleNamespace(
            seed=args.seed,
            device=args.device,
            experiment=args.experiment,
            approach='ucb_idea2',
            mode='ucb',
            reverse=args.reverse,
            buffer_size=0,
            replay_alpha=0.0,
            num_tasks=1,
            inputsize=inputsize,
            taskcla=[(0, ncla)],
            nepochs=args.nepochs,
            sbatch=args.sbatch,
            lr=args.lr,
            nlayers=args.nlayers,
            nhid=args.nhid,
            shared_head=True,
            samples=args.samples,
            eval_samples=args.eval_samples,
            rho=args.rho,
            sig1=args.sig1,
            sig2=args.sig2,
            pi=args.pi,
            arch='mlp',
            checkpoint=args.out_dir,
        )

        model = BayesianMLPNet(run_args).to(args.device)
        appr = Appr(model, run_args)
        appr.save_model = lambda _t: None
        appr.diag_buffer = None

        xtrain = data[t]['train']['x'].to(args.device)
        ytrain = data[t]['train']['y'].to(args.device)
        xvalid = data[t]['valid']['x'].to(args.device)
        yvalid = data[t]['valid']['y'].to(args.device)
        xtest = data[t]['test']['x'].to(args.device)
        ytest = data[t]['test']['y'].to(args.device)

        print(f"\n--- Standalone Task {t} ({data[t]['name']}) ---")
        appr.train(0, xtrain, ytrain, xvalid, yvalid)
        test_loss, test_acc = appr.eval(0, xtest, ytest, debug=False)
        task_runtime = time.perf_counter() - t_task_0

        auc = float(appr.task_diagnostics[0]['learning_curve_auc'])
        epoch_accs = [float(v) for v in appr.learning_curves[0]]

        print(
            f">>> From-Scratch Task {t} ({data[t]['name']}): "
            f"test_acc={100.0 * test_acc:5.2f}% | LC-AUC={100.0 * auc:5.2f}% | time={task_runtime:.1f}s"
        )

        tasks_out[str(t)] = {
            'task_id': int(t),
            'name': data[t]['name'],
            'learning_curve_auc': auc,
            'epoch_valid_accs': epoch_accs,
            'test_acc': float(test_acc),
            'test_loss': float(test_loss),
            'runtime_sec': float(task_runtime),
        }

    total_runtime = time.perf_counter() - t_total_0
    mean_auc_all = float(np.mean([v['learning_curve_auc'] for v in tasks_out.values()]))
    mean_auc_later = float(
        np.mean([tasks_out[str(k)]['learning_curve_auc'] for k in range(1, len(taskcla))])
    ) if len(taskcla) > 1 else mean_auc_all
    mean_test_acc = float(np.mean([v['test_acc'] for v in tasks_out.values()]))

    payload = {
        'experiment': args.experiment,
        'tag': args.tag if args.tag else args.experiment,
        'num_tasks': len(taskcla),
        'reverse': bool(args.reverse),
        'seed': int(args.seed),
        'nepochs': int(args.nepochs),
        'sbatch': int(args.sbatch),
        'lr': float(args.lr),
        'nhid': int(args.nhid),
        'nlayers': int(args.nlayers),
        'samples': int(args.samples),
        'eval_samples': int(args.eval_samples),
        'mean_learning_curve_auc_all': mean_auc_all,
        'mean_learning_curve_auc_tasks_1_to_T': mean_auc_later,
        'mean_test_acc': mean_test_acc,
        'total_runtime_sec': float(total_runtime),
        'peak_cpu_memory_mb': get_peak_cpu_memory_mb(),
        'peak_gpu_memory_mb': get_peak_gpu_memory_mb(),
        'tasks': tasks_out,
    }

    file_tag = args.tag if args.tag else args.experiment
    out_path = os.path.join(args.out_dir, f'from_scratch_{file_tag}_{order_tag}_seed{args.seed}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print('\n' + '=' * 96)
    print(f"Saved From-Scratch Reference JSON to: {out_path}")
    print(f"Mean LC-AUC (all tasks): {100.0 * mean_auc_all:5.2f}% | Mean Test Acc: {100.0 * mean_test_acc:5.2f}% | Time: {total_runtime:.1f}s")
    print('=' * 96)


if __name__ == '__main__':
    main()
