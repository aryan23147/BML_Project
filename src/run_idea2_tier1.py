import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
import time
import numpy as np


TIER1_MODES = ['ucb', 'ucb_replay', 'agem', 'ucb_d_proj']


def run_single(
    experiment,
    mode,
    seed,
    nepochs,
    sbatch,
    lr,
    samples,
    eval_samples,
    device,
    reverse,
    data_path,
    checkpoint_dir,
    threads_per_job=3,
):
    order_tag = 'rev' if reverse else 'fwd'
    out_name = f"{experiment}_{order_tag}_{mode}_s{seed}_ep{nepochs}"
    json_path = os.path.join(checkpoint_dir, out_name, f"metrics_{experiment}_{mode}_seed{seed}.json")

    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return experiment, mode, seed, json.load(f)

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'run.py'),
        '--experiment', experiment,
        '--approach', 'ucb_idea2',
        '--mode', mode,
        '--seed', str(seed),
        '--nepochs', str(nepochs),
        '--sbatch', str(sbatch),
        '--lr', str(lr),
        '--nhid', '200',
        '--nlayers', '2',
        '--samples', str(samples),
        '--eval_samples', str(eval_samples),
        '--device', device,
        '--data_path', data_path,
        '--checkpoint_dir', checkpoint_dir,
        '--output', out_name,
    ]
    if reverse:
        cmd.append('--reverse')

    env = os.environ.copy()
    env['UCB_THREADS'] = str(threads_per_job)
    env['OMP_NUM_THREADS'] = str(threads_per_job)
    env['MKL_NUM_THREADS'] = str(threads_per_job)

    print(f"[START] {experiment} | mode={mode} | seed={seed} | nepochs={nepochs}", flush=True)
    t_job = time.time()
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)
    print(f"[DONE]  {experiment} | mode={mode} | seed={seed} in {time.time() - t_job:.1f}s", flush=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        return experiment, mode, seed, json.load(f)


def print_summary_table(results_by_mode, experiment, reverse):
    order_str = "Reverse" if reverse else "Forward"
    print("\n" + "=" * 108)
    print(f"IDEA 2 TIER-1 SUMMARY: {experiment} ({order_str})")
    print("=" * 108)
    header = (
        f"{'Method':<14} | {'A_T (%)':>12} | {'BWT (%)':>12} | {'Forget (%)':>12} | "
        f"{'cos(g,h)':>10} | {'Conflict (%)':>12} | {'LC-AUC (%)':>11} | {'Time (s)':>9}"
    )
    print(header)
    print("-" * 108)

    for mode in TIER1_MODES:
        runs = results_by_mode.get(mode, [])
        if not runs:
            continue
        a_t = 100.0 * np.array([r['A_T'] for r in runs])
        bwt = 100.0 * np.array([r['BWT'] for r in runs])
        fgt = 100.0 * np.array([r['Forgetting'] for r in runs])
        cos = np.array([r['mean_cos_sim_tasks_1_to_T'] for r in runs])
        cnf = 100.0 * np.array([r['mean_conflict_rate_tasks_1_to_T'] for r in runs])
        auc = 100.0 * np.array([r['mean_learning_curve_auc'] for r in runs])
        rt = np.array([r['runtime_sec'] for r in runs])

        print(
            f"{mode:<14} | {a_t.mean():5.2f}±{a_t.std():4.2f} | "
            f"{bwt.mean():+5.2f}±{bwt.std():4.2f} | "
            f"{fgt.mean():5.2f}±{fgt.std():4.2f} | "
            f"{cos.mean():+8.4f}  | "
            f"{cnf.mean():5.1f}±{cnf.std():4.1f} | "
            f"{auc.mean():5.2f}±{auc.std():4.2f} | "
            f"{rt.mean():8.1f}s"
        )
    print("=" * 108)


def main():
    parser = argparse.ArgumentParser(description='Run Tier-1 Idea 2 Baselines & Proposed Method')
    parser.add_argument('--experiments', nargs='+', default=['rmnist10', 'rmnist30'])
    parser.add_argument('--modes', nargs='+', default=TIER1_MODES)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--nepochs', type=int, default=10)
    parser.add_argument('--sbatch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--eval_samples', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--data_path', type=str, default=os.path.join(os.path.dirname(__file__), 'data'))
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
    args = parser.parse_args()

    all_summaries = {exp: {m: [] for m in args.modes} for exp in args.experiments}
    t0 = time.time()

    jobs = []
    for exp in args.experiments:
        for seed in args.seeds:
            for mode in args.modes:
                jobs.append((exp, mode, seed))

    threads_per_job = max(1, (os.cpu_count() or 8) // max(1, args.workers))
    print(f"Launching {len(jobs)} jobs across {args.workers} parallel workers ({threads_per_job} CPU threads/worker)...")

    completed_results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(
                run_single,
                exp,
                mode,
                seed,
                args.nepochs,
                args.sbatch,
                args.lr,
                args.samples,
                args.eval_samples,
                args.device,
                args.reverse,
                args.data_path,
                args.checkpoint_dir,
                threads_per_job,
            ): (exp, mode, seed)
            for (exp, mode, seed) in jobs
        }
        for fut in as_completed(future_map):
            exp, mode, seed, res = fut.result()
            completed_results[(exp, mode, seed)] = res

    for exp in args.experiments:
        for mode in args.modes:
            for seed in args.seeds:
                all_summaries[exp][mode].append(completed_results[(exp, mode, seed)])
        print_summary_table(all_summaries[exp], exp, args.reverse)

    out_summary = os.path.join(args.checkpoint_dir, 'idea2_tier1_summary.json')
    with open(out_summary, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved combined Tier-1 summary to: {out_summary} (Total wall-clock elapsed: {time.time() - t0:.1f}s)")


if __name__ == '__main__':
    main()
