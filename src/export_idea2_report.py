import argparse
import csv
import json
import os
import tempfile

if 'MPLCONFIGDIR' not in os.environ:
    os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'mplconfig_ucb')
    os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


ALL_METHODS = ['ucb', 'bft_replay', 'ucb_replay', 'agem', 'ucb_raw_proj', 'ucb_d_proj', 'ucb_d_proj_2nd']
EXP1_MODES = ['ucb', 'ucb_raw_proj', 'ucb_d_proj', 'ucb_d_proj_2nd']

DISPLAY_NAMES = {
    'ucb': 'UCB (Replay-Free)',
    'bft_replay': 'BFT + Replay (200)',
    'ucb_replay': 'UCB + Replay (200)',
    'agem': 'A-GEM (200)',
    'ucb_raw_proj': 'UCB + Raw Proj (200)',
    'ucb_d_proj': 'UCB + D-Proj (1st-Order)',
    'ucb_d_proj_2nd': 'UCB + D-Proj (2nd-Order*)',
}

SHORT_LABELS = {
    'ucb': 'UCB\n(No Buf)',
    'bft_replay': 'BFT\nReplay',
    'ucb_replay': 'UCB\nReplay',
    'agem': 'A-GEM\n(Euclid)',
    'ucb_raw_proj': 'UCB\nRaw-Proj',
    'ucb_d_proj': 'UCB\nD-Proj 1st',
    'ucb_d_proj_2nd': 'UCB\nD-Proj 2nd*',
}

COLOR_MAP = {
    'ucb': '#6c757d',
    'bft_replay': '#ff9896',
    'ucb_replay': '#2ca02c',
    'agem': '#ff7f0e',
    'ucb_raw_proj': '#9467bd',
    'ucb_d_proj': '#4c9be8',
    'ucb_d_proj_2nd': '#0b4f8a',
}


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _available_modes(checkpoints_dir, candidate_modes, experiment, seed, run_prefix=''):
    avail = []
    for mode in candidate_modes:
        run_dir = f'{run_prefix}_{mode}_s{seed}' if run_prefix else f'{experiment}_exp1_{mode}_s{seed}'
        m_path = os.path.join(checkpoints_dir, run_dir, f'metrics_{experiment}_{mode}_seed{seed}.json')
        if os.path.exists(m_path):
            avail.append((mode, m_path))
    return avail


def build_combined_summary(checkpoints_dir, scratch_path, experiment='rmnist10', seed=0, run_prefix='', reverse=False):
    scratch = load_json(scratch_path)
    scratch_tasks = scratch['tasks']
    num_tasks = len(scratch_tasks)

    avail = _available_modes(checkpoints_dir, ALL_METHODS, experiment, seed, run_prefix=run_prefix)
    methods = {}
    for mode, m_path in avail:
        m = load_json(m_path)
        td = m['task_diagnostics']
        R = m['R_matrix']

        per_task = {}
        delta_aucs_1_to_T = []
        delta_aucs_all = []
        for t in range(num_tasks):
            t_key = str(t)
            cl_auc = float(td[t_key]['learning_curve_auc'])
            sc_auc = float(scratch_tasks[t_key]['learning_curve_auc'])
            d_auc = cl_auc - sc_auc
            delta_aucs_all.append(d_auc)
            if t > 0:
                delta_aucs_1_to_T.append(d_auc)

            per_task[t_key] = {
                'task_name': scratch_tasks[t_key]['name'],
                'cl_learning_curve_auc': cl_auc,
                'scratch_learning_curve_auc': sc_auc,
                'delta_auc': float(d_auc),
                'immediate_test_acc': float(R[t][t]),
                'final_test_acc': float(R[num_tasks - 1][t]),
                'scratch_test_acc': float(scratch_tasks[t_key]['test_acc']),
                'mean_cos_sim': float(td[t_key]['mean_cos_sim']),
                'conflict_rate': float(td[t_key]['conflict_rate']),
                'post_proj_violation_rate': float(td[t_key]['post_proj_violation_rate']),
                'mean_post_proj_hd': float(td[t_key]['mean_post_proj_hd']),
                'mean_scale_interference_delta_loss': float(td[t_key]['mean_scale_interference_delta_loss']),
                'scale_interference_rate': float(td[t_key]['scale_interference_rate']),
                'mean_curvature_margin': float(td[t_key].get('mean_curvature_margin', 0.0)),
                'finite_step_conflict_rate': float(td[t_key].get('finite_step_conflict_rate', 0.0)),
                'lr_d_max_min_ratio': float(td[t_key]['lr_d_spread']['max_min_ratio']),
                'buffer_size_after_task': int(td[t_key]['buffer_size_after_task']),
            }

        buf_regime = 'Replay-Free (0)' if mode == 'ucb' else '200-Example Buffer'
        methods[mode] = {
            'mode': mode,
            'display_name': DISPLAY_NAMES[mode],
            'memory_regime': buf_regime,
            'buffer_size': 0 if mode == 'ucb' else int(m['buffer_size']),
            'A_T': float(m['A_T']),
            'BWT': float(m['BWT']),
            'Forgetting': float(m['Forgetting']),
            'mean_learning_curve_auc': float(m['mean_learning_curve_auc']),
            'mean_forward_transfer_delta_auc_tasks_1_to_T': float(np.mean(delta_aucs_1_to_T)),
            'mean_forward_transfer_delta_auc_all': float(np.mean(delta_aucs_all)),
            'mean_cos_sim_tasks_1_to_T': float(m['mean_cos_sim_tasks_1_to_T']),
            'mean_conflict_rate_tasks_1_to_T': float(m['mean_conflict_rate_tasks_1_to_T']),
            'mean_post_proj_violation_rate': float(m['mean_post_proj_violation_rate']),
            'mean_post_proj_hd': float(m['mean_post_proj_hd']),
            'mean_scale_interference_delta_loss': float(m['mean_scale_interference_delta_loss']),
            'scale_interference_rate': float(m['scale_interference_rate']),
            'mean_curvature_margin': float(m.get('mean_curvature_margin', 0.0)),
            'mean_finite_step_conflict_rate': float(m.get('mean_finite_step_conflict_rate', 0.0)),
            'lr_d_max_min_ratio_final': float(td[str(num_tasks - 1)]['lr_d_spread']['max_min_ratio']),
            'runtime_sec': float(m['runtime_sec']),
            'train_runtime_sec': float(m['train_runtime_sec']),
            'diag_overhead_sec': float(m['diag_overhead_sec']),
            'peak_cpu_memory_mb': float(m['peak_cpu_memory_mb']),
            'peak_gpu_memory_mb': float(m['peak_gpu_memory_mb']),
            'R_matrix': R,
            'per_task': per_task,
        }

    return {
        'experiment': experiment,
        'order': 'reverse' if reverse else 'forward',
        'num_tasks': int(num_tasks),
        'seed': int(seed),
        'nepochs': int(scratch['nepochs']),
        'lr': float(scratch['lr']),
        'method_order': [mode for mode, _ in avail],
        'from_scratch_reference': scratch,
        'methods': methods,
    }


def export_csvs(summary, csv_dir, report_tag='', skip_exp1=False):
    os.makedirs(csv_dir, exist_ok=True)
    methods = summary['methods']
    scratch = summary['from_scratch_reference']
    active_modes = summary['method_order']
    exp1_active = [m for m in EXP1_MODES if m in methods]
    num_tasks = len(scratch['tasks'])
    ft_col = f"Forward_Transfer_dAUC_1_to_{num_tasks - 1}_pct"
    tag = report_tag if report_tag else summary['experiment']
    use_custom_names = bool(report_tag) or skip_exp1

    written = []

    if not skip_exp1:
        # 1. Exp 1 Diagnostics CSV
        exp1_fields = [
            'mode', 'display_name', 'memory_regime', 'A_T_pct', 'BWT_pct', 'Forgetting_pct',
            'LC_AUC_pct', ft_col, 'lr_D_max_min_ratio',
            'mean_cos_sim', 'conflict_rate_pct', 'post_proj_violation_rate_pct',
            'mean_post_proj_hd', 'mean_curvature_margin', 'finite_step_conflict_rate_pct',
            'scale_interference_delta_loss', 'scale_interference_rate_pct',
            'train_runtime_sec', 'peak_cpu_memory_mb',
        ]
        exp1_rows = []
        for mode in exp1_active:
            r = methods[mode]
            exp1_rows.append({
                'mode': mode,
                'display_name': r['display_name'],
                'memory_regime': r['memory_regime'],
                'A_T_pct': f"{100.0 * r['A_T']:.4f}",
                'BWT_pct': f"{100.0 * r['BWT']:.4f}",
                'Forgetting_pct': f"{100.0 * r['Forgetting']:.4f}",
                'LC_AUC_pct': f"{100.0 * r['mean_learning_curve_auc']:.4f}",
                ft_col: f"{100.0 * r['mean_forward_transfer_delta_auc_tasks_1_to_T']:.4f}",
                'lr_D_max_min_ratio': f"{r['lr_d_max_min_ratio_final']:.4f}",
                'mean_cos_sim': f"{r['mean_cos_sim_tasks_1_to_T']:.6f}",
                'conflict_rate_pct': f"{100.0 * r['mean_conflict_rate_tasks_1_to_T']:.2f}",
                'post_proj_violation_rate_pct': f"{100.0 * r['mean_post_proj_violation_rate']:.2f}",
                'mean_post_proj_hd': f"{r['mean_post_proj_hd']:.4e}",
                'mean_curvature_margin': f"{r['mean_curvature_margin']:.4e}",
                'finite_step_conflict_rate_pct': f"{100.0 * r['mean_finite_step_conflict_rate']:.2f}",
                'scale_interference_delta_loss': f"{r['mean_scale_interference_delta_loss']:.4e}",
                'scale_interference_rate_pct': f"{100.0 * r['scale_interference_rate']:.2f}",
                'train_runtime_sec': f"{r['train_runtime_sec']:.2f}",
                'peak_cpu_memory_mb': f"{r['peak_cpu_memory_mb']:.2f}",
            })
        exp1_csv = os.path.join(csv_dir, f"exp1_diagnostics_{summary['experiment']}_seed{summary['seed']}.csv")
        write_csv(exp1_csv, exp1_fields, exp1_rows)
        written.append(exp1_csv)

    # 2. From-Scratch CSV
    exp2_fields = ['task_id', 'task_name', 'learning_curve_auc_pct', 'test_acc_pct', 'test_loss', 'runtime_sec']
    exp2_rows = []
    for t_key in sorted(scratch['tasks'].keys(), key=int):
        tv = scratch['tasks'][t_key]
        exp2_rows.append({
            'task_id': tv['task_id'],
            'task_name': tv['name'],
            'learning_curve_auc_pct': f"{100.0 * tv['learning_curve_auc']:.4f}",
            'test_acc_pct': f"{100.0 * tv['test_acc']:.4f}",
            'test_loss': f"{tv['test_loss']:.4f}",
            'runtime_sec': f"{tv['runtime_sec']:.2f}",
        })
    exp2_name = f"{tag}_from_scratch_seed{summary['seed']}.csv" if use_custom_names else f"exp2_from_scratch_{summary['experiment']}_seed{summary['seed']}.csv"
    exp2_csv = os.path.join(csv_dir, exp2_name)
    write_csv(exp2_csv, exp2_fields, exp2_rows)
    written.append(exp2_csv)

    # 3. All-Methods Summary CSV
    exp3_fields = [
        'mode', 'display_name', 'memory_regime', 'buffer_size',
        'A_T_pct', 'BWT_pct', 'Forgetting_pct',
        'LC_AUC_pct', ft_col, 'Forward_Transfer_dAUC_all_pct',
        'mean_cos_sim', 'conflict_rate_pct', 'post_proj_violation_rate_pct',
        'mean_post_proj_hd', 'mean_curvature_margin', 'finite_step_conflict_rate_pct',
        'scale_interference_delta_loss', 'scale_interference_rate_pct',
        'lr_D_max_min_ratio', 'train_runtime_sec', 'total_runtime_sec', 'peak_cpu_memory_mb', 'peak_gpu_memory_mb',
    ]
    exp3_rows = []
    for mode in active_modes:
        r = methods[mode]
        exp3_rows.append({
            'mode': mode,
            'display_name': r['display_name'],
            'memory_regime': r['memory_regime'],
            'buffer_size': r['buffer_size'],
            'A_T_pct': f"{100.0 * r['A_T']:.4f}",
            'BWT_pct': f"{100.0 * r['BWT']:.4f}",
            'Forgetting_pct': f"{100.0 * r['Forgetting']:.4f}",
            'LC_AUC_pct': f"{100.0 * r['mean_learning_curve_auc']:.4f}",
            ft_col: f"{100.0 * r['mean_forward_transfer_delta_auc_tasks_1_to_T']:.4f}",
            'Forward_Transfer_dAUC_all_pct': f"{100.0 * r['mean_forward_transfer_delta_auc_all']:.4f}",
            'mean_cos_sim': f"{r['mean_cos_sim_tasks_1_to_T']:.6f}",
            'conflict_rate_pct': f"{100.0 * r['mean_conflict_rate_tasks_1_to_T']:.2f}",
            'post_proj_violation_rate_pct': f"{100.0 * r['mean_post_proj_violation_rate']:.2f}",
            'mean_post_proj_hd': f"{r['mean_post_proj_hd']:.4e}",
            'mean_curvature_margin': f"{r['mean_curvature_margin']:.4e}",
            'finite_step_conflict_rate_pct': f"{100.0 * r['mean_finite_step_conflict_rate']:.2f}",
            'scale_interference_delta_loss': f"{r['mean_scale_interference_delta_loss']:.4e}",
            'scale_interference_rate_pct': f"{100.0 * r['scale_interference_rate']:.2f}",
            'lr_D_max_min_ratio': f"{r['lr_d_max_min_ratio_final']:.4f}",
            'train_runtime_sec': f"{r['train_runtime_sec']:.2f}",
            'total_runtime_sec': f"{r['runtime_sec']:.2f}",
            'peak_cpu_memory_mb': f"{r['peak_cpu_memory_mb']:.2f}",
            'peak_gpu_memory_mb': f"{r['peak_gpu_memory_mb']:.2f}",
        })
    exp3_name = f"{tag}_all_methods_seed{summary['seed']}.csv" if use_custom_names else f"exp3_all_methods_{summary['experiment']}_seed{summary['seed']}.csv"
    exp3_csv = os.path.join(csv_dir, exp3_name)
    write_csv(exp3_csv, exp3_fields, exp3_rows)
    written.append(exp3_csv)

    # 4. Per-Task Transfer & Retention Tidy CSV
    pt_fields = [
        'mode', 'task_id', 'task_name', 'cl_lc_auc_pct', 'scratch_lc_auc_pct', 'delta_auc_pct',
        'immediate_test_acc_pct', 'final_test_acc_pct', 'scratch_test_acc_pct',
        'mean_cos_sim', 'conflict_rate_pct', 'post_proj_violation_rate_pct',
        'mean_curvature_margin', 'finite_step_conflict_rate_pct',
    ]
    pt_rows = []
    for mode in active_modes:
        per_task = methods[mode]['per_task']
        for t_key in sorted(per_task.keys(), key=int):
            tv = per_task[t_key]
            pt_rows.append({
                'mode': mode,
                'task_id': int(t_key),
                'task_name': tv['task_name'],
                'cl_lc_auc_pct': f"{100.0 * tv['cl_learning_curve_auc']:.4f}",
                'scratch_lc_auc_pct': f"{100.0 * tv['scratch_learning_curve_auc']:.4f}",
                'delta_auc_pct': f"{100.0 * tv['delta_auc']:.4f}",
                'immediate_test_acc_pct': f"{100.0 * tv['immediate_test_acc']:.4f}",
                'final_test_acc_pct': f"{100.0 * tv['final_test_acc']:.4f}",
                'scratch_test_acc_pct': f"{100.0 * tv['scratch_test_acc']:.4f}",
                'mean_cos_sim': f"{tv['mean_cos_sim']:.6f}",
                'conflict_rate_pct': f"{100.0 * tv['conflict_rate']:.2f}",
                'post_proj_violation_rate_pct': f"{100.0 * tv['post_proj_violation_rate']:.2f}",
                'mean_curvature_margin': f"{tv['mean_curvature_margin']:.4e}",
                'finite_step_conflict_rate_pct': f"{100.0 * tv['finite_step_conflict_rate']:.2f}",
            })
    pt_name = f"{tag}_per_task_transfer_seed{summary['seed']}.csv" if use_custom_names else f"exp3_per_task_transfer_{summary['experiment']}_seed{summary['seed']}.csv"
    pt_csv = os.path.join(csv_dir, pt_name)
    write_csv(pt_csv, pt_fields, pt_rows)
    written.append(pt_csv)

    return written


def build_charts_figure(summary):
    methods = summary['methods']
    modes = summary['method_order']
    n_tasks = len(summary['from_scratch_reference']['tasks'])
    labels = [SHORT_LABELS[m] for m in modes]
    colors = [COLOR_MAP[m] for m in modes]

    plt.rcParams.update({
        'font.size': 9.5,
        'axes.titlesize': 11,
        'axes.labelsize': 9.5,
        'xtick.labelsize': 8.2,
        'figure.dpi': 150,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0))
    x = np.arange(len(modes))

    # (a) Final Average Accuracy A_T (%)
    a_t = [100.0 * methods[m]['A_T'] for m in modes]
    bars_a = axes[0, 0].bar(x, a_t, color=colors, edgecolor='#222222', linewidth=0.7)
    y_lo_a = max(0.0, min(a_t) - 6.0)
    y_hi_a = min(100.0, max(a_t) + max(4.5, 0.12 * (max(a_t) - min(a_t))))
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].set_ylim(y_lo_a, y_hi_a)
    axes[0, 0].set_ylabel(r'Final Average Accuracy $A_T$ (%) $\uparrow$')
    axes[0, 0].set_title(r'(a) Final Accuracy $A_T$ Across All Methods')
    axes[0, 0].grid(True, axis='y', alpha=0.3)
    off_a = max(0.45, 0.018 * (y_hi_a - y_lo_a))
    for b in bars_a:
        axes[0, 0].text(b.get_x() + b.get_width() / 2, b.get_height() + off_a, f"{b.get_height():.2f}%", ha='center', fontsize=8.0)

    # (b) Average Forgetting F (%)
    fgt = [100.0 * methods[m]['Forgetting'] for m in modes]
    bars_b = axes[0, 1].bar(x, fgt, color=colors, edgecolor='#222222', linewidth=0.7)
    y_hi_b = max(fgt) * 1.24 + 0.8
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels)
    axes[0, 1].set_ylim(0, y_hi_b)
    axes[0, 1].set_ylabel(r'Average Forgetting $F$ (%) $\downarrow$')
    axes[0, 1].set_title(r'(b) Catastrophic Forgetting $F$ (Lower is Better)')
    axes[0, 1].grid(True, axis='y', alpha=0.3)
    off_b = max(0.25, 0.018 * y_hi_b)
    for b in bars_b:
        axes[0, 1].text(b.get_x() + b.get_width() / 2, b.get_height() + off_b, f"{b.get_height():.2f}%", ha='center', fontsize=8.0)

    # (c) Forward Transfer Delta-AUC (Tasks 1..T-1) vs From-Scratch Reference
    d_auc = [100.0 * methods[m]['mean_forward_transfer_delta_auc_tasks_1_to_T'] for m in modes]
    bars_c = axes[1, 0].bar(x, d_auc, color=colors, edgecolor='#222222', linewidth=0.7)
    axes[1, 0].axhline(0.0, color='#111111', linestyle='--', linewidth=1.1)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels)
    y_min_c = min(0.0, min(d_auc)) - 0.65
    y_max_c = max(0.0, max(d_auc)) + 0.85
    axes[1, 0].set_ylim(y_min_c, y_max_c)
    axes[1, 0].set_ylabel(rf'Transfer $\Delta\mathrm{{AUC}}_{{1\dots{n_tasks - 1}}}$ vs. From-Scratch (%) $\uparrow$')
    axes[1, 0].set_title(r'(c) New-Task Adaptation / Forward Transfer ($\mathrm{AUC}_{\mathrm{CL}} - \mathrm{AUC}_{\mathrm{scratch}}$)')
    axes[1, 0].grid(True, axis='y', alpha=0.3)
    span_c = y_max_c - y_min_c
    for b in bars_c:
        val = b.get_height()
        offset = max(0.08, 0.02 * span_c) if val >= 0 else -max(0.25, 0.055 * span_c)
        axes[1, 0].text(b.get_x() + b.get_width() / 2, val + offset, f"{val:+.2f}%", ha='center', fontsize=8.0)

    # (d) Post-Projection Compatibility Violation Rate (% of conflicting steps)
    viol = [100.0 * methods[m]['mean_post_proj_violation_rate'] for m in modes]
    bars_d = axes[1, 1].bar(x, viol, color=colors, edgecolor='#222222', linewidth=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_ylim(0, 115)
    axes[1, 1].set_ylabel(r'Post-Step Violation $\Pr(h^\top d < 0 \mid \mathrm{conflict})$ (%) $\downarrow$')
    axes[1, 1].set_title(r'(d) Compatibility Violations After Forming Update Step $d$')
    axes[1, 1].grid(True, axis='y', alpha=0.3)
    for b in bars_d:
        axes[1, 1].text(b.get_x() + b.get_width() / 2, b.get_height() + 2.0, f"{b.get_height():.1f}%", ha='center', fontsize=8.0)

    fig.suptitle(
        f"Idea 2 ({summary['experiment']} — {n_tasks} Tasks, {summary['order'].title()}, Seed {summary['seed']}, {summary['nepochs']} Epochs/Task): "
        "Proposal Baselines + 2nd-Order Finite-Step D-Projection",
        fontsize=12.0,
        fontweight='bold',
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    return fig


def build_per_task_figure(summary):
    methods = summary['methods']
    modes = summary['method_order']
    scratch_tasks = summary['from_scratch_reference']['tasks']
    n_tasks = len(scratch_tasks)
    task_ids = list(range(n_tasks))
    later_ids = list(range(1, n_tasks))

    markers = {
        'ucb': 'o',
        'bft_replay': 'v',
        'ucb_replay': 's',
        'agem': '^',
        'ucb_raw_proj': 'D',
        'ucb_d_proj': 'P',
        'ucb_d_proj_2nd': '*',
    }

    plt.rcParams.update({
        'font.size': 9.5,
        'axes.titlesize': 11,
        'axes.labelsize': 9.5,
        'xtick.labelsize': 8.2,
        'figure.dpi': 150,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))

    x_ticks_all = np.arange(n_tasks)
    tick_labels_all = [
        f"T{t}\n({scratch_tasks[str(t)]['name'].replace('rmnist-', '').replace('deg', '°')})"
        for t in task_ids
    ]

    # (a) Final Retained Test Accuracy R_{T-1, t} (%) across tasks
    sc_acc = [100.0 * scratch_tasks[str(t)]['test_acc'] for t in task_ids]
    axes[0].plot(
        x_ticks_all, sc_acc, linestyle='--', color='#444444', linewidth=1.4, marker='x',
        markersize=5, label='From-Scratch (Ref)', zorder=2,
    )
    for m in modes:
        r_acc = [100.0 * methods[m]['per_task'][str(t)]['final_test_acc'] for t in task_ids]
        lw = 2.4 if m == 'ucb_d_proj_2nd' else 1.7
        ms = 8.5 if m == 'ucb_d_proj_2nd' else 5.5
        axes[0].plot(
            x_ticks_all, r_acc, color=COLOR_MAP[m], marker=markers.get(m, 'o'),
            linewidth=lw, markersize=ms, label=DISPLAY_NAMES[m], zorder=4 if m == 'ucb_d_proj_2nd' else 3,
        )
    axes[0].set_xticks(x_ticks_all)
    axes[0].set_xticklabels(tick_labels_all)
    axes[0].set_ylabel(r'Final Retained Test Accuracy $R_{T-1, t}$ (%) $\uparrow$')
    axes[0].set_title(rf'(a) Final Retained Accuracy Across All {n_tasks} Tasks')
    axes[0].grid(True, alpha=0.3)

    # (b) Forward Transfer Delta-AUC_t (%) across tasks 1..T-1
    x_ticks_later = np.arange(len(later_ids))
    tick_labels_later = [tick_labels_all[t] for t in later_ids]
    axes[1].axhline(0.0, color='#111111', linestyle='--', linewidth=1.1)
    for m in modes:
        d_auc = [100.0 * methods[m]['per_task'][str(t)]['delta_auc'] for t in later_ids]
        lw = 2.4 if m == 'ucb_d_proj_2nd' else 1.7
        ms = 8.5 if m == 'ucb_d_proj_2nd' else 5.5
        axes[1].plot(
            x_ticks_later, d_auc, color=COLOR_MAP[m], marker=markers.get(m, 'o'),
            linewidth=lw, markersize=ms, zorder=4 if m == 'ucb_d_proj_2nd' else 3,
        )
    axes[1].set_xticks(x_ticks_later)
    axes[1].set_xticklabels(tick_labels_later)
    axes[1].set_ylabel(r'Forward Transfer $\Delta\mathrm{AUC}_t = \mathrm{AUC}_{\mathrm{CL}}(t) - \mathrm{AUC}_{\mathrm{scratch}}(t)$ (%) $\uparrow$')
    axes[1].set_title(rf'(b) Per-Task Forward Transfer ($\mathrm{{Tasks}}\ 1\dots{n_tasks - 1}$)')
    axes[1].grid(True, alpha=0.3)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        f"Per-Task Retention & Forward Transfer — {summary['experiment']} ({n_tasks} Tasks, {summary['order'].title()}, Seed {summary['seed']})",
        fontsize=12.0,
        fontweight='bold',
        y=0.985,
    )
    fig.legend(
        handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.93),
        ncol=min(6, len(legend_labels)), fontsize=8.3, frameon=True,
    )
    fig.tight_layout(rect=[0, 0.01, 1, 0.84])
    return fig


def export_pdf(summary, pdf_path, png_path, per_task_png_path=None):
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    methods = summary['methods']
    scratch = summary['from_scratch_reference']
    active_modes = summary['method_order']
    sc_tasks = scratch['tasks']
    n_t = len(sc_tasks)
    per_bucket = 200.0 / max(1, n_t * 10)
    per_bucket_str = f"{int(per_bucket)} ex/class/task" if abs(per_bucket - round(per_bucket)) < 1e-9 else f"{per_bucket:.1f} ex/class/task"

    with PdfPages(pdf_path) as pdf:
        # Page 1: Structured Landscape Summary Table + Methodology Box
        fig_tbl = plt.figure(figsize=(14.5, 8.8))
        ax = fig_tbl.add_subplot(111)
        ax.axis('off')

        ax.text(
            0.01, 0.97,
            f"Idea 2 Report: Interference Between Similar Tasks — {summary['experiment']} ({n_t} Tasks, {summary['order'].title()}, Seed {summary['seed']})",
            fontsize=14, fontweight='bold', va='top',
        )
        ax.text(
            0.01, 0.925,
            f"Shared-Head 2-Layer Bayesian MLP (nhid=200, pure SGD, lr={summary['lr']}, nepochs={summary['nepochs']}, "
            f"Exact Algorithm 1 BBB Loss). Replay-free UCB separated from 200-example buffer ({per_bucket_str}).",
            fontsize=9.2, color='#333333', va='top',
        )

        col_labels = [
            'Method', 'Memory', 'A_T\n(%)', 'BWT\n(%)', 'Forget\n(%)',
            'LC-AUC\n(%)', 'Transfer\ndAUC (%)', 'cos(g,h)', 'Conflict\nRate (%)',
            'Post-Step\nViol (%)', 'Curvature\nMargin', 'Scale dL\n(rho)', 'Time (s)', 'RAM (MB)'
        ]
        table_rows = []
        for mode in active_modes:
            r = methods[mode]
            curv_str = f"{r['mean_curvature_margin']:+.2e}" if mode == 'ucb_d_proj_2nd' else '—'
            table_rows.append([
                r['display_name'],
                '0 (Free)' if r['buffer_size'] == 0 else '200 ex.',
                f"{100.0 * r['A_T']:.2f}",
                f"{100.0 * r['BWT']:+.2f}",
                f"{100.0 * r['Forgetting']:.2f}",
                f"{100.0 * r['mean_learning_curve_auc']:.2f}",
                f"{100.0 * r['mean_forward_transfer_delta_auc_tasks_1_to_T']:+.2f}",
                f"{r['mean_cos_sim_tasks_1_to_T']:+.4f}",
                f"{100.0 * r['mean_conflict_rate_tasks_1_to_T']:.1f}",
                f"{100.0 * r['mean_post_proj_violation_rate']:.1f}",
                curv_str,
                f"{r['mean_scale_interference_delta_loss']:+.1e}",
                f"{r['train_runtime_sec']:.1f}",
                f"{r['peak_cpu_memory_mb']:.0f}",
            ])

        col_widths_p1 = [0.179, 0.064, 0.056, 0.058, 0.063, 0.065, 0.062, 0.062, 0.062, 0.065, 0.068, 0.065, 0.062, 0.069]
        tbl = ax.table(
            cellText=table_rows,
            colLabels=col_labels,
            colWidths=col_widths_p1,
            loc='center',
            cellLoc='center',
            bbox=[0.01, 0.45, 0.98, 0.43],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.1)
        for (row_idx, col_idx), cell in tbl.get_celld().items():
            if row_idx == 0:
                cell.set_facecolor('#1f4e78')
                cell.set_text_props(color='white', fontweight='bold')
            elif row_idx == 1:
                cell.set_facecolor('#f2f2f2')
            elif active_modes[row_idx - 1] == 'ucb_d_proj':
                cell.set_facecolor('#eef5fc')
            elif active_modes[row_idx - 1] == 'ucb_d_proj_2nd':
                cell.set_facecolor('#d7e9fa')
                cell.set_text_props(fontweight='bold')
            else:
                cell.set_facecolor('#ffffff')
            if col_idx == 0 and row_idx > 0:
                cell.set_text_props(ha='left')

        # Second compact table: From-Scratch Transfer Reference per task
        ax.text(
            0.01, 0.40,
            f"Experiment 2: From-Scratch Single-Task Reference (Fresh BNN trained per domain, Seed {summary['seed']})",
            fontsize=10.8, fontweight='bold', va='top',
        )
        if n_t > 6:
            sc_cols = ['Metric'] + [
                f"T{t}\n({sc_tasks[str(t)]['name'].replace('rmnist-', '').replace('deg', '°')})"
                for t in range(n_t)
            ] + [f"Mean\n(1–{n_t - 1})"]
            row1 = ['LC-AUC (%)'] + [f"{100.0 * sc_tasks[str(t)]['learning_curve_auc']:.2f}" for t in range(n_t)] + [
                f"{100.0 * scratch['mean_learning_curve_auc_tasks_1_to_T']:.2f}"
            ]
            row2 = ['Test Acc (%)'] + [f"{100.0 * sc_tasks[str(t)]['test_acc']:.2f}" for t in range(n_t)] + [
                f"{100.0 * np.mean([sc_tasks[str(t)]['test_acc'] for t in range(1, n_t)]):.2f}"
            ]
            col_widths_sc = [0.095] + [0.815 / n_t] * n_t + [0.090]
            tbl_sc = ax.table(
                cellText=[row1, row2],
                colLabels=sc_cols,
                colWidths=col_widths_sc,
                loc='center',
                cellLoc='center',
                bbox=[0.01, 0.23, 0.98, 0.14],
            )
            tbl_sc.auto_set_font_size(False)
            tbl_sc.set_fontsize(7.8)
            for (row_idx, col_idx), cell in tbl_sc.get_celld().items():
                if row_idx == 0:
                    cell.set_facecolor('#44546a')
                    cell.set_text_props(color='white', fontweight='bold')
                elif col_idx == 0:
                    cell.set_facecolor('#f2f4f7')
                    cell.set_text_props(fontweight='bold')
        else:
            sc_cols = [f"Task {t} ({sc_tasks[str(t)]['name'].replace('rmnist-', '')})" for t in range(n_t)] + [f'Mean (Tasks 1–{n_t - 1})']
            sc_aucs = [f"{100.0 * sc_tasks[str(t)]['learning_curve_auc']:.2f}% (Test {100.0 * sc_tasks[str(t)]['test_acc']:.2f}%)" for t in range(n_t)]
            sc_aucs.append(f"{100.0 * scratch['mean_learning_curve_auc_tasks_1_to_T']:.2f}% AUC")
            tbl_sc = ax.table(
                cellText=[sc_aucs],
                colLabels=sc_cols,
                loc='center',
                cellLoc='center',
                bbox=[0.01, 0.24, 0.98, 0.12],
            )
            tbl_sc.auto_set_font_size(False)
            tbl_sc.set_fontsize(8.2)
            for (row_idx, _), cell in tbl_sc.get_celld().items():
                if row_idx == 0:
                    cell.set_facecolor('#44546a')
                    cell.set_text_props(color='white', fontweight='bold')

        # Explicit methodology box documenting the 2nd-order finite-step curvature extension
        method_box = (
            "2nd-Order Finite-Step Curvature Extension (ucb_d_proj_2nd):\n"
            "  For a finite UCB candidate step d0 = D*g, buffer loss changes by  dL_exact = L_buf(mu - d0) - L_buf(mu) = -h^T d0 + (1/2) d0^T H_buf d0 + O(||d0||^3).\n"
            "  1st-order ucb_d_proj removes only -h^T d0 (leaving positive 2nd-order curvature drift). Using 1 CRN no-grad forward pass on the 200 buffer examples,\n"
            "  ucb_d_proj_2nd computes dL_exact and applies  d* = D*g + [max(0, -h^T D*g, dL_exact) / (h^T D*h)] * D*h, eliminating both 1st- and 2nd-order\n"
            "  finite-step interference when dL_exact > 0 while keeping compatible steps (h^T D*g >= 0 and dL_exact <= 0) completely untouched for full forward transfer."
        )
        ax.text(
            0.01, 0.19, method_box,
            fontsize=8.3, color='#1b2a38', va='top', linespacing=1.45,
            bbox=dict(boxstyle='round,pad=0.55', facecolor='#f4f8fc', edgecolor='#1f4e78', linewidth=1.0),
        )

        pdf.savefig(fig_tbl, bbox_inches='tight')
        plt.close(fig_tbl)

        # Page 2: 2x2 Visual Comparison Charts
        fig_charts = build_charts_figure(summary)
        fig_charts.savefig(png_path, bbox_inches='tight')
        pdf.savefig(fig_charts, bbox_inches='tight')
        plt.close(fig_charts)

        # Page 3: Per-Task Retention & Forward Transfer Curves
        fig_pt = build_per_task_figure(summary)
        if per_task_png_path:
            fig_pt.savefig(per_task_png_path, bbox_inches='tight')
        pdf.savefig(fig_pt, bbox_inches='tight')
        plt.close(fig_pt)


def export_exp1_only(checkpoints_dir, json_dir, csv_dir, pdf_dir, experiment='rmnist10', seed=0):
    avail = _available_modes(checkpoints_dir, EXP1_MODES, experiment, seed)
    methods = {}
    for mode, m_path in avail:
        m = load_json(m_path)
        td = m['task_diagnostics']
        num_tasks = len(m['R_matrix'])
        methods[mode] = {
            'mode': mode,
            'display_name': DISPLAY_NAMES[mode],
            'memory_regime': 'Replay-Free (0)' if mode == 'ucb' else '200-Example Buffer',
            'buffer_size': 0 if mode == 'ucb' else int(m['buffer_size']),
            'A_T': float(m['A_T']),
            'BWT': float(m['BWT']),
            'Forgetting': float(m['Forgetting']),
            'mean_learning_curve_auc': float(m['mean_learning_curve_auc']),
            'mean_cos_sim_tasks_1_to_T': float(m['mean_cos_sim_tasks_1_to_T']),
            'mean_conflict_rate_tasks_1_to_T': float(m['mean_conflict_rate_tasks_1_to_T']),
            'mean_post_proj_violation_rate': float(m['mean_post_proj_violation_rate']),
            'mean_post_proj_hd': float(m['mean_post_proj_hd']),
            'mean_curvature_margin': float(m.get('mean_curvature_margin', 0.0)),
            'mean_finite_step_conflict_rate': float(m.get('mean_finite_step_conflict_rate', 0.0)),
            'mean_scale_interference_delta_loss': float(m['mean_scale_interference_delta_loss']),
            'scale_interference_rate': float(m['scale_interference_rate']),
            'lr_d_max_min_ratio_final': float(td[str(num_tasks - 1)]['lr_d_spread']['max_min_ratio']),
            'runtime_sec': float(m['runtime_sec']),
            'train_runtime_sec': float(m['train_runtime_sec']),
            'diag_overhead_sec': float(m['diag_overhead_sec']),
            'peak_cpu_memory_mb': float(m['peak_cpu_memory_mb']),
            'peak_gpu_memory_mb': float(m['peak_gpu_memory_mb']),
            'R_matrix': m['R_matrix'],
        }

    summary = {
        'experiment': experiment,
        'order': 'forward',
        'seed': int(seed),
        'method_order': [mode for mode, _ in avail],
        'methods': methods,
    }
    json_path = os.path.join(json_dir, f'exp1_diagnostics_{experiment}_seed{seed}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    fields = [
        'mode', 'display_name', 'memory_regime', 'A_T_pct', 'BWT_pct', 'Forgetting_pct',
        'LC_AUC_pct', 'lr_D_max_min_ratio', 'mean_cos_sim', 'conflict_rate_pct',
        'post_proj_violation_rate_pct', 'mean_post_proj_hd',
        'mean_curvature_margin', 'finite_step_conflict_rate_pct',
        'scale_interference_delta_loss', 'scale_interference_rate_pct',
        'train_runtime_sec', 'peak_cpu_memory_mb',
    ]
    rows = []
    for mode in summary['method_order']:
        r = methods[mode]
        rows.append({
            'mode': mode,
            'display_name': r['display_name'],
            'memory_regime': r['memory_regime'],
            'A_T_pct': f"{100.0 * r['A_T']:.4f}",
            'BWT_pct': f"{100.0 * r['BWT']:.4f}",
            'Forgetting_pct': f"{100.0 * r['Forgetting']:.4f}",
            'LC_AUC_pct': f"{100.0 * r['mean_learning_curve_auc']:.4f}",
            'lr_D_max_min_ratio': f"{r['lr_d_max_min_ratio_final']:.4f}",
            'mean_cos_sim': f"{r['mean_cos_sim_tasks_1_to_T']:.6f}",
            'conflict_rate_pct': f"{100.0 * r['mean_conflict_rate_tasks_1_to_T']:.2f}",
            'post_proj_violation_rate_pct': f"{100.0 * r['mean_post_proj_violation_rate']:.2f}",
            'mean_post_proj_hd': f"{r['mean_post_proj_hd']:.4e}",
            'mean_curvature_margin': f"{r['mean_curvature_margin']:.4e}",
            'finite_step_conflict_rate_pct': f"{100.0 * r['mean_finite_step_conflict_rate']:.2f}",
            'scale_interference_delta_loss': f"{r['mean_scale_interference_delta_loss']:.4e}",
            'scale_interference_rate_pct': f"{100.0 * r['scale_interference_rate']:.2f}",
            'train_runtime_sec': f"{r['train_runtime_sec']:.2f}",
            'peak_cpu_memory_mb': f"{r['peak_cpu_memory_mb']:.2f}",
        })
    csv_path = os.path.join(csv_dir, f'exp1_diagnostics_{experiment}_seed{seed}.csv')
    write_csv(csv_path, fields, rows)

    pdf_path = os.path.join(pdf_dir, f'exp1_diagnostics_{experiment}_seed{seed}.pdf')
    png_path = os.path.join(pdf_dir, f'exp1_diagnostics_{experiment}_seed{seed}.png')

    plt.rcParams.update({'font.size': 9.5, 'figure.dpi': 150})
    fig = plt.figure(figsize=(14.0, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.3], hspace=0.36, wspace=0.28)

    ax_tbl = fig.add_subplot(gs[0, :])
    ax_tbl.axis('off')
    ax_tbl.text(
        0.0, 0.97,
        f"Experiment 1: Projection Geometry & 2nd-Order Curvature Diagnostic ({experiment}, Seed {seed})",
        fontsize=13.5, fontweight='bold', va='top',
    )
    ax_tbl.text(
        0.0, 0.85,
        "Exact Algorithm 1 BBB Loss (nhid=200, nlayers=2, pure SGD, lr=0.05, nepochs=5). Replay-free UCB separated from 200-example buffer runs.",
        fontsize=9.2, color='#333333', va='top',
    )

    col_labels = [
        'Method', 'Memory', 'A_T (%)', 'BWT (%)', 'Forget (%)', 'LC-AUC (%)',
        'D Spread', 'cos(g,h)', 'Conflict (%)', 'Post-Step\nViol (%)', 'Mean h^T d', 'Scale dL', 'Time (s)'
    ]
    tbl_rows = []
    for mode in summary['method_order']:
        r = methods[mode]
        tbl_rows.append([
            r['display_name'],
            '0 (Free)' if r['buffer_size'] == 0 else '200 ex.',
            f"{100.0 * r['A_T']:.2f}",
            f"{100.0 * r['BWT']:+.2f}",
            f"{100.0 * r['Forgetting']:.2f}",
            f"{100.0 * r['mean_learning_curve_auc']:.2f}",
            f"{r['lr_d_max_min_ratio_final']:.2f}x",
            f"{r['mean_cos_sim_tasks_1_to_T']:+.4f}",
            f"{100.0 * r['mean_conflict_rate_tasks_1_to_T']:.1f}",
            f"{100.0 * r['mean_post_proj_violation_rate']:.1f}",
            f"{r['mean_post_proj_hd']:+.2e}",
            f"{r['mean_scale_interference_delta_loss']:+.1e}",
            f"{r['train_runtime_sec']:.1f}",
        ])

    col_widths_exp1 = [0.185, 0.068, 0.060, 0.062, 0.064, 0.068, 0.065, 0.066, 0.066, 0.070, 0.080, 0.070, 0.076]
    tbl = ax_tbl.table(
        cellText=tbl_rows,
        colLabels=col_labels,
        colWidths=col_widths_exp1,
        loc='center',
        cellLoc='center',
        bbox=[0.0, 0.05, 1.0, 0.70],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.3)
    for (row_idx, col_idx), cell in tbl.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor('#1f4e78')
            cell.set_text_props(color='white', fontweight='bold')
        elif row_idx == 1:
            cell.set_facecolor('#f2f2f2')
        elif summary['method_order'][row_idx - 1] == 'ucb_d_proj_2nd':
            cell.set_facecolor('#d7e9fa')
            cell.set_text_props(fontweight='bold')
        else:
            cell.set_facecolor('#ffffff')
        if col_idx == 0 and row_idx > 0:
            cell.set_text_props(ha='left')

    colors = [COLOR_MAP[m] for m in summary['method_order']]
    x_labels = [SHORT_LABELS[m] for m in summary['method_order']]
    x = np.arange(len(summary['method_order']))

    ax1 = fig.add_subplot(gs[1, 0])
    vals_a = [100.0 * methods[m]['A_T'] for m in summary['method_order']]
    b1 = ax1.bar(x, vals_a, color=colors, edgecolor='#222222', width=0.58)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylim(max(75, min(vals_a) - 5), min(100, max(vals_a) + 4))
    ax1.set_ylabel(r'Final Accuracy $A_T$ (%) $\uparrow$')
    ax1.set_title(r'(a) Final Accuracy $A_T$')
    ax1.grid(True, axis='y', alpha=0.3)
    for b in b1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.35, f"{b.get_height():.2f}%", ha='center', fontsize=8.2)

    ax2 = fig.add_subplot(gs[1, 1])
    vals_f = [100.0 * methods[m]['Forgetting'] for m in summary['method_order']]
    b2 = ax2.bar(x, vals_f, color=colors, edgecolor='#222222', width=0.58)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.set_ylim(0, max(vals_f) * 1.24 + 0.5)
    ax2.set_ylabel(r'Forgetting $F$ (%) $\downarrow$')
    ax2.set_title(r'(b) Average Forgetting $F$')
    ax2.grid(True, axis='y', alpha=0.3)
    for b in b2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.25, f"{b.get_height():.2f}%", ha='center', fontsize=8.2)

    ax3 = fig.add_subplot(gs[1, 2])
    vals_v = [100.0 * methods[m]['mean_post_proj_violation_rate'] for m in summary['method_order']]
    b3 = ax3.bar(x, vals_v, color=colors, edgecolor='#222222', width=0.58)
    ax3.set_xticks(x)
    ax3.set_xticklabels(x_labels)
    ax3.set_ylim(0, 114)
    ax3.set_ylabel(r'Post-Step Violation $\Pr(h^\top d < 0)$ (%) $\downarrow$')
    ax3.set_title(r'(c) Compatibility Violations on Conflict')
    ax3.grid(True, axis='y', alpha=0.3)
    for b in b3:
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 2.0, f"{b.get_height():.1f}%", ha='center', fontsize=8.2)

    fig.savefig(png_path, bbox_inches='tight')
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    return json_path, csv_path, pdf_path, png_path


def _cleanup_stale_6methods(json_dir, csv_dir, pdf_dir, experiment, seed):
    stale = [
        os.path.join(csv_dir, f'exp3_6methods_{experiment}_seed{seed}.csv'),
        os.path.join(json_dir, f'{experiment}_6methods_seed{seed}_summary.json'),
        os.path.join(pdf_dir, f'{experiment}_6methods_seed{seed}_report.pdf'),
        os.path.join(pdf_dir, f'{experiment}_6methods_seed{seed}_charts.png'),
    ]
    for p in stale:
        if os.path.exists(p):
            os.remove(p)


def main():
    parser = argparse.ArgumentParser(description='Compile Idea 2 CSV & PDF Report')
    parser.add_argument('--experiment', type=str, default='rmnist10')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp1_only', action='store_true', help='Export Experiment 1 diagnostics only')
    parser.add_argument('--run_prefix', type=str, default='', help='Optional checkpoint dir prefix before _<mode>_s<seed>')
    parser.add_argument('--report_tag', type=str, default='', help='Optional output filename stem prefix for multi-experiment reports')
    parser.add_argument('--scratch_json', type=str, default='', help='Optional explicit path to from-scratch reference JSON')
    parser.add_argument('--reverse', action='store_true', help='Mark stream order as reverse')
    parser.add_argument('--skip_exp1', action='store_true', help='Skip writing exp1_diagnostics files')
    parser.add_argument(
        '--checkpoints_dir',
        type=str,
        default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints'),
    )
    parser.add_argument(
        '--reports_dir',
        type=str,
        default=os.path.join(os.path.dirname(__file__), '..', 'results', 'idea2_reports'),
    )
    args = parser.parse_args()

    json_dir = os.path.join(args.reports_dir, 'json')
    csv_dir = os.path.join(args.reports_dir, 'csv')
    pdf_dir = os.path.join(args.reports_dir, 'pdf')
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    e1_json = e1_csv = e1_pdf = e1_png = None
    if not args.skip_exp1:
        e1_json, e1_csv, e1_pdf, e1_png = export_exp1_only(
            args.checkpoints_dir, json_dir, csv_dir, pdf_dir, experiment=args.experiment, seed=args.seed
        )
        if args.exp1_only:
            print('EXPORTED EXP1 FILES:', e1_json, e1_csv, e1_pdf, e1_png, sep='\n  ')
            return

    order_tag = 'rev' if args.reverse else 'fwd'
    scratch_path = (
        args.scratch_json
        if args.scratch_json
        else os.path.join(json_dir, f'from_scratch_{args.experiment}_{order_tag}_seed{args.seed}.json')
    )
    summary = build_combined_summary(
        args.checkpoints_dir,
        scratch_path,
        experiment=args.experiment,
        seed=args.seed,
        run_prefix=args.run_prefix,
        reverse=args.reverse,
    )

    if args.report_tag:
        stem = f'{args.report_tag}_seed{args.seed}'
        summary_json_path = os.path.join(json_dir, f'{stem}_summary.json')
        pdf_path = os.path.join(pdf_dir, f'{stem}_report.pdf')
        png_path = os.path.join(pdf_dir, f'{stem}_charts.png')
        per_task_png_path = os.path.join(pdf_dir, f'{stem}_per_task.png')
    else:
        summary_json_path = os.path.join(json_dir, f'{args.experiment}_all_methods_seed{args.seed}_summary.json')
        pdf_path = os.path.join(pdf_dir, f'{args.experiment}_all_methods_seed{args.seed}_report.pdf')
        png_path = os.path.join(pdf_dir, f'{args.experiment}_all_methods_seed{args.seed}_charts.png')
        per_task_png_path = os.path.join(pdf_dir, f'{args.experiment}_all_methods_seed{args.seed}_per_task.png')

    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    csv_paths = export_csvs(summary, csv_dir, report_tag=args.report_tag, skip_exp1=args.skip_exp1)
    export_pdf(summary, pdf_path, png_path, per_task_png_path=per_task_png_path)
    if not args.report_tag and not args.skip_exp1:
        _cleanup_stale_6methods(json_dir, csv_dir, pdf_dir, args.experiment, args.seed)

    print('=' * 96)
    print(f"EXPORTED IDEA 2 REPORT FILES ({len(summary['method_order'])} methods):")
    print(f'  JSON Summary : {summary_json_path}')
    if e1_json:
        print(f'  JSON Exp 1   : {e1_json}')
    for cp in csv_paths:
        print(f'  CSV          : {cp}')
    print(f'  PDF Report   : {pdf_path}')
    print(f'  PNG Charts   : {png_path}')
    print(f'  PNG Per-Task : {per_task_png_path}')
    if e1_pdf:
        print(f'  PDF Exp 1    : {e1_pdf}')
        print(f'  PNG Exp 1    : {e1_png}')
    print('=' * 96)


if __name__ == '__main__':
    main()
