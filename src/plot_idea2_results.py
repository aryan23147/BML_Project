import json
import os
import tempfile

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mplconfig_ucb")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    summary_path = os.path.join(base_dir, "checkpoints", "idea2_tier1_summary.json")
    out_dir = os.path.join(base_dir, "results", "idea2")
    os.makedirs(out_dir, exist_ok=True)

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
    })

    modes = ["ucb", "agem", "ucb_replay", "ucb_d_proj"]
    labels = ["UCB\n(No Buf)", "A-GEM\n(200 Buf)", "UCB-Replay\n(200 Buf)", "UCB-D-Proj\n(Proposed)"]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))

    # Panel (a): Average Forgetting (%) across 3 seeds (mean +/- std)
    x = np.arange(len(modes))
    width = 0.36
    fgt_10_mean = [100.0 * np.mean([r["Forgetting"] for r in data["rmnist10"][m]]) for m in modes]
    fgt_10_std = [100.0 * np.std([r["Forgetting"] for r in data["rmnist10"][m]]) for m in modes]
    fgt_30_mean = [100.0 * np.mean([r["Forgetting"] for r in data["rmnist30"][m]]) for m in modes]
    fgt_30_std = [100.0 * np.std([r["Forgetting"] for r in data["rmnist30"][m]]) for m in modes]

    b1 = axes[0].bar(
        x - width / 2, fgt_10_mean, width, yerr=fgt_10_std, capsize=4,
        label=r"Rotated-MNIST $10^\circ$ (3 seeds)", color="#1f77b4", alpha=0.88
    )
    b2 = axes[0].bar(
        x + width / 2, fgt_30_mean, width, yerr=fgt_30_std, capsize=4,
        label=r"Rotated-MNIST $30^\circ$ (3 seeds)", color="#d62728", alpha=0.85
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 52)
    axes[0].set_ylabel(r"Average Forgetting $F$ (%) $\downarrow$ (Mean $\pm$ SD)")
    axes[0].set_title(r"(a) Catastrophic Forgetting $F$ (10 Epochs, 3 Seeds)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="upper right")
    for bar, std_val in zip(b1, fgt_10_std):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + std_val + 0.9,
            f"{bar.get_height():.1f}%", ha="center", fontsize=8
        )
    for bar, std_val in zip(b2, fgt_30_std):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + std_val + 0.9,
            f"{bar.get_height():.1f}%", ha="center", fontsize=8
        )

    # Panel (b): Task-by-Task Gradient Conflict Frequency (mean +/- std across 3 seeds for ucb_d_proj)
    tasks = [1, 2, 3, 4]
    conf_10_runs = [[100.0 * r["task_diagnostics"][str(t)]["conflict_rate"] for t in tasks] for r in data["rmnist10"]["ucb_d_proj"]]
    conf_30_runs = [[100.0 * r["task_diagnostics"][str(t)]["conflict_rate"] for t in tasks] for r in data["rmnist30"]["ucb_d_proj"]]
    conf_10_m, conf_10_s = np.mean(conf_10_runs, axis=0), np.std(conf_10_runs, axis=0)
    conf_30_m, conf_30_s = np.mean(conf_30_runs, axis=0), np.std(conf_30_runs, axis=0)

    axes[1].errorbar(
        tasks, conf_10_m, yerr=conf_10_s, marker="o", capsize=4, linewidth=2.3,
        color="#1f77b4", label=r"$10^\circ$ Stream ($\mathrm{mean}=54.7 \pm 0.2\%$)"
    )
    axes[1].errorbar(
        tasks, conf_30_m, yerr=conf_30_s, marker="s", capsize=4, linewidth=2.3,
        color="#d62728", label=r"$30^\circ$ Stream ($\mathrm{mean}=67.6 \pm 1.4\%$)"
    )
    axes[1].set_xticks(tasks)
    axes[1].set_xticklabels([f"Task {t}\n({10*t}° / {30*t}°)" for t in tasks])
    axes[1].set_ylabel(r"Preconditioned Conflict Rate $\Pr(h^\top D g < 0)$ (%)")
    axes[1].set_title("(b) Preconditioned Gradient Conflict (Mean $\pm$ SD)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    # Panel (c): Task-by-Task Cosine Similarity cos(g_lik, h) (mean +/- std across 3 seeds)
    cos_10_runs = [[r["task_diagnostics"][str(t)]["mean_cos_sim"] for t in tasks] for r in data["rmnist10"]["ucb_d_proj"]]
    cos_30_runs = [[r["task_diagnostics"][str(t)]["mean_cos_sim"] for t in tasks] for r in data["rmnist30"]["ucb_d_proj"]]
    cos_10_m, cos_10_s = np.mean(cos_10_runs, axis=0), np.std(cos_10_runs, axis=0)
    cos_30_m, cos_30_s = np.mean(cos_30_runs, axis=0), np.std(cos_30_runs, axis=0)

    axes[2].errorbar(
        tasks, cos_10_m, yerr=cos_10_s, marker="o", capsize=4, linewidth=2.3,
        color="#1f77b4", label=r"$10^\circ$ Stream ($\cos(g,h)$)"
    )
    axes[2].errorbar(
        tasks, cos_30_m, yerr=cos_30_s, marker="s", capsize=4, linewidth=2.3,
        color="#d62728", label=r"$30^\circ$ Stream ($\cos(g,h)$)"
    )
    axes[2].axhline(0.0, color="#333333", linestyle="--", linewidth=1.2)
    axes[2].set_xticks(tasks)
    axes[2].set_xticklabels([f"Task {t}\n({10*t}° / {30*t}°)" for t in tasks])
    axes[2].set_ylabel(r"Likelihood Gradient Cosine Similarity $\cos(g_{\mathrm{lik}}, h)$")
    axes[2].set_title("(c) Gradient Agreement vs. Angle (Mean $\pm$ SD)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    png_path = os.path.join(out_dir, "idea2_tier1_comparison.png")
    pdf_path = os.path.join(out_dir, "idea2_tier1_comparison.pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figures:\n  {png_path}\n  {pdf_path}")


if __name__ == "__main__":
    main()
