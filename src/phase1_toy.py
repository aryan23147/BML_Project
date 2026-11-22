import argparse
import json
import os
import tempfile

# Ensure matplotlib has a writable cache directory on Windows sandbox/environments
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mplconfig_ucb")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from audit_utils import (
    central_second_diff,
    diagonal_predicted_damage,
    one_sided_diff,
    quadratic_damage,
    spearman_rank_corr,
)


class ToyBayesianLinearRegression:
    """
    Exact and Mean-Field Bayesian Linear Regression:
        y = X w + eps,   eps ~ N(0, noise_var * I)
        w ~ N(0, prior_precision^-1 * I)

    Exact Posterior:
        p(w | X, y) = N(w; mu, H^-1)
        H = (1 / noise_var) * X^T X + prior_precision * I
        mu = (1 / noise_var) * H^-1 X^T y

    Negative Log-Posterior (up to constant):
        L(theta) = 0.5 / noise_var * ||y - X theta||_2^2 + 0.5 * prior_precision * ||theta||_2^2
                 = 0.5 * (theta - mu)^T H (theta - mu) + L(mu)

    Optimal Mean-Field Gaussian q*(w) = N(mu, diag(sigma_mf^2)) minimizing KL(q || p):
        sigma_{i, MF}^2 = 1 / H_{ii}
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor, noise_var: float = 1.0, prior_precision: float = 0.1):
        self.X = X.to(torch.float64)
        self.y = y.to(torch.float64)
        self.noise_var = float(noise_var)
        self.prior_precision = float(prior_precision)
        self.n_samples, self.dim = self.X.shape

        # Exact precision (Hessian of negative log posterior) and covariance
        self.H = (self.X.T @ self.X) / self.noise_var + self.prior_precision * torch.eye(
            self.dim, dtype=torch.float64
        )
        self.Sigma_exact = torch.linalg.inv(self.H)
        self.mu = (self.Sigma_exact @ (self.X.T @ self.y)) / self.noise_var

        # Candidate variance / importance quantities
        self.H_diag = torch.diag(self.H)
        self.var_mf = 1.0 / self.H_diag                      # Mean-field variance: 1 / H_ii
        self.sigma_mf = torch.sqrt(self.var_mf)              # Mean-field std: sigma_i
        self.var_exact_marginal = torch.diag(self.Sigma_exact)  # True marginal variance: [H^-1]_ii
        self.sigma_exact_marginal = torch.sqrt(self.var_exact_marginal)

    def neg_log_posterior(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Evaluate negative log posterior L(theta) directly from data (X, y) and prior.
        Supports single parameter vector (D,) or batch (B, D).
        """
        theta64 = theta.to(torch.float64)
        if theta64.ndim == 1:
            residual = self.y - self.X @ theta64
            nll = 0.5 * torch.dot(residual, residual) / self.noise_var
            prior = 0.5 * self.prior_precision * torch.dot(theta64, theta64)
            return nll + prior
        else:
            # theta64: (B, D)
            residuals = self.y.unsqueeze(0) - theta64 @ self.X.T
            nll = 0.5 * torch.sum(residuals ** 2, dim=-1) / self.noise_var
            prior = 0.5 * self.prior_precision * torch.sum(theta64 ** 2, dim=-1)
            return nll + prior


def generate_correlated_2d_data(
    n_samples: int = 1000,
    rho: float = 0.99,
    scale_x2: float = 3.0,
    w_true=(1.5, -0.8),
    noise_std: float = 1.0,
    seed: int = 42,
):
    """
    Generate 2D regression data with controlled feature correlation rho:
        x1 ~ N(0, 1)
        x2_raw = rho * x1 + sqrt(1 - rho^2) * z,  z ~ N(0, 1)
        x2 = scale_x2 * x2_raw
    Scaling x2 by scale_x2 != 1 ensures H_11 != H_22 so the two weights do not tie.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n_samples)
    z = rng.standard_normal(n_samples)
    x2 = scale_x2 * (rho * x1 + np.sqrt(max(0.0, 1.0 - rho ** 2)) * z)

    X = np.column_stack([x1, x2])
    w = np.array(w_true, dtype=np.float64)
    eps = rng.normal(0.0, noise_std, size=n_samples)
    y = X @ w + eps

    return torch.from_numpy(X), torch.from_numpy(y)


def run_test_a(seed: int = 42) -> dict:
    """
    Test A: Does the perturbation code recover the truth?
      1. Compare central second difference [L(theta+d) + L(theta-d) - 2L(theta)] / 2
         with analytic 0.5 * d^T H d both at theta = mu and at an off-optimum theta != mu.
      2. Compare with one-sided difference L(theta+d) - L(theta) off-optimum to show
         why symmetric differences are essential when weights are not exactly at the mode.
    """
    X, y = generate_correlated_2d_data(n_samples=1000, rho=0.99, scale_x2=3.0, seed=seed)
    model = ToyBayesianLinearRegression(X, y, noise_var=1.0, prior_precision=0.1)

    rng = torch.Generator().manual_seed(seed)
    deltas = 0.05 * torch.randn(50, 2, dtype=torch.float64, generator=rng)

    # 1. At posterior mode theta = mu
    theta_opt = model.mu
    central_at_opt = central_second_diff(model.neg_log_posterior, theta_opt.unsqueeze(0), deltas)
    onesided_at_opt = one_sided_diff(model.neg_log_posterior, theta_opt.unsqueeze(0), deltas)
    analytic = quadratic_damage(model.H, deltas)

    max_err_central_opt = float(torch.max(torch.abs(central_at_opt - analytic)).item())
    max_err_onesided_opt = float(torch.max(torch.abs(onesided_at_opt - analytic)).item())

    # 2. Slightly off the optimum: theta = mu + offset (simulating finite training / VI noise)
    offset = torch.tensor([0.03, -0.02], dtype=torch.float64)
    theta_off = model.mu + offset
    central_off_opt = central_second_diff(model.neg_log_posterior, theta_off.unsqueeze(0), deltas)
    onesided_off_opt = one_sided_diff(model.neg_log_posterior, theta_off.unsqueeze(0), deltas)

    # Analytic first-order contamination in one-sided diff: grad L(theta_off)^T delta = (H offset)^T delta
    expected_linear_contamination = deltas @ (model.H @ offset)

    max_err_central_off = float(torch.max(torch.abs(central_off_opt - analytic)).item())
    max_err_onesided_off = float(torch.max(torch.abs(onesided_off_opt - analytic)).item())
    max_err_onesided_explained = float(
        torch.max(torch.abs((onesided_off_opt - analytic) - expected_linear_contamination)).item()
    )

    return {
        "H": model.H.tolist(),
        "mu": model.mu.tolist(),
        "sigma_mf": model.sigma_mf.tolist(),
        "sigma_exact_marginal": model.sigma_exact_marginal.tolist(),
        "max_err_central_at_opt": max_err_central_opt,
        "max_err_onesided_at_opt": max_err_onesided_opt,
        "max_err_central_off_opt": max_err_central_off,
        "max_err_onesided_off_opt": max_err_onesided_off,
        "max_err_onesided_linear_term_check": max_err_onesided_explained,
    }


def run_test_b() -> dict:
    """
    Test B: Do coordinate directions hide the structure?
    Uses the canonical precision matrix:
        H = [[10.0, 9.9],
             [ 9.9, 10.0]]
    Evaluates damage along:
      - Co-moving eigenvector:    v_+ = (1, 1) / sqrt(2),  lambda_+ = 19.9, unit damage = 9.95
      - Coordinate axis 1:        e_1 = (1, 0),            unit damage = 5.00
      - Coordinate axis 2:        e_2 = (0, 1),            unit damage = 5.00
      - Compensating eigenvector: v_- = (1, -1) / sqrt(2), lambda_- = 0.1,  unit damage = 0.05
    """
    H = torch.tensor([[10.0, 9.9], [9.9, 10.0]], dtype=torch.float64)
    mu = torch.zeros(2, dtype=torch.float64)

    def canonical_loss(theta: torch.Tensor) -> torch.Tensor:
        theta64 = theta.to(torch.float64)
        return quadratic_damage(H, theta64 - mu)

    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    directions = {
        "eig_plus_(1,1)/sqrt2": torch.tensor([inv_sqrt2, inv_sqrt2], dtype=torch.float64),
        "coord_w1_(1,0)": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "coord_w2_(0,1)": torch.tensor([0.0, 1.0], dtype=torch.float64),
        "eig_minus_(1,-1)/sqrt2": torch.tensor([inv_sqrt2, -inv_sqrt2], dtype=torch.float64),
    }

    unit_damages = {}
    for name, d_vec in directions.items():
        measured = float(central_second_diff(canonical_loss, mu, d_vec).item())
        analytic = float(quadratic_damage(H, d_vec).item())
        mf_pred = float(diagonal_predicted_damage(torch.diag(H), d_vec).item())
        unit_damages[name] = {
            "measured_damage": measured,
            "analytic_damage": analytic,
            "mf_diagonal_predicted_damage": mf_pred,
        }

    # Step-size sweep along the 4 canonical directions
    radii = np.linspace(0.0, 1.0, 101)
    step_curves = {}
    for name, d_vec in directions.items():
        deltas = torch.from_numpy(radii[:, None]) * d_vec.unsqueeze(0)
        step_curves[name] = central_second_diff(canonical_loss, mu.unsqueeze(0), deltas).numpy()

    # Polar angle sweep: delta(phi) = (cos phi, sin phi)
    angles = np.linspace(0.0, 2.0 * np.pi, 361)
    unit_circle = torch.from_numpy(np.column_stack([np.cos(angles), np.sin(angles)]))
    true_polar_damage = central_second_diff(canonical_loss, mu.unsqueeze(0), unit_circle).numpy()
    mf_polar_damage = diagonal_predicted_damage(torch.diag(H), unit_circle).numpy()
    exact_marg_polar_damage = diagonal_predicted_damage(
        1.0 / torch.diag(torch.linalg.inv(H)), unit_circle
    ).numpy()

    return {
        "H": H.tolist(),
        "unit_damages": unit_damages,
        "anisotropy_ratio_eig_plus_over_minus": (
            unit_damages["eig_plus_(1,1)/sqrt2"]["measured_damage"]
            / unit_damages["eig_minus_(1,-1)/sqrt2"]["measured_damage"]
        ),
        "radii": radii,
        "step_curves": step_curves,
        "angles": angles,
        "true_polar_damage": true_polar_damage,
        "mf_polar_damage": mf_polar_damage,
        "exact_marg_polar_damage": exact_marg_polar_damage,
    }


def run_test_c(seed: int = 42) -> dict:
    """
    Test C: Where exactly do the scores succeed and fail?
      Part 1 (2D correlated features with x2 scaled by 3):
        Compare 1/sigma_MF^2 = H_ii, Fisher diagonal = H_ii, and 1/[H^-1]_ii on:
          - Single-weight perturbations
          - Group perturbations (same-sign vs opposite-sign)
      Part 2 (Multi-pair D=20 block-correlated model with varying scales & correlations):
        Compute Spearman rank correlation across 20 weights for single-weight perturbations
        vs. group perturbations.
        - Shows Spearman = 1.0 for 1/sigma_MF^2 and H_ii on single coordinates, while
          1/[H^-1]_ii fails when correlations vary across feature pairs.
      Part 3 (Correlation sweep rho in [0, 0.999]):
        Track the variance inflation ratio [H^-1]_ii / (1/H_ii) = 1 / (1 - r^2)
        and group-perturbation prediction error vs. correlation r.
    """
    # --- Part 1: 2D correlated model with scale_x2 = 3.0 ---
    X2d, y2d = generate_correlated_2d_data(n_samples=2000, rho=0.99, scale_x2=3.0, seed=seed)
    m2d = ToyBayesianLinearRegression(X2d, y2d, noise_var=1.0, prior_precision=0.1)

    delta_mag = 0.1
    e1 = torch.tensor([delta_mag, 0.0], dtype=torch.float64)
    e2 = torch.tensor([0.0, delta_mag], dtype=torch.float64)
    single_measured_2d = [
        float(central_second_diff(m2d.neg_log_posterior, m2d.mu, e1).item()),
        float(central_second_diff(m2d.neg_log_posterior, m2d.mu, e2).item()),
    ]
    single_mf_pred_2d = [
        float(diagonal_predicted_damage(m2d.H_diag, e1).item()),
        float(diagonal_predicted_damage(m2d.H_diag, e2).item()),
    ]

    # Normalized group moves in 2D: scaled by 1/sqrt(H_ii) so each coordinate contributes equally
    # under the diagonal mean-field model
    d_same = delta_mag * torch.tensor([1.0, 1.0], dtype=torch.float64) / np.sqrt(2.0)
    d_opp = delta_mag * torch.tensor([1.0, -1.0], dtype=torch.float64) / np.sqrt(2.0)
    group_same_measured = float(central_second_diff(m2d.neg_log_posterior, m2d.mu, d_same).item())
    group_opp_measured = float(central_second_diff(m2d.neg_log_posterior, m2d.mu, d_opp).item())
    group_diag_pred = float(diagonal_predicted_damage(m2d.H_diag, d_same).item())

    # --- Part 2: D=20 multi-pair block-correlated linear model for rigorous Spearman test ---
    # 10 pairs of features; each pair k has its own scale s_k and correlation r_k.
    rng = np.random.default_rng(seed)
    n_pairs = 10
    dim = 2 * n_pairs
    scales = np.linspace(0.5, 3.0, dim)
    pair_rhos = np.array([0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.90, 0.95, 0.98, 0.995])

    H_blocks = []
    for k in range(n_pairs):
        s1, s2 = scales[2 * k], scales[2 * k + 1]
        rk = pair_rhos[k]
        block = np.array(
            [
                [s1 ** 2, rk * s1 * s2],
                [rk * s1 * s2, s2 ** 2],
            ],
            dtype=np.float64,
        )
        H_blocks.append(block)

    H_multi = torch.from_numpy(
        np.block([[H_blocks[i] if i == j else np.zeros((2, 2)) for j in range(n_pairs)] for i in range(n_pairs)])
    )
    Sigma_multi = torch.linalg.inv(H_multi)
    mu_multi = torch.zeros(dim, dtype=torch.float64)

    def multi_loss(theta: torch.Tensor) -> torch.Tensor:
        return quadratic_damage(H_multi, theta - mu_multi)

    # Scores for each of the 20 parameters:
    score_mf_inv_var = torch.diag(H_multi)                 # 1 / sigma_{i,MF}^2 = H_ii
    score_mf_inv_std = torch.sqrt(score_mf_inv_var)        # 1 / sigma_{i,MF} = sqrt(H_ii)
    score_fisher_diag = torch.diag(H_multi)                # Exact Fisher / Hessian diagonal = H_ii
    score_exact_marg_inv_var = 1.0 / torch.diag(Sigma_multi)  # 1 / [H^-1]_ii

    # (a) Single-weight perturbations of constant magnitude delta_mag across all 20 coordinates
    eye_deltas = delta_mag * torch.eye(dim, dtype=torch.float64)
    single_damages_20d = central_second_diff(multi_loss, mu_multi.unsqueeze(0), eye_deltas)

    sp_single_mf, _ = spearman_rank_corr(score_mf_inv_var, single_damages_20d)
    sp_single_mf_std, _ = spearman_rank_corr(score_mf_inv_std, single_damages_20d)
    sp_single_fisher, _ = spearman_rank_corr(score_fisher_diag, single_damages_20d)
    sp_single_exact_marg, _ = spearman_rank_corr(score_exact_marg_inv_var, single_damages_20d)

    # (b) Pairwise group perturbations where each pair k is perturbed in its compensating direction
    # delta_pair_k = delta_mag * (1/s1, -1/s2) / norm, or standardized co-moving vs compensating
    pair_diag_scores = []
    pair_compensating_damages = []
    pair_comoving_damages = []
    for k in range(n_pairs):
        d_comp = torch.zeros(dim, dtype=torch.float64)
        # Standardized compensating perturbation (w_{2k}/s_{2k} - w_{2k+1}/s_{2k+1})
        d_comp[2 * k] = delta_mag / np.sqrt(2.0)
        d_comp[2 * k + 1] = -delta_mag / np.sqrt(2.0)

        d_co = torch.zeros(dim, dtype=torch.float64)
        d_co[2 * k] = delta_mag / np.sqrt(2.0)
        d_co[2 * k + 1] = delta_mag / np.sqrt(2.0)

        pair_diag_scores.append(float(diagonal_predicted_damage(score_mf_inv_var, d_comp).item()))
        pair_compensating_damages.append(float(central_second_diff(multi_loss, mu_multi, d_comp).item()))
        pair_comoving_damages.append(float(central_second_diff(multi_loss, mu_multi, d_co).item()))

    sp_pair_comp_vs_diag, _ = spearman_rank_corr(pair_diag_scores, pair_compensating_damages)

    # --- Part 3: Sweep correlation r from 0 to 0.999 ---
    r_values = np.concatenate([np.linspace(0.0, 0.95, 30), np.linspace(0.955, 0.999, 30)])
    variance_ratios = []
    analytic_ratios = []
    comoving_damage_ratios = []      # True damage / MF diagonal predicted damage for (1, 1)/sqrt(2)
    compensating_damage_ratios = []  # True damage / MF diagonal predicted damage for (1, -1)/sqrt(2)

    for r in r_values:
        H_r = torch.tensor([[10.0, 10.0 * r], [10.0 * r, 10.0]], dtype=torch.float64)
        Sigma_r = torch.linalg.inv(H_r)
        var_mf_r = 1.0 / H_r[0, 0].item()
        var_exact_r = Sigma_r[0, 0].item()
        variance_ratios.append(var_exact_r / var_mf_r)
        analytic_ratios.append(1.0 / (1.0 - r ** 2))

        d_plus = torch.tensor([1.0, 1.0], dtype=torch.float64) / np.sqrt(2.0)
        d_minus = torch.tensor([1.0, -1.0], dtype=torch.float64) / np.sqrt(2.0)
        diag_pred_r = float(diagonal_predicted_damage(torch.diag(H_r), d_plus).item())
        true_plus_r = float(quadratic_damage(H_r, d_plus).item())
        true_minus_r = float(quadratic_damage(H_r, d_minus).item())

        comoving_damage_ratios.append(true_plus_r / diag_pred_r)
        compensating_damage_ratios.append(true_minus_r / diag_pred_r)

    # Specific check at r = 0.0 and r = 0.99
    idx_0 = 0
    r_99 = 0.99
    ratio_r0 = 1.0 / (1.0 - 0.0 ** 2)
    ratio_r99 = 1.0 / (1.0 - r_99 ** 2)

    return {
        "part1_2d": {
            "H_diag": m2d.H_diag.tolist(),
            "var_mf": m2d.var_mf.tolist(),
            "var_exact_marginal": m2d.var_exact_marginal.tolist(),
            "single_measured_damage": single_measured_2d,
            "single_mf_predicted_damage": single_mf_pred_2d,
            "group_same_sign_measured": group_same_measured,
            "group_opp_sign_measured": group_opp_measured,
            "group_diagonal_predicted": group_diag_pred,
        },
        "part2_ranking_20d": {
            "spearman_single_mf_inv_var": sp_single_mf,
            "spearman_single_mf_inv_std": sp_single_mf_std,
            "spearman_single_fisher_diag": sp_single_fisher,
            "spearman_single_exact_marginal_inv_var": sp_single_exact_marg,
            "spearman_pair_compensating_vs_diag_pred": sp_pair_comp_vs_diag,
            "single_damages_20d": single_damages_20d.tolist(),
            "score_mf_inv_var": score_mf_inv_var.tolist(),
            "score_exact_marg_inv_var": score_exact_marg_inv_var.tolist(),
        },
        "part3_rho_sweep": {
            "r_values": r_values,
            "variance_ratios": np.array(variance_ratios),
            "analytic_ratios": np.array(analytic_ratios),
            "comoving_damage_ratios": np.array(comoving_damage_ratios),
            "compensating_damage_ratios": np.array(compensating_damage_ratios),
            "ratio_at_r_0": ratio_r0,
            "ratio_at_r_099": ratio_r99,
        },
    }


def make_phase1_figures(test_b_res: dict, test_c_res: dict, out_dir: str):
    """
    Create paper-ready figures for Phase 1:
      1. phase1_test_b_directions.png / .pdf:
         (Left) Damage 0.5 * d^T H d vs. step size along coordinate axes vs. eigenvectors of H.
         (Right) Angular sweep over direction angle phi showing true curvature vs. flat mean-field prediction.
      2. phase1_test_c_correlation_sweep.png / .pdf:
         (Left) Variance inflation ratio [H^-1]_ii / sigma_{i,MF}^2 = 1 / (1 - r^2) vs. correlation r.
         (Right) Single-weight vs. Group perturbation damage comparison (showing where MF succeeds and fails).
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
    })

    # -------------------------------------------------------------------------
    # Figure 1: Test B — Coordinate Directions vs. Eigen-Directions
    # -------------------------------------------------------------------------
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    radii = test_b_res["radii"]
    curves = test_b_res["step_curves"]

    ax1.plot(
        radii,
        curves["eig_plus_(1,1)/sqrt2"],
        color="#d62728",
        linewidth=2.4,
        label=r"Eigenvector $v_+ = (1, 1)/\sqrt{2}$ ($\lambda_+ = 19.9$, dmg$=9.95$)",
    )
    ax1.plot(
        radii,
        curves["coord_w1_(1,0)"],
        color="#1f77b4",
        linewidth=2.2,
        linestyle="--",
        label=r"Coordinate $w_1$ $(1, 0)$ & $w_2$ $(0, 1)$ (dmg$=5.00$)",
    )
    ax1.plot(
        radii,
        curves["eig_minus_(1,-1)/sqrt2"],
        color="#2ca02c",
        linewidth=2.4,
        label=r"Eigenvector $v_- = (1, -1)/\sqrt{2}$ ($\lambda_- = 0.1$, dmg$=0.05$)",
    )
    ax1.set_xlabel(r"Perturbation Step Size $\|\delta\|_2$")
    ax1.set_ylabel(r"Measured Loss Damage $\Delta L(\delta) = \frac{1}{2}\delta^\top H \delta$")
    ax1.set_title("(a) Damage Along Coordinate Axes vs. Eigenvectors of $H$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    angles_deg = np.degrees(test_b_res["angles"])
    ax2.plot(
        angles_deg,
        test_b_res["true_polar_damage"],
        color="#111111",
        linewidth=2.3,
        label=r"True Measured Damage $\frac{1}{2}\delta(\phi)^\top H \delta(\phi)$",
    )
    ax2.plot(
        angles_deg,
        test_b_res["mf_polar_damage"],
        color="#1f77b4",
        linewidth=2.0,
        linestyle="--",
        label=r"Mean-Field $\sigma_i^{-2} = H_{ii}$ Prediction (flat $= 5.0$)",
    )
    ax2.plot(
        angles_deg,
        test_b_res["exact_marg_polar_damage"],
        color="#ff7f0e",
        linewidth=2.0,
        linestyle=":",
        label=r"Marginal Variance $[H^{-1}]_{ii}^{-1}$ Prediction (flat $= 0.0995$)",
    )
    ax2.axvline(45, color="#d62728", alpha=0.35, linestyle="-.")
    ax2.axvline(135, color="#2ca02c", alpha=0.35, linestyle="-.")
    ax2.text(48, 9.1, r"$v_+$ ($45^\circ$)", color="#d62728", fontsize=8.5)
    ax2.text(138, 0.7, r"$v_-$ ($135^\circ$)", color="#2ca02c", fontsize=8.5)
    ax2.set_xlim(0, 360)
    ax2.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax2.set_xlabel(r"Perturbation Direction Angle $\phi$ (degrees, $\|\delta\|_2 = 1$)")
    ax2.set_ylabel(r"Loss Damage $\Delta L(\delta(\phi))$")
    ax2.set_title(r"(b) Directional Anisotropy ($199\times$ Gap Invisible to Coordinates)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    fig1.tight_layout()
    fig1_png = os.path.join(out_dir, "phase1_test_b_directions.png")
    fig1_pdf = os.path.join(out_dir, "phase1_test_b_directions.pdf")
    fig1.savefig(fig1_png, bbox_inches="tight")
    fig1.savefig(fig1_pdf, bbox_inches="tight")
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # Figure 2: Test C — Correlation Sweep & Single vs. Group Ranking Audit
    # -------------------------------------------------------------------------
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    sweep = test_c_res["part3_rho_sweep"]
    r_vals = sweep["r_values"]
    var_ratios = sweep["variance_ratios"]

    ax3.plot(
        r_vals,
        var_ratios,
        color="#9467bd",
        linewidth=2.4,
        label=r"Variance Inflation $[H^{-1}]_{ii} \,/\, \sigma_{i,\mathrm{MF}}^2 = \frac{1}{1 - r^2}$",
    )
    ax3.scatter([0.0, 0.99], [1.0, sweep["ratio_at_r_099"]], color="#d62728", zorder=5, s=45)
    ax3.annotate(
        f"r = 0.0: ratio = 1.00x\n(No gap)",
        xy=(0.0, 1.0),
        xytext=(0.12, 2.2),
        arrowprops=dict(arrowstyle="->", color="#444444"),
        fontsize=8.5,
    )
    ax3.annotate(
        f"r = 0.99: ratio = {sweep['ratio_at_r_099']:.1f}x",
        xy=(0.99, sweep["ratio_at_r_099"]),
        xytext=(0.52, 65.0),
        arrowprops=dict(arrowstyle="->", color="#d62728"),
        fontsize=8.5,
        color="#d62728",
    )
    ax3.set_yscale("log")
    ax3.set_xlabel(r"Feature / Weight Correlation $r$")
    ax3.set_ylabel(r"Ratio of True Marginal Variance to MF Variance (log scale)")
    ax3.set_title(r"(a) Mean-Field Variance Underestimation vs. Correlation $r$")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(loc="upper left")

    # Right panel: Ratio of True Group Damage to MF Diagonal Prediction as a function of r
    ax4.plot(
        r_vals,
        sweep["comoving_damage_ratios"],
        color="#d62728",
        linewidth=2.3,
        label=r"Co-moving Group $\delta \propto (+1, +1)$: True / MF Pred $= 1 + r$",
    )
    ax4.plot(
        r_vals,
        sweep["compensating_damage_ratios"],
        color="#2ca02c",
        linewidth=2.3,
        label=r"Compensating Group $\delta \propto (+1, -1)$: True / MF Pred $= 1 - r$",
    )
    ax4.axhline(
        1.0,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.8,
        label=r"Single-Weight Move $\delta \propto (1, 0)$: True / MF Pred $= 1.0$ (Exact)",
    )
    ax4.set_yscale("log")
    ax4.set_xlabel(r"Feature / Weight Correlation $r$")
    ax4.set_ylabel(r"True Damage / Mean-Field Predicted Damage (log scale)")
    ax4.set_title(r"(b) Why $\sigma_{\mathrm{MF}}$ Succeeds on Single Weights but Fails on Groups")
    ax4.grid(True, which="both", alpha=0.3)
    ax4.legend(loc="lower left")

    fig2.tight_layout()
    fig2_png = os.path.join(out_dir, "phase1_test_c_correlation_sweep.png")
    fig2_pdf = os.path.join(out_dir, "phase1_test_c_correlation_sweep.pdf")
    fig2.savefig(fig2_png, bbox_inches="tight")
    fig2.savefig(fig2_pdf, bbox_inches="tight")
    plt.close(fig2)

    return {
        "fig1_png": fig1_png,
        "fig1_pdf": fig1_pdf,
        "fig2_png": fig2_png,
        "fig2_pdf": fig2_pdf,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Toy Audit: Correlated Bayesian Linear Regression")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results", "phase1"),
        help="Output directory for figures and JSON summary",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 88)
    print("PHASE 1: TOY BAYESIAN LINEAR REGRESSION AUDIT (CPU, Exact & Mean-Field)")
    print("=" * 88)

    # Run Test A
    res_a = run_test_a(seed=args.seed)
    print("\n[Test A] Perturbation Pipeline Sanity Check (float64)")
    print("-" * 88)
    print(f"  Max |central_diff - 0.5*d^T H d| at optimum (theta = mu):      {res_a['max_err_central_at_opt']:.3e}")
    print(f"  Max |central_diff - 0.5*d^T H d| off optimum (theta != mu):    {res_a['max_err_central_off_opt']:.3e}")
    print(f"  Max |one_sided_diff - 0.5*d^T H d| off optimum (theta != mu):  {res_a['max_err_onesided_off_opt']:.3e}")
    print(f"  One-sided error explained by linear grad term (theta-mu)^T H d: {res_a['max_err_onesided_linear_term_check']:.3e}")
    assert res_a["max_err_central_at_opt"] < 1e-11, "Test A failed at optimum!"
    assert res_a["max_err_central_off_opt"] < 1e-11, "Test A failed off optimum!"

    # Run Test B
    res_b = run_test_b()
    print("\n[Test B] Coordinate Directions vs. Eigen-Directions (H = [[10, 9.9], [9.9, 10]])")
    print("-" * 88)
    print(f"  {'Direction':<28} | {'Measured Damage':>15} | {'Analytic 0.5 d^T H d':>20} | {'MF Diag Pred':>14}")
    print("  " + "-" * 84)
    for name, info in res_b["unit_damages"].items():
        print(
            f"  {name:<28} | {info['measured_damage']:>15.6f} | "
            f"{info['analytic_damage']:>20.6f} | {info['mf_diagonal_predicted_damage']:>14.6f}"
        )
    print(f"  Eigen-direction anisotropy ratio (v_+ / v_-): {res_b['anisotropy_ratio_eig_plus_over_minus']:.1f}x")

    # Run Test C
    res_c = run_test_c(seed=args.seed)
    p1 = res_c["part1_2d"]
    p2 = res_c["part2_ranking_20d"]
    p3 = res_c["part3_rho_sweep"]

    print("\n[Test C] Candidate Importance Scores vs. Single-Weight and Group Damage")
    print("-" * 88)
    print("  1. 2D Correlated Model (rho = 0.99, x2 scaled by 3x so H_11 != H_22):")
    print(f"     H_diag (1/sigma_MF^2):         [{p1['H_diag'][0]:.2f}, {p1['H_diag'][1]:.2f}]")
    print(f"     Mean-field variance (1/H_ii):  [{p1['var_mf'][0]:.6f}, {p1['var_mf'][1]:.6f}]")
    print(f"     Exact marginal var ([H^-1]_ii):[{p1['var_exact_marginal'][0]:.6f}, {p1['var_exact_marginal'][1]:.6f}]")
    print(f"     Single-weight measured damage: [{p1['single_measured_damage'][0]:.4f}, {p1['single_measured_damage'][1]:.4f}]")
    print(f"     Single-weight MF pred damage:  [{p1['single_mf_predicted_damage'][0]:.4f}, {p1['single_mf_predicted_damage'][1]:.4f}]")
    print(f"     Group same-sign (+d, +d) dmg:  {p1['group_same_sign_measured']:.4f} (MF diagonal predicts {p1['group_diagonal_predicted']:.4f})")
    print(f"     Group opp-sign  (+d, -d) dmg:  {p1['group_opp_sign_measured']:.4f} (MF diagonal predicts {p1['group_diagonal_predicted']:.4f})")

    print("\n  2. Multi-Pair Ranking Audit (D = 20 weights across 10 correlated feature pairs):")
    print(f"     Spearman(1/sigma_MF^2, single-weight damage):        {p2['spearman_single_mf_inv_var']:.4f} (expected 1.0000)")
    print(f"     Spearman(1/sigma_MF,   single-weight damage):        {p2['spearman_single_mf_inv_std']:.4f} (expected 1.0000)")
    print(f"     Spearman(Fisher H_ii,  single-weight damage):        {p2['spearman_single_fisher_diag']:.4f} (expected 1.0000)")
    print(f"     Spearman(1/[H^-1]_ii,  single-weight damage):        {p2['spearman_single_exact_marginal_inv_var']:.4f} (degraded by r_k)")
    print(f"     Spearman(MF diag pred, compensating pair damage):    {p2['spearman_pair_compensating_vs_diag_pred']:.4f} (fails on group move)")

    print("\n  3. Correlation Sweep (r in [0, 0.999]):")
    print(f"     Variance inflation [H^-1]_ii / (1/H_ii) at r = 0.00: {p3['ratio_at_r_0']:.2f}x")
    print(f"     Variance inflation [H^-1]_ii / (1/H_ii) at r = 0.99: {p3['ratio_at_r_099']:.2f}x")

    # Generate figures
    fig_paths = make_phase1_figures(res_b, res_c, out_dir=out_dir)
    print("\n[Figures Saved]")
    for k, path in fig_paths.items():
        print(f"  {k}: {path}")

    # Save JSON summary
    summary_path = os.path.join(out_dir, "phase1_summary.json")
    summary_payload = {
        "test_a": res_a,
        "test_b": {
            "H": res_b["H"],
            "unit_damages": res_b["unit_damages"],
            "anisotropy_ratio_eig_plus_over_minus": res_b["anisotropy_ratio_eig_plus_over_minus"],
        },
        "test_c": {
            "part1_2d": p1,
            "part2_ranking_20d": p2,
            "part3_rho_sweep_summary": {
                "ratio_at_r_0": p3["ratio_at_r_0"],
                "ratio_at_r_099": p3["ratio_at_r_099"],
            },
        },
        "figures": fig_paths,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"  summary_json: {summary_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()

