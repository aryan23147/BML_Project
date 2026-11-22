import numpy as np
import torch
from scipy.stats import spearmanr


def central_second_diff(loss_fn, theta: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Compute the symmetric central second difference:
        [L(theta + delta) + L(theta - delta) - 2 * L(theta)] / 2

    Uses float64 evaluation to avoid catastrophic cancellation for small perturbations.
    For a quadratic loss L(theta) = 0.5 * (theta - mu)^T H (theta - mu) + c,
    this equals 0.5 * delta^T H delta everywhere, even when theta != mu.
    """
    theta64 = theta.detach().to(torch.float64)
    delta64 = delta.detach().to(torch.float64)

    with torch.no_grad():
        l_center = loss_fn(theta64)
        l_plus = loss_fn(theta64 + delta64)
        l_minus = loss_fn(theta64 - delta64)

    return 0.5 * (l_plus + l_minus - 2.0 * l_center)


def one_sided_diff(loss_fn, theta: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Compute the one-sided perturbation difference:
        L(theta + delta) - L(theta)

    Unlike central_second_diff, this includes the first-order gradient term
    grad L(theta)^T delta when theta is not strictly at a stationary point.
    """
    theta64 = theta.detach().to(torch.float64)
    delta64 = delta.detach().to(torch.float64)

    with torch.no_grad():
        l_center = loss_fn(theta64)
        l_plus = loss_fn(theta64 + delta64)

    return l_plus - l_center


def quadratic_damage(H: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Exact quadratic Taylor damage: 0.5 * delta^T H delta.
    Supports single vector delta (D,) or batch of deltas (B, D).
    """
    H64 = H.detach().to(torch.float64)
    delta64 = delta.detach().to(torch.float64)
    if delta64.ndim == 1:
        return 0.5 * torch.dot(delta64, H64 @ delta64)
    return 0.5 * torch.sum((delta64 @ H64) * delta64, dim=-1)


def diagonal_predicted_damage(diag_precision: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Damage predicted by a diagonal curvature / importance approximation d_i:
        0.5 * sum_i d_i * delta_i^2
    Supports single vector delta (D,) or batch of deltas (B, D).
    """
    d64 = diag_precision.detach().to(torch.float64)
    delta64 = delta.detach().to(torch.float64)
    return 0.5 * torch.sum(d64 * (delta64 ** 2), dim=-1)


def spearman_rank_corr(scores, damages):
    """
    Compute Spearman rank correlation with consistent average tie-breaking.
    Returns (correlation, p_value).
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(damages, torch.Tensor):
        damages = damages.detach().cpu().numpy()

    scores = np.asarray(scores, dtype=np.float64).ravel()
    damages = np.asarray(damages, dtype=np.float64).ravel()

    res = spearmanr(scores, damages)
    return float(res.statistic), float(res.pvalue)

