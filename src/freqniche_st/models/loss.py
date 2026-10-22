"""
Loss functions and latent space regularizers for FreqNiche-ST.
Strictly implements:
- Diffusion MSE denoising loss with target padding mask
- VICReg-style variance regularization (enforcing Var(z) >= 1.0)
- VICReg-style covariance decorrelation regularization (penalizing off-diagonal Cov(z)^2)
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_diffusion_loss(
    pred_noise: torch.Tensor,
    target_noise: torch.Tensor,
    target_mask: torch.Tensor
) -> torch.Tensor:
    """
    Computes masked mean-squared error for noise prediction.

    Args:
        pred_noise: (B, M_max, k) predicted Gaussian noise.
        target_noise: (B, M_max, k) ground truth noise epsilon.
        target_mask: (B, M_max) bool mask, True for valid target cells.

    Returns:
        Scalar MSE loss averaged over valid target tokens.
    """
    # (B, M_max, k)
    diff_sq = (pred_noise - target_noise) ** 2
    # Sum over feature dimension k -> (B, M_max)
    token_loss = diff_sq.sum(dim=-1)

    # Mask out padding tokens
    mask_float = target_mask.float()
    n_valid = mask_float.sum()

    if n_valid > 0:
        return (token_loss * mask_float).sum() / n_valid
    else:
        return torch.tensor(0.0, device=pred_noise.device, requires_grad=True)


def compute_vicreg_regularization(
    z: torch.Tensor,
    gamma: float = 1.0,
    eps: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes VICReg-style variance and covariance regularization on bottleneck z.
    Prevents latent dimension collapse and decorrelates embedding dimensions.

    Args:
        z: (B, d) bottleneck representations.
        gamma: Target standard deviation threshold (1.0).
        eps: Small constant for numerical stability.

    Returns:
        loss_var: Variance loss, hinge penalty if std(z_j) < gamma.
        loss_cov: Covariance loss, sum of squared off-diagonal entries in Cov(z).
    """
    B, d = z.shape
    if B <= 1:
        zero_t = torch.tensor(0.0, device=z.device, requires_grad=True)
        return zero_t, zero_t

    # 1. Center z
    z_centered = z - z.mean(dim=0, keepdim=True)

    # 2. Variance loss: max(0, gamma - std(z_j))
    std = torch.sqrt(z_centered.var(dim=0, unbiased=False) + eps)
    loss_var = torch.mean(F.relu(gamma - std))

    # 3. Covariance loss: off-diagonal covariance matrix penalty
    cov = (z_centered.T @ z_centered) / (B - 1)
    diag_mask = torch.eye(d, device=z.device, dtype=torch.bool)
    off_diag_cov = cov.masked_select(~diag_mask)
    loss_cov = (off_diag_cov ** 2).sum() / d

    return loss_var, loss_cov
