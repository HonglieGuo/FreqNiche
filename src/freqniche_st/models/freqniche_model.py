"""
Unified Frequency-Conditioned Niche Diffusion (FreqNiche-DiT) Model.
Strictly implements:
- Full-Freq & Raw-Input Transformer Encoder (z_v^{niche} in R^128)
- Cosine continuous noise diffusion schedule (T = 100)
- DiT Denoiser with AdaLN-Zero modulation predicting noise epsilon
- Total Loss = L_diff + lambda_var * L_var + lambda_cov * L_cov with isfinite checks
- Deterministic inference API encode_niche()
- Zero-z diagnostic evaluation API evaluate_zero_z_loss()
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np

from .encoder import NicheTransformerEncoder
from .diffusion_schedule import CosineDiffusionSchedule
from .dit import NicheDiTDenoiser
from .loss import compute_diffusion_loss, compute_vicreg_regularization


class FreqNicheDiTModel(nn.Module):
    """
    FreqNiche-ST Main Model Architecture.
    """
    def __init__(
        self,
        in_features: int,
        k_target: int = 32,
        d_model: int = 128,
        n_enc_layers: int = 2,
        n_dit_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        timesteps: int = 100,
        mode: str = "full_freq",
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04
    ):
        super().__init__()
        self.in_features = in_features
        self.k_target = k_target
        self.d_model = d_model
        self.timesteps = timesteps
        self.mode = mode
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

        # 1. Subgraph Context Encoder
        self.encoder = NicheTransformerEncoder(
            in_features=in_features,
            d_model=d_model,
            n_layers=n_enc_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            mode=mode
        )

        # 2. Diffusion Schedule
        self.schedule = CosineDiffusionSchedule(timesteps=timesteps)

        # 3. DiT Denoiser
        self.denoiser = NicheDiTDenoiser(
            k_target=k_target,
            d_model=d_model,
            n_layers=n_dit_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio
        )

    def forward(
        self,
        X_vis: torch.Tensor,
        delta_s_vis: torch.Tensor,
        vis_pad_mask: torch.Tensor,
        H_0_target: torch.Tensor,
        delta_s_target: torch.Tensor,
        target_mask: torch.Tensor,
        L_vis_cond: Optional[torch.Tensor] = None,
        H_vis_cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Training / Evaluation forward pass.
        """
        B, M_max, k = H_0_target.shape
        device = H_0_target.device

        # 1. Encode visible nodes to get bottleneck z_v^{niche}
        z_v_niche = self.encoder(
            X_vis=X_vis,
            delta_s_vis=delta_s_vis,
            vis_pad_mask=vis_pad_mask,
            L_vis_cond=L_vis_cond,
            H_vis_cond=H_vis_cond
        )

        # 2. Sample random timesteps t ~ Uniform(1, T)
        if generator is not None:
            t = torch.randint(1, self.timesteps + 1, size=(B,), device=device, generator=generator)
            noise = torch.randn(H_0_target.shape, device=device, dtype=H_0_target.dtype, generator=generator)
        else:
            t = torch.randint(1, self.timesteps + 1, size=(B,), device=device)
            noise = torch.randn_like(H_0_target)

        # 4. Forward diffusion: compute noisy latent H_t
        H_t = self.schedule.q_sample(x_0=H_0_target, t=t, noise=noise)

        # 5. Predict noise with DiT Denoiser
        pred_noise = self.denoiser(
            H_t=H_t,
            t=t,
            z_v_niche=z_v_niche,
            delta_s_target=delta_s_target,
            target_mask=target_mask
        )

        # 6. Denoising loss
        loss_diff = compute_diffusion_loss(pred_noise, noise, target_mask)

        # 7. VICReg regularizations
        loss_var, loss_cov = compute_vicreg_regularization(z_v_niche)

        total_loss = loss_diff + self.lambda_var * loss_var + self.lambda_cov * loss_cov

        assert torch.isfinite(total_loss).all(), f"Non-finite loss encountered in training forward: {total_loss}"

        return {
            "loss": total_loss,
            "loss_diff": loss_diff,
            "loss_var": loss_var,
            "loss_cov": loss_cov,
            "z_v_niche": z_v_niche
        }

    @torch.no_grad()
    def encode_niche(
        self,
        X_vis: torch.Tensor,
        delta_s_vis: torch.Tensor,
        vis_pad_mask: torch.Tensor,
        L_vis_cond: Optional[torch.Tensor] = None,
        H_vis_cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Deterministic inference API: extracts bottleneck z_v^{niche} for evaluation.
        """
        self.eval()
        return self.encoder(
            X_vis=X_vis,
            delta_s_vis=delta_s_vis,
            vis_pad_mask=vis_pad_mask,
            L_vis_cond=L_vis_cond,
            H_vis_cond=H_vis_cond
        )

    @torch.no_grad()
    def evaluate_zero_z_loss(
        self,
        X_vis: torch.Tensor,
        delta_s_vis: torch.Tensor,
        vis_pad_mask: torch.Tensor,
        H_0_target: torch.Tensor,
        delta_s_target: torch.Tensor,
        target_mask: torch.Tensor,
        L_vis_cond: Optional[torch.Tensor] = None,
        H_vis_cond: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Diagnostic API: evaluates held-out diffusion loss with true z_v vs Zeroed-z_v.
        """
        self.eval()
        B, M_max, k = H_0_target.shape
        device = H_0_target.device

        # If no valid targets exist in batch, return safe 0
        if not target_mask.any():
            return {
                "loss_normal": 0.0,
                "loss_zero_z": 0.0,
                "loss_increase_pct": 0.0
            }

        # Deterministic paired noise & timesteps for exact ablation comparison
        t = torch.randint(1, self.timesteps + 1, size=(B,), device=device)
        noise = torch.randn_like(H_0_target)
        H_t = self.schedule.q_sample(x_0=H_0_target, t=t, noise=noise)

        # 1. Normal prediction with real z_v^{niche}
        z_v_niche = self.encoder(
            X_vis=X_vis,
            delta_s_vis=delta_s_vis,
            vis_pad_mask=vis_pad_mask,
            L_vis_cond=L_vis_cond,
            H_vis_cond=H_vis_cond
        )
        pred_normal = self.denoiser(H_t, t, z_v_niche, delta_s_target, target_mask)
        loss_normal = float(compute_diffusion_loss(pred_normal, noise, target_mask).item())

        # 2. Zero-shot ablation: zero out z_v^{niche}
        z_zero = torch.zeros_like(z_v_niche)
        pred_zero = self.denoiser(H_t, t, z_zero, delta_s_target, target_mask)
        loss_zero = float(compute_diffusion_loss(pred_zero, noise, target_mask).item())

        if np.isnan(loss_normal) or np.isnan(loss_zero):
            loss_increase_pct = 0.0
        else:
            loss_increase_pct = ((loss_zero - loss_normal) / max(loss_normal, 1e-6)) * 100.0

        return {
            "loss_normal": loss_normal,
            "loss_zero_z": loss_zero,
            "loss_increase_pct": float(loss_increase_pct)
        }
