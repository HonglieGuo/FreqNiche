"""
Same-Encoder Masked Autoencoder (MAE) Baseline for FreqNiche-ST.
Strictly implements:
- Identical 2-layer Transformer Encoder producing bottleneck z_v^{niche} in R^128
- Deterministic 2-layer MLP/Cross-Attention Decoder directly predicting H_0^{target}
- Target padding mask handling
- Masked MSE reconstruction loss with VICReg regularizers
"""

from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import NicheTransformerEncoder
from .loss import compute_vicreg_regularization


class MAEDecoder(nn.Module):
    """Deterministic decoder for Masked Autoencoder baseline."""
    def __init__(self, k_target: int = 32, d_model: int = 128, mlp_ratio: float = 2.0):
        super().__init__()
        self.proj_pos = nn.Linear(2, d_model)
        self.proj_niche = nn.Linear(d_model, d_model)
        
        d_ff = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, k_target)
        )

    def forward(
        self,
        z_v_niche: torch.Tensor,
        delta_s_target: torch.Tensor,
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_v_niche: (B, d_model) bottleneck niche condition.
            delta_s_target: (B, M_max, 2) relative positions of target cells.
            target_mask: (B, M_max) bool mask.

        Returns:
            H_0_pred: (B, M_max, k_target) predicted clean target latent.
        """
        pos_emb = self.proj_pos(delta_s_target)           # (B, M_max, d_model)
        niche_emb = self.proj_niche(z_v_niche).unsqueeze(1) # (B, 1, d_model)
        
        query = pos_emb + niche_emb                       # (B, M_max, d_model)
        H_0_pred = self.net(query)                        # (B, M_max, k_target)

        return H_0_pred


class NicheMAEModel(nn.Module):
    """
    Complete Masked Autoencoder (MAE) baseline model.
    """
    def __init__(
        self,
        in_features: int,
        k_target: int = 32,
        d_model: int = 128,
        n_enc_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        mode: str = "full_freq",
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04
    ):
        super().__init__()
        self.encoder = NicheTransformerEncoder(
            in_features=in_features,
            d_model=d_model,
            n_layers=n_enc_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            mode=mode
        )
        self.decoder = MAEDecoder(
            k_target=k_target,
            d_model=d_model,
            mlp_ratio=mlp_ratio
        )
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

    def forward(
        self,
        X_vis: torch.Tensor,
        delta_s_vis: torch.Tensor,
        vis_pad_mask: torch.Tensor,
        H_0_target: torch.Tensor,
        delta_s_target: torch.Tensor,
        target_mask: torch.Tensor,
        L_vis_cond: Optional[torch.Tensor] = None,
        H_vis_cond: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes forward pass and deterministic reconstruction loss.
        """
        # 1. Encode visible nodes into niche bottleneck
        z_v_niche = self.encoder(
            X_vis=X_vis,
            delta_s_vis=delta_s_vis,
            vis_pad_mask=vis_pad_mask,
            L_vis_cond=L_vis_cond,
            H_vis_cond=H_vis_cond
        )

        # 2. Decode target positions deterministically
        H_0_pred = self.decoder(
            z_v_niche=z_v_niche,
            delta_s_target=delta_s_target,
            target_mask=target_mask
        )

        # 3. Compute masked MSE loss
        diff_sq = (H_0_pred - H_0_target) ** 2
        token_loss = diff_sq.sum(dim=-1)  # (B, M_max)
        mask_float = target_mask.float()
        n_valid = mask_float.sum()
        loss_recon = (token_loss * mask_float).sum() / max(n_valid, 1.0)

        # 4. Compute regularizations
        loss_var, loss_cov = compute_vicreg_regularization(z_v_niche)
        total_loss = loss_recon + self.lambda_var * loss_var + self.lambda_cov * loss_cov

        return {
            "loss": total_loss,
            "loss_recon": loss_recon,
            "loss_var": loss_var,
            "loss_cov": loss_cov,
            "z_v_niche": z_v_niche,
            "H_0_pred": H_0_pred
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
