"""
Capacity-Matched Deterministic Transformer Reconstruction Model for FreqNiche-ST.
Matches the architectural capacity of FreqNicheDiTModel:
- Identical NicheEncoder (Transformer blocks, d_model=128, n_heads=4, mlp_ratio=2.0)
- Identical 2-layer Transformer Decoder architecture (Self-Attn, Cross-Attn, MLP, and z-driven AdaLN-Zero)
- Shared learnable mask token + positional encoding for target query inputs
- Deterministic single-step prediction of whitened high-frequency residual H_0_target
- Strict target isolation: target expressions and target residuals do NOT enter decoder inputs
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import NicheTransformerEncoder
from .dit import DiTBlock, modulate
from .loss import compute_vicreg_regularization


def compute_reconstruction_loss(
    pred_h: torch.Tensor,
    target_h: torch.Tensor,
    target_mask: torch.Tensor
) -> torch.Tensor:
    """Masked reconstruction loss on valid target nodes."""
    diff_sq = (pred_h - target_h) ** 2
    token_loss = diff_sq.sum(dim=-1)
    mask_float = target_mask.float()
    n_valid = mask_float.sum()
    if n_valid > 0:
        return (token_loss * mask_float).sum() / n_valid
    else:
        return torch.tensor(0.0, device=pred_h.device, requires_grad=True)


class NicheTransformerDecoder(nn.Module):
    """
    Capacity-matched deterministic Transformer decoder.
    Reuses the 2-layer DiTBlock architecture and z_niche-driven AdaLN modulation,
    without timestep embedding or diffusion noise injection.
    """
    def __init__(
        self,
        k_target: int = 32,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: float = 2.0
    ):
        super().__init__()
        self.k_target = k_target
        self.d_model = d_model

        # Shared learnable mask token for permutation-equivariant target queries
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        self.proj_pos = nn.Linear(2, d_model)

        self.proj_niche = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        self.blocks = nn.ModuleList([
            DiTBlock(d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True)
        )
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)

        self.final_proj = nn.Linear(d_model, k_target, bias=True)
        nn.init.constant_(self.final_proj.weight, 0)
        nn.init.constant_(self.final_proj.bias, 0)

    def forward(
        self,
        z_v_niche: torch.Tensor,
        delta_s_target: torch.Tensor,
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward deterministic reconstruction pass.
        Args:
            z_v_niche: (B, d_model) latent niche representation from Encoder.
            delta_s_target: (B, M_max, 2) relative spatial coordinates of masked target nodes.
            target_mask: (B, M_max) bool mask, True where target node is active (non-padded).
        Returns:
            H_0_hat: (B, M_max, k_target) predicted whitened target residuals.
        """
        B, M_max, _ = delta_s_target.shape

        # Query token is shared mask token + spatial offset embedding
        x = self.mask_token.expand(B, M_max, -1) + self.proj_pos(delta_s_target)

        # Conditioning vector c is driven purely by z_niche (no timestep embedding)
        niche_feat = self.proj_niche(z_v_niche)
        c = niche_feat

        key_padding_mask = ~target_mask

        for block in self.blocks:
            x = block(x, c=c, z_niche=niche_feat, key_padding_mask=key_padding_mask)

        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x_norm = modulate(self.final_norm(x), shift, scale)
        H_0_hat = self.final_proj(x_norm)

        return H_0_hat


class NicheTransformerReconModel(nn.Module):
    """
    Complete Model combining NicheTransformerEncoder and Capacity-Matched NicheTransformerDecoder.
    Trained with deterministic MSE reconstruction of whitened target residuals + VICReg regularization.
    """
    def __init__(
        self,
        in_features: int,
        k_target: int = 32,
        d_model: int = 128,
        mode: str = "full_freq",
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04
    ):
        super().__init__()
        self.in_features = in_features
        self.k_target = k_target
        self.d_model = d_model
        self.mode = mode
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

        self.encoder = NicheTransformerEncoder(
            in_features=in_features,
            d_model=d_model,
            n_layers=n_enc_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            mode=mode
        )

        self.decoder = NicheTransformerDecoder(
            k_target=k_target,
            d_model=d_model,
            n_layers=n_dec_layers,
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
        H_vis_cond: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        Calculates deterministic MSE loss on whitened target residuals and VICReg loss.
        """
        z_v_niche = self.encoder(
            X_vis=X_vis,
            delta_s_vis=delta_s_vis,
            vis_pad_mask=vis_pad_mask,
            L_vis_cond=L_vis_cond,
            H_vis_cond=H_vis_cond
        )

        H_0_hat = self.decoder(
            z_v_niche=z_v_niche,
            delta_s_target=delta_s_target,
            target_mask=target_mask
        )

        # Masked MSE loss on valid target nodes
        loss_recon = compute_reconstruction_loss(H_0_hat, H_0_target, target_mask)

        # VICReg regularization
        loss_var, loss_cov = compute_vicreg_regularization(z_v_niche)

        total_loss = loss_recon + self.lambda_var * loss_var + self.lambda_cov * loss_cov

        assert torch.isfinite(total_loss).all(), f"Non-finite loss in training forward: {total_loss}"

        return {
            "loss": total_loss,
            "loss_recon": loss_recon,
            "loss_var": loss_var,
            "loss_cov": loss_cov,
            "z_v_niche": z_v_niche,
            "H_0_hat": H_0_hat
        }

    @torch.no_grad()
    def encode_niche(
        self,
        X_vis: torch.Tensor,
        delta_s_vis: torch.Tensor,
        vis_pad_mask: torch.Tensor,
        L_vis_cond: torch.Tensor,
        H_vis_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference encoding bottleneck representation z_v_niche.
        """
        return self.encoder(
            X_vis=X_vis,
            delta_s_vis=delta_s_vis,
            vis_pad_mask=vis_pad_mask,
            L_vis_cond=L_vis_cond,
            H_vis_cond=H_vis_cond
        )
