"""
Diffusion Transformer (DiT) Denoiser for FreqNiche-ST.
Strictly implements:
- Input: Noisy target latent H_t in R^{B x M_max x k_target} (k_target = 32)
- Conditioning on bottleneck z_v^{niche} in R^{B x 128}, relative coords Delta s_{vm}, and timestep t
- 2-layer DiT Block with zero-initialized AdaLN modulation
- Target attention mask handling (padding up to M_max = 32) with NaN guards
- Output predicted Gaussian noise epsilon_hat in R^{B x M_max x k_target}
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_schedule import get_sinusoidal_timestep_embedding


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Applies AdaLN scale and shift: x * (1 + scale) + shift."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block with AdaLN-Zero modulation and Cross-Attention.
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        d_ff = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        # AdaLN modulation MLP: outputs (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        z_niche: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, M_max, d_model) target token features.
            c: (B, d_model) condition vector (timestep + bottleneck).
            z_niche: (B, d_model) bottleneck niche condition.
            key_padding_mask: (B, M_max) bool mask, True where target is PADDED (ignored).
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        # NaN guard: if a row in key_padding_mask is ALL True, unmask index 0 to avoid softmax NaN
        safe_pad_mask = key_padding_mask
        if key_padding_mask is not None:
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                safe_pad_mask = key_padding_mask.clone()
                safe_pad_mask[all_masked, 0] = False

        # 1. Modulated Self-Attention on target tokens
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=safe_pad_mask,
            need_weights=False
        )
        x = x + gate_msa.unsqueeze(1) * attn_out

        # 2. Cross-Attention on z_niche condition
        z_context = z_niche.unsqueeze(1)
        x_cross = self.norm_cross(x)
        cross_out, _ = self.cross_attn(
            query=x_cross,
            key=z_context,
            value=z_context,
            need_weights=False
        )
        x = x + cross_out

        # 3. Modulated MLP
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm2)

        return x


class NicheDiTDenoiser(nn.Module):
    """
    Complete Diffusion Transformer Denoiser.
    Predicts Gaussian noise epsilon_hat on masked target high-frequency residuals.
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

        self.proj_target = nn.Linear(k_target, d_model)
        self.proj_pos = nn.Linear(2, d_model)

        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

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
        H_t: torch.Tensor,
        t: torch.Tensor,
        z_v_niche: torch.Tensor,
        delta_s_target: torch.Tensor,
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward denoising pass.
        """
        B, M_max, _ = H_t.shape

        x = self.proj_target(H_t) + self.proj_pos(delta_s_target)

        t_emb = get_sinusoidal_timestep_embedding(t, self.d_model)
        t_feat = self.time_embed(t_emb)
        niche_feat = self.proj_niche(z_v_niche)
        c = t_feat + niche_feat

        key_padding_mask = ~target_mask

        for block in self.blocks:
            x = block(x, c=c, z_niche=niche_feat, key_padding_mask=key_padding_mask)

        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x_norm = modulate(self.final_norm(x), shift, scale)
        eps_hat = self.final_proj(x_norm)

        return eps_hat
