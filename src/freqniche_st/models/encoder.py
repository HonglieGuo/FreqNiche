"""
Lightweight Transformer Subgraph Encoder for FreqNiche-ST.
Strictly implements:
- Full-Freq Token: e_j = W_R X_j + W_L L_j^{cond} + W_H H_j^{cond} + W_s Delta s_{vj}
- Raw Token: e_j = W_R X_j + W_s Delta s_{vj}
- Learnable [NICHE] token prepended at index 0
- 2-layer Pre-LayerNorm Transformer (d=128, heads=4, MLP expansion 2.0)
- Padding attention mask handling
- Output bottleneck z_v^{niche} = Encoder(...)[NICHE] in R^128
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderBlock(nn.Module):
    """Pre-LayerNorm Transformer Encoder Layer with Residual Connections."""
    def __init__(self, d_model: int = 128, n_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        d_ff = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, SeqLen, d_model)
            key_padding_mask: (B, SeqLen) bool mask, True where token is PADDED (ignored).
        """
        # Pre-LN Self-Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + attn_out

        # Pre-LN MLP
        x = x + self.mlp(self.norm2(x))
        return x


class NicheTransformerEncoder(nn.Module):
    """
    2-layer Lightweight Transformer Subgraph Encoder.
    Encodes visible neighbor tokens and outputs the bottleneck niche representation z_v^{niche}.
    """
    def __init__(
        self,
        in_features: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        mode: str = "full_freq"  # "full_freq" or "raw_input"
    ):
        super().__init__()
        self.in_features = in_features
        self.d_model = d_model
        self.mode = mode

        # Linear projections for Full-Freq tokens
        self.proj_raw = nn.Linear(in_features, d_model, bias=False)
        self.proj_pos = nn.Linear(2, d_model, bias=True)

        if mode == "full_freq":
            self.proj_low = nn.Linear(in_features, d_model, bias=False)
            self.proj_high = nn.Linear(in_features, d_model, bias=False)
        elif mode == "concat_raw_low":
            self.proj_concat = nn.Linear(in_features * 2, d_model, bias=False)

        # Learnable [NICHE] token embedding
        self.niche_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Pre-LN Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        X_vis: torch.Tensor,
        delta_s_vis: torch.Tensor,
        vis_pad_mask: torch.Tensor,
        L_vis_cond: Optional[torch.Tensor] = None,
        H_vis_cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward encoding pass on visible subgraph tokens.

        Args:
            X_vis: (B, N_vis_max, F) raw expression of visible nodes.
            delta_s_vis: (B, N_vis_max, 2) relative spatial coordinates.
            vis_pad_mask: (B, N_vis_max) bool mask, True for REAL visible nodes, False for PADDED.
            L_vis_cond: (B, N_vis_max, F) target-excluded low frequency expression.
            H_vis_cond: (B, N_vis_max, F) target-excluded high frequency expression.

        Returns:
            z_v_niche: (B, d_model) bottleneck niche token representation.
        """
        B, N_max, _ = X_vis.shape

        # 1. Project tokens according to mode
        pos_emb = self.proj_pos(delta_s_vis)

        if self.mode == "full_freq":
            assert L_vis_cond is not None and H_vis_cond is not None, "Full-Freq requires L and H conditions!"
            e_j = self.proj_raw(X_vis) + self.proj_low(L_vis_cond) + self.proj_high(H_vis_cond) + pos_emb
        elif self.mode == "concat_raw_low":
            assert L_vis_cond is not None, "Concat requires L condition!"
            concat_input = torch.cat([X_vis, L_vis_cond], dim=-1)
            e_j = self.proj_concat(concat_input) + pos_emb
        else:  # raw_input
            e_j = self.proj_raw(X_vis) + pos_emb

        # 2. Prepend [NICHE] token at position 0
        niche_tokens = self.niche_token.expand(B, -1, -1)  # (B, 1, d_model)
        seq = torch.cat([niche_tokens, e_j], dim=1)         # (B, 1 + N_max, d_model)

        # 3. Construct key padding mask (True where token is IGNORED by PyTorch MultiheadAttention)
        # [NICHE] token is never ignored -> False
        niche_pad = torch.zeros((B, 1), dtype=torch.bool, device=vis_pad_mask.device)
        # For visible nodes, True in vis_pad_mask means VALID -> in PyTorch MHA, ignored = (~vis_pad_mask)
        vis_ignored = ~vis_pad_mask
        key_padding_mask = torch.cat([niche_pad, vis_ignored], dim=1)  # (B, 1 + N_max)

        # 4. Pass through Transformer encoder blocks
        for block in self.blocks:
            seq = block(seq, key_padding_mask=key_padding_mask)

        seq = self.final_norm(seq)

        # 5. Extract [NICHE] token at index 0 as bottleneck z_v^{niche}
        z_v_niche = seq[:, 0, :]  # (B, d_model)

        return z_v_niche
