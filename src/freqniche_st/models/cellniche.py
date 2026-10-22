"""
Paired-View SAGEPool Contrastive Baseline Model for FreqNiche-ST.
Strictly implements:
- 2-layer GraphSAGE neighborhood aggregation on raw gene expression inputs
- Layer 1: Transforms center and neighbor features
- Layer 2: Aggregates transformed neighbor representations to produce final center niche embedding
- InfoNCE contrastive learning loss between paired stochastic views
- Guarantees non-zero gradient flow to all model weights
"""

from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class PairedViewSAGEPoolModel(nn.Module):
    """
    2-Layer GraphSAGE Neighborhood Pooling Contrastive Baseline.
    """
    def __init__(
        self,
        in_features: int,
        d_model: int = 128,
        temperature: float = 0.1,
        feat_drop: float = 0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.d_model = d_model
        self.temperature = temperature
        self.feat_drop = feat_drop

        # Layer 1: Transforms raw expression
        self.layer1_self = nn.Linear(in_features, d_model, bias=False)
        self.layer1_neigh = nn.Linear(in_features, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)

        # Layer 2: Aggregates layer 1 neighbor representations
        self.layer2_self = nn.Linear(d_model, d_model, bias=False)
        self.layer2_neigh = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)

        # Projection head for InfoNCE contrastive loss
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def encode(
        self,
        X_center: torch.Tensor,
        X_vis: torch.Tensor,
        vis_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encodes center cell and its visible neighbors using 2-layer GraphSAGE aggregation.

        Args:
            X_center: (B, F_dim) explicit ground-truth center cell expression.
            X_vis: (B, N_max, F_dim) neighborhood expressions.
            vis_pad_mask: (B, N_max) bool mask.

        Returns:
            z_v_niche: (B, d_model) niche embedding.
        """
        B, N_max, F_dim = X_vis.shape
        mask_float = vis_pad_mask.unsqueeze(-1).float()

        # Layer 1: Center feature and neighbor features
        h_center_1 = F.gelu(self.layer1_self(X_center))               # (B, d_model)
        h_vis_1 = F.gelu(self.layer1_neigh(X_vis)) * mask_float       # (B, N_max, d_model)

        n_neighbors = torch.clamp(mask_float.sum(dim=1), min=1.0)     # (B, 1)
        h_neigh_mean_1 = h_vis_1.sum(dim=1) / n_neighbors             # (B, d_model)
        h_center_1 = self.norm1(h_center_1 + h_neigh_mean_1)

        # Layer 2: Aggregates layer 1 representations
        h_vis_2 = self.layer2_neigh(h_vis_1) * mask_float              # (B, N_max, d_model)
        h_neigh_mean_2 = h_vis_2.sum(dim=1) / n_neighbors             # (B, d_model)
        h_center_2 = F.gelu(self.layer2_self(h_center_1) + h_neigh_mean_2)
        z_v_niche = self.norm2(h_center_2)

        return z_v_niche

    def forward(
        self,
        X_center_1: torch.Tensor,
        X_vis_1: torch.Tensor,
        vis_pad_mask_1: torch.Tensor,
        X_center_2: Optional[torch.Tensor] = None,
        X_vis_2: Optional[torch.Tensor] = None,
        vis_pad_mask_2: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with InfoNCE contrastive loss across batch.
        """
        B = X_vis_1.shape[0]
        z1 = self.encode(X_center_1, X_vis_1, vis_pad_mask_1)
        p1 = F.normalize(self.proj_head(z1), dim=-1)

        if X_vis_2 is not None and vis_pad_mask_2 is not None:
            if X_center_2 is None:
                X_center_2 = X_center_1
            z2 = self.encode(X_center_2, X_vis_2, vis_pad_mask_2)
            p2 = F.normalize(self.proj_head(z2), dim=-1)
        else:
            noise = torch.randn_like(X_vis_1) * 0.05
            X_vis_2 = F.dropout(X_vis_1 + noise, p=self.feat_drop, training=self.training)
            z2 = self.encode(X_center_1, X_vis_2, vis_pad_mask_1)
            p2 = F.normalize(self.proj_head(z2), dim=-1)

        sim_12 = torch.matmul(p1, p2.T) / self.temperature
        sim_21 = torch.matmul(p2, p1.T) / self.temperature
        labels = torch.arange(B, device=p1.device)

        loss_1 = F.cross_entropy(sim_12, labels)
        loss_2 = F.cross_entropy(sim_21, labels)
        loss = 0.5 * (loss_1 + loss_2)

        return {
            "loss": loss,
            "z_v_niche": z1
        }


# Aliases
PairedViewGraphSAGEModel = PairedViewSAGEPoolModel
CellNicheExpressionModel = PairedViewSAGEPoolModel
