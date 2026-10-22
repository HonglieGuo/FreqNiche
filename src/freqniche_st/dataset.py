"""
PyTorch Dataset and Collate Functions for Subgraph Training in FreqNiche-ST.
Strictly handles:
- Training: Dynamic target-excluded subgraph extraction with random masking
- Validation Objective: Deterministic fixed masking per center node (seed + center_idx) for stable validation loss
- Deterministic Paired View: Deterministic second view generation for contrastive baseline in validation
- Inference: Deterministic full-subgraph extraction (zero masking, all neighbors visible)
- True Center Expression: Explicitly passes X_center_true for downstream baseline models
- Condition vs Target Isolation: Accepts X_cond_global and bundle_map for topological ablations
"""

from typing import List, Dict, Any, Optional
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from .subgraph import sample_subgraph_and_mask, extract_subgraph_inference
from .whitening import TrainOnlyWhitening


class SubgraphDataset(Dataset):
    """
    Cell-Centric Subgraph Dataset for FreqNiche-ST and Baselines.
    """
    def __init__(
        self,
        center_indices: np.ndarray,
        A_global: sp.csr_matrix,
        P_global: sp.csr_matrix,
        X_global: np.ndarray,
        coords_global: np.ndarray,
        X_cond_global: Optional[np.ndarray] = None,
        A_cond_global: Optional[sp.csr_matrix] = None,
        bundle_map: Optional[np.ndarray] = None,
        whitener: Optional[TrainOnlyWhitening] = None,
        k_max: int = 64,
        m_max: int = 32,
        rho: float = 0.5,
        p_center_mask: float = 0.30,
        is_train: bool = True,
        is_val_objective: bool = False,
        return_paired_view: bool = False,
        seed: int = 42,
        data_cache=None
    ):
        self.center_indices = np.asarray(center_indices, dtype=np.int64)
        self.A_global = A_global
        self.P_global = P_global
        self.X_global = X_global
        self.coords_global = coords_global
        self.X_cond_global = X_cond_global if X_cond_global is not None else X_global
        self.A_cond_global = A_cond_global
        self.bundle_map = bundle_map
        self.whitener = whitener
        self.k_max = k_max
        self.m_max = m_max
        self.rho = rho
        self.p_center_mask = p_center_mask
        self.is_train = is_train
        self.is_val_objective = is_val_objective
        self.return_paired_view = return_paired_view
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.data_cache = data_cache

    def __len__(self) -> int:
        return len(self.center_indices)

    def _pad_and_format_subgraph(self, sub_res: Dict[str, Any], center_idx: int) -> Dict[str, torch.Tensor]:
        n_vis = len(sub_res["V_vis"])
        F_dim = self.X_global.shape[1]

        X_vis_pad = np.zeros((self.k_max, F_dim), dtype=np.float32)
        delta_s_vis_pad = np.zeros((self.k_max, 2), dtype=np.float32)
        L_vis_pad = np.zeros((self.k_max, F_dim), dtype=np.float32)
        H_vis_pad = np.zeros((self.k_max, F_dim), dtype=np.float32)
        vis_mask = np.zeros(self.k_max, dtype=bool)

        actual_vis = min(n_vis, self.k_max)
        if actual_vis > 0:
            X_vis_pad[:actual_vis] = sub_res["X_vis"][:actual_vis]
            delta_s_vis_pad[:actual_vis] = sub_res["delta_s_vis"][:actual_vis]
            L_vis_pad[:actual_vis] = sub_res["L_vis_cond"][:actual_vis]
            H_vis_pad[:actual_vis] = sub_res["H_vis_cond"][:actual_vis]
            vis_mask[:actual_vis] = True

        k_target = self.whitener.n_components_fitted_ if (self.whitener and self.whitener.is_fitted) else 32
        H_0_target_pad = np.zeros((self.m_max, k_target), dtype=np.float32)
        delta_s_target_pad = np.zeros((self.m_max, 2), dtype=np.float32)
        target_mask = sub_res["target_mask"]

        n_targets = len(sub_res["M_target"])
        actual_targets = min(n_targets, self.m_max)

        if actual_targets > 0 and self.whitener and self.whitener.is_fitted:
            H_raw_target = sub_res["H_target"][:actual_targets]
            H_0_target = self.whitener.transform(H_raw_target)
            H_0_target_pad[:actual_targets] = H_0_target

            center_coord = self.coords_global[center_idx:center_idx+1]
            target_coords = self.coords_global[sub_res["M_target"][:actual_targets]]
            delta_s_target_pad[:actual_targets] = (target_coords - center_coord).astype(np.float32)

        return {
            "center_idx": center_idx,
            "is_isolated": sub_res["is_isolated"],
            "X_center_true": torch.from_numpy(sub_res["X_center_true"]),
            "X_vis": torch.from_numpy(X_vis_pad),
            "delta_s_vis": torch.from_numpy(delta_s_vis_pad),
            "L_vis_cond": torch.from_numpy(L_vis_pad),
            "H_vis_cond": torch.from_numpy(H_vis_pad),
            "vis_pad_mask": torch.from_numpy(vis_mask),
            "H_0_target": torch.from_numpy(H_0_target_pad),
            "delta_s_target": torch.from_numpy(delta_s_target_pad),
            "target_mask": torch.from_numpy(target_mask)
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        center_idx = int(self.center_indices[idx])
        bundle_center_idx = int(self.bundle_map[center_idx]) if self.bundle_map is not None else center_idx

        if self.is_train:
            sub_res_1 = sample_subgraph_and_mask(
                center_idx=center_idx,
                A_global=self.A_global,
                P_global=self.P_global,
                X_global=self.X_global,
                coords_global=self.coords_global,
                X_cond_global=self.X_cond_global,
                A_cond_global=self.A_cond_global,
                bundle_center_idx=bundle_center_idx,
                k_max=self.k_max,
                rho=self.rho,
                p_center_mask=self.p_center_mask,
                m_max=self.m_max,
                rng=self.rng,
                data_cache=self.data_cache
            )
            item = self._pad_and_format_subgraph(sub_res_1, center_idx)

            if self.return_paired_view:
                sub_res_2 = sample_subgraph_and_mask(
                    center_idx=center_idx,
                    A_global=self.A_global,
                    P_global=self.P_global,
                    X_global=self.X_global,
                    coords_global=self.coords_global,
                    X_cond_global=self.X_cond_global,
                    A_cond_global=self.A_cond_global,
                    bundle_center_idx=bundle_center_idx,
                    k_max=self.k_max,
                    rho=self.rho,
                    p_center_mask=self.p_center_mask,
                    m_max=self.m_max,
                    rng=self.rng,
                    data_cache=self.data_cache
                )
                item_2 = self._pad_and_format_subgraph(sub_res_2, center_idx)
                item["X_vis_2"] = item_2["X_vis"]
                item["vis_pad_mask_2"] = item_2["vis_pad_mask"]

            return item

        elif self.is_val_objective:
            # Deterministic fixed masking per center node for stable validation objective evaluation
            node_rng = np.random.RandomState((self.seed + center_idx) % (2**31 - 1))
            sub_res = sample_subgraph_and_mask(
                center_idx=center_idx,
                A_global=self.A_global,
                P_global=self.P_global,
                X_global=self.X_global,
                coords_global=self.coords_global,
                X_cond_global=self.X_cond_global,
                A_cond_global=self.A_cond_global,
                bundle_center_idx=bundle_center_idx,
                k_max=self.k_max,
                rho=self.rho,
                p_center_mask=self.p_center_mask,
                m_max=self.m_max,
                rng=node_rng,
                data_cache=self.data_cache
            )
            item = self._pad_and_format_subgraph(sub_res, center_idx)

            if self.return_paired_view:
                node_rng_2 = np.random.RandomState((self.seed + center_idx + 500000) % (2**31 - 1))
                sub_res_2 = sample_subgraph_and_mask(
                    center_idx=center_idx,
                    A_global=self.A_global,
                    P_global=self.P_global,
                    X_global=self.X_global,
                    coords_global=self.coords_global,
                    X_cond_global=self.X_cond_global,
                    A_cond_global=self.A_cond_global,
                    bundle_center_idx=bundle_center_idx,
                    k_max=self.k_max,
                    rho=self.rho,
                    p_center_mask=self.p_center_mask,
                    m_max=self.m_max,
                    rng=node_rng_2,
                    data_cache=self.data_cache
                )
                item_2 = self._pad_and_format_subgraph(sub_res_2, center_idx)
                item["X_vis_2"] = item_2["X_vis"]
                item["vis_pad_mask_2"] = item_2["vis_pad_mask"]

            return item

        else:
            # Deterministic inference path: NO MASKING
            sub_res = extract_subgraph_inference(
                center_idx=center_idx,
                A_global=self.A_global,
                P_global=self.P_global,
                X_global=self.X_global,
                coords_global=self.coords_global,
                X_cond_global=self.X_cond_global,
                A_cond_global=self.A_cond_global,
                bundle_center_idx=bundle_center_idx,
                k_max=self.k_max,
                m_max=self.m_max,
                data_cache=self.data_cache
            )
            return self._pad_and_format_subgraph(sub_res, center_idx)


def collate_subgraphs(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate list of sample dicts into batched PyTorch tensors."""
    res = {
        "center_idx": torch.tensor([item["center_idx"] for item in batch], dtype=torch.long),
        "is_isolated": torch.tensor([item["is_isolated"] for item in batch], dtype=torch.bool),
        "X_center_true": torch.stack([item["X_center_true"] for item in batch], dim=0),
        "X_vis": torch.stack([item["X_vis"] for item in batch], dim=0),
        "delta_s_vis": torch.stack([item["delta_s_vis"] for item in batch], dim=0),
        "L_vis_cond": torch.stack([item["L_vis_cond"] for item in batch], dim=0),
        "H_vis_cond": torch.stack([item["H_vis_cond"] for item in batch], dim=0),
        "vis_pad_mask": torch.stack([item["vis_pad_mask"] for item in batch], dim=0),
        "H_0_target": torch.stack([item["H_0_target"] for item in batch], dim=0),
        "delta_s_target": torch.stack([item["delta_s_target"] for item in batch], dim=0),
        "target_mask": torch.stack([item["target_mask"] for item in batch], dim=0)
    }
    if "X_vis_2" in batch[0]:
        res["X_vis_2"] = torch.stack([item["X_vis_2"] for item in batch], dim=0)
        res["vis_pad_mask_2"] = torch.stack([item["vis_pad_mask_2"] for item in batch], dim=0)

    return res
