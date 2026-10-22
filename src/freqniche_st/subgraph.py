"""
Subgraph Extraction and Target-Excluded Masking for FreqNiche-ST.
Strictly implements:
1. Training:
   - Cell-centric radius subgraph (K_max = 64)
   - True Target Extraction: ALWAYS sampled from true center v's physical neighborhood V_sub(v)
   - Condition Bundle Extraction:
     - Full-DiT: V_vis = V_sub(v) \\ M_target
     - Shuffled-G-Bundle: V_vis = (V_sub(u) \\ M_target)[:n_vis_required]
       (Strict target exclusion: zero target leakage; exact capacity matching with Full-DiT)
   - True residual target: H_M^{target} = X_{M, :}^{true} - P_{M, :}^{true} X^{true} (100% bitwise identical across Full and Shuffled-G)
   - Target-excluded condition operator: P_cond = D_{vis}^{-1} A_{V_vis, V_vis}
2. Deterministic Inference:
   - Full subgraph (M = empty, target_mask all False, no masked nodes)
   - All K <= K_max neighbors visible, sorted deterministically by distance
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import scipy.sparse as sp


def _get_sorted_subgraph_nodes(
    center_idx: int,
    A_global: sp.csr_matrix,
    k_max: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to extract deterministically distance-sorted neighborhood nodes for a center."""
    row_start = A_global.indptr[center_idx]
    row_end = A_global.indptr[center_idx + 1]

    raw_neighbors = A_global.indices[row_start:row_end]
    raw_weights = A_global.data[row_start:row_end]

    if len(raw_neighbors) <= 1:
        return np.array([center_idx], dtype=np.int64), np.array([], dtype=np.int64)

    center_mask = (raw_neighbors == center_idx)
    other_nbrs = raw_neighbors[~center_mask]
    other_weights = raw_weights[~center_mask]

    sort_keys = np.lexsort((other_nbrs, -other_weights))
    sorted_other = other_nbrs[sort_keys]

    if len(sorted_other) > (k_max - 1):
        sorted_other = sorted_other[:(k_max - 1)]

    V_sub = np.concatenate([[center_idx], sorted_other])
    return V_sub, sorted_other


def extract_subgraph_inference(
    center_idx: int,
    A_global: sp.csr_matrix,
    P_global: sp.csr_matrix,
    X_global: np.ndarray,
    coords_global: np.ndarray,
    X_cond_global: Optional[np.ndarray] = None,
    A_cond_global: Optional[sp.csr_matrix] = None,
    bundle_center_idx: Optional[int] = None,
    k_max: int = 64,
    m_max: int = 32,
    data_cache=None
) -> Dict[str, Any]:
    """
    Deterministic inference extractor: NO MASKING (M = empty set).
    Extracts the complete cell-centric subgraph G_v with deterministic distance-based ordering.
    Guarantees:
    - V_vis is ALWAYS extracted from true physical graph A_global.
    - If A_cond_global is provided (e.g. Shuffled-P), only the condition operator P_cond uses it.
    """
    if X_cond_global is None:
        X_cond_global = X_global
    
    query_idx = bundle_center_idx if bundle_center_idx is not None else center_idx

    if data_cache is not None:
        data_cache.validate(A_global, X_global, P_global)
        V_vis, _ = data_cache.nodes(query_idx, k_max)
    else:
        V_vis, _ = _get_sorted_subgraph_nodes(query_idx, A_global, k_max=k_max)
    is_isolated = (len(V_vis) <= 1)

    center_coord = coords_global[query_idx:query_idx+1]
    delta_s_vis = (coords_global[V_vis] - center_coord).astype(np.float32)
    X_vis = X_cond_global[V_vis].astype(np.float32)

    A_filter = A_cond_global if A_cond_global is not None else A_global
    if data_cache is not None and A_cond_global is None:
        A_local = data_cache.local_adjacency(query_idx, V_vis, k_max)
    else:
        A_local = A_filter[V_vis, :][:, V_vis].toarray().astype(np.float32)
    row_sums = A_local.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-6)
    P_cond = A_local / row_sums

    L_vis_cond = (P_cond @ X_vis).astype(np.float32)
    H_vis_cond = (X_vis - L_vis_cond).astype(np.float32)

    return {
        "center_idx": center_idx,
        "is_isolated": is_isolated,
        "X_center_true": X_global[center_idx].astype(np.float32),
        "V_sub": V_vis,
        "V_vis": V_vis,
        "M_target": np.array([], dtype=np.int64),
        "X_vis": X_vis,
        "delta_s_vis": delta_s_vis,
        "L_vis_cond": L_vis_cond,
        "H_vis_cond": H_vis_cond,
        "H_target": np.zeros((0, X_global.shape[1]), dtype=np.float32),
        "target_mask": np.zeros(m_max, dtype=bool)
    }


def sample_subgraph_and_mask(
    center_idx: int,
    A_global: sp.csr_matrix,
    P_global: sp.csr_matrix,
    X_global: np.ndarray,
    coords_global: np.ndarray,
    X_cond_global: Optional[np.ndarray] = None,
    A_cond_global: Optional[sp.csr_matrix] = None,
    bundle_center_idx: Optional[int] = None,
    k_max: int = 64,
    rho: float = 0.5,
    p_center_mask: float = 0.30,
    m_max: int = 32,
    rng: Optional[np.random.RandomState] = None,
    data_cache=None
) -> Dict[str, Any]:
    """
    Extracts cell-centric radius subgraph and generates Target-Excluded condition features for training.
    Guarantees:
    - Target M_target and H_target are ALWAYS extracted from the TRUE center v's physical neighborhood V_sub(v) on A_global.
    - If A_cond_global is provided (e.g. Shuffled-P), neighborhood members V_sub and M_target remain strictly from A_global,
      while only the local filtering operator P_cond uses A_cond_global.
    - If bundle_center_idx is specified (Shuffled-G-Bundle), condition V_vis comes from cell u's neighborhood V_sub(u),
      strictly filtered against M_target (zero leakage) and capacity-matched to len(V_vis_true).
    - H_target is 100% bitwise identical across Full-DiT and Shuffled controls for the same center_idx and rng state.
    """
    if rng is None:
        rng = np.random.RandomState()
    if X_cond_global is None:
        X_cond_global = X_global

    # 1. ALWAYS sample true targets from true center v's physical neighborhood on A_global
    if data_cache is not None:
        data_cache.validate(A_global, X_global, P_global)
        V_sub_true, sorted_other_true = data_cache.nodes(center_idx, k_max)
    else:
        V_sub_true, sorted_other_true = _get_sorted_subgraph_nodes(center_idx, A_global, k_max=k_max)
    k_nodes_true = len(V_sub_true)

    if k_nodes_true <= 1:
        return {
            "center_idx": center_idx,
            "is_isolated": True,
            "X_center_true": X_global[center_idx].astype(np.float32),
            "V_sub": V_sub_true,
            "V_vis": V_sub_true,
            "M_target": np.array([], dtype=np.int64),
            "X_vis": X_cond_global[center_idx:center_idx+1].astype(np.float32),
            "delta_s_vis": np.zeros((1, 2), dtype=np.float32),
            "L_vis_cond": X_cond_global[center_idx:center_idx+1].astype(np.float32),
            "H_vis_cond": np.zeros((1, X_cond_global.shape[1]), dtype=np.float32),
            "H_target": np.zeros((0, X_global.shape[1]), dtype=np.float32),
            "target_mask": np.zeros(m_max, dtype=bool)
        }

    # Mask size calculation for true center
    m = int(min(max(1, np.floor(rho * k_nodes_true)), min(k_nodes_true - 1, m_max)))

    # Two-stage center masking on true center
    mask_center = (rng.uniform(0.0, 1.0) < p_center_mask)
    target_nodes = []

    if mask_center:
        target_nodes.append(center_idx)
        remaining_budget = m - 1
        other_candidates = list(sorted_other_true)
        if remaining_budget > 0 and len(other_candidates) > 0:
            chosen = rng.choice(other_candidates, size=min(remaining_budget, len(other_candidates)), replace=False)
            target_nodes.extend(chosen)
    else:
        other_candidates = list(sorted_other_true)
        chosen = rng.choice(other_candidates, size=min(m, len(other_candidates)), replace=False)
        target_nodes.extend(chosen)

    M_target = np.array(target_nodes, dtype=np.int64)

    # 2. Full-DiT Visible Set & Capacity
    vis_mask = np.isin(V_sub_true, M_target, invert=True)
    V_vis_true = V_sub_true[vis_mask]
    if len(V_vis_true) == 0:
        V_vis_true = np.array([center_idx], dtype=np.int64)
    n_vis_required = len(V_vis_true)

    # 3. Extract Visible Condition Bundle (Zero Leakage + Capacity Matching)
    if bundle_center_idx is not None and bundle_center_idx != center_idx:
        # Shuffled-G-Bundle:
        # a) Retrieve sorted neighborhood of cell u
        V_sub_bundle, _ = (data_cache.nodes(bundle_center_idx, k_max)
                           if data_cache is not None else
                           _get_sorted_subgraph_nodes(bundle_center_idx, A_global, k_max=k_max))
        # b) Strictly exclude any M_target nodes to prevent algebraic leakage
        bundle_candidates = V_sub_bundle[~np.isin(V_sub_bundle, M_target)]
        if len(bundle_candidates) == 0:
            bundle_candidates = np.array([bundle_center_idx], dtype=np.int64)
        # c) Match capacity: take exactly n_vis_required nodes
        V_vis = bundle_candidates[:n_vis_required]
        cond_center_coord = coords_global[bundle_center_idx:bundle_center_idx+1]
    else:
        V_vis = V_vis_true
        cond_center_coord = coords_global[center_idx:center_idx+1]

    # 4. Relative positions and condition features
    delta_s_vis = (coords_global[V_vis] - cond_center_coord).astype(np.float32)
    X_vis = X_cond_global[V_vis].astype(np.float32)

    # Target-Excluded Condition Operator: P_cond = D_vis^{-1} A_{V_vis, V_vis}
    query_idx = (bundle_center_idx if bundle_center_idx is not None
                 and bundle_center_idx != center_idx else center_idx)
    A_filter = A_cond_global if A_cond_global is not None else A_global
    A_vis = (data_cache.local_adjacency(query_idx, V_vis, k_max)
             if (data_cache is not None and A_cond_global is None) else
             A_filter[V_vis, :][:, V_vis].toarray().astype(np.float32))
    row_sums = A_vis.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-6)
    P_cond = A_vis / row_sums

    L_vis_cond = (P_cond @ X_vis).astype(np.float32)
    H_vis_cond = (X_vis - L_vis_cond).astype(np.float32)

    # 5. True supervision target: ALWAYS computed on TRUE X_global and TRUE P_global for TRUE target nodes
    if data_cache is not None:
        H_target = data_cache.residuals.rows(M_target)
    else:
        X_M_true = X_global[M_target].astype(np.float32)
        P_M_true = P_global[M_target, :]
        PX_M_true = (P_M_true @ X_global).astype(np.float32)
        H_target = X_M_true - PX_M_true

    target_mask = np.zeros(m_max, dtype=bool)
    n_targets = min(len(M_target), m_max)
    target_mask[:n_targets] = True

    return {
        "center_idx": center_idx,
        "is_isolated": False,
        "X_center_true": X_global[center_idx].astype(np.float32),
        "V_sub": V_sub_true,
        "V_vis": V_vis,
        "M_target": M_target,
        "X_vis": X_vis,
        "delta_s_vis": delta_s_vis,
        "L_vis_cond": L_vis_cond,
        "H_vis_cond": H_vis_cond,
        "H_target": H_target,
        "target_mask": target_mask
    }
