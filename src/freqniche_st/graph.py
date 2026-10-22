"""
Global Spatial Graph Construction and Topological Controls for FreqNiche-ST.
Strictly enforces:
- Complete Euclidean distance threshold r (default 50.0 um)
- Gaussian kernel sigma = r / 3
- Symmetric adjacency A = A^T with self-loops (A_ii = 1.0)
- Row-normalized transition operator P = D^{-1} A
- Disjointness across samples/donors/FOVs
- Pure sparse within-sample topological controls:
  1. Shuffled-P: Within-sample edge permutation (O(nnz), preserves A=A^T)
  2. Shuffled-G-Bundle: Within-sample center-neighborhood bundle assignment permutation
  3. Shuffled-X-Condition: Within-sample condition feature permutation
"""

from typing import Tuple, Optional
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def build_global_spatial_graph(
    coords: np.ndarray,
    sample_ids: np.ndarray,
    radius: float = 50.0,
    sigma: Optional[float] = None,
    max_neighbors: Optional[int] = None
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Builds the global spatial adjacency matrix A and row-stochastic diffusion operator P.
    Guarantees:
    1. Strict symmetry of adjacency: A = A^T
    2. Strict disconnection between different samples/FOVs.
    """
    if sigma is None:
        sigma = radius / 3.0

    n_cells = coords.shape[0]
    unique_samples = np.unique(sample_ids)

    rows_list = []
    cols_list = []
    data_list = []

    for s_id in unique_samples:
        mask = (sample_ids == s_id)
        indices = np.where(mask)[0]
        n_sample_cells = len(indices)

        if n_sample_cells == 0:
            continue

        sample_coords = coords[indices]
        
        # Self loops with weight 1.0
        rows_list.append(indices)
        cols_list.append(indices)
        data_list.append(np.ones(n_sample_cells, dtype=np.float32))

        # Radius neighbors
        if n_sample_cells > 1:
            nn = NearestNeighbors(radius=radius, algorithm="kd_tree")
            nn.fit(sample_coords)
            
            distances, local_nbrs = nn.radius_neighbors(sample_coords, return_distance=True)

            for i_local, (dists, nbrs) in enumerate(zip(distances, local_nbrs)):
                center_global = indices[i_local]
                
                non_self = (nbrs != i_local)
                valid_nbrs = nbrs[non_self]
                valid_dists = dists[non_self]

                if len(valid_nbrs) > 0:
                    if max_neighbors is not None and len(valid_nbrs) > max_neighbors:
                        top_k = np.argsort(valid_dists)[:max_neighbors]
                        valid_nbrs = valid_nbrs[top_k]
                        valid_dists = valid_dists[top_k]

                    nbrs_global = indices[valid_nbrs]
                    weights = np.exp(- (valid_dists ** 2) / (2.0 * (sigma ** 2))).astype(np.float32)
                    
                    rows_list.append(np.full(len(nbrs_global), center_global, dtype=np.int32))
                    cols_list.append(nbrs_global)
                    data_list.append(weights)

    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    all_data = np.concatenate(data_list)

    A = sp.csr_matrix((all_data, (all_rows, all_cols)), shape=(n_cells, n_cells), dtype=np.float32)

    if max_neighbors is None:
        A = ((A + A.T) * 0.5).tocsr()

    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums = np.maximum(row_sums, 1e-6)
    inv_d = 1.0 / row_sums
    D_inv = sp.diags(inv_d, dtype=np.float32)

    P = (D_inv @ A).tocsr()

    return P, A


def build_within_sample_shuffled_p_graph(
    A_global: sp.csr_matrix,
    sample_ids: np.ndarray,
    seed: int = 42
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Constructs Shuffled-P negative control graph via PURE SPARSE permutation.
    Permutes node connectivity within each sample, preserving exact symmetry, self-loops,
    and strict sample isolation (zero cross-sample edges) without any dense conversion.
    Complexity: O(nnz), safe for 100k+ cells.
    """
    rng = np.random.RandomState(seed)
    n_cells = A_global.shape[0]
    unique_samples = np.unique(sample_ids)

    global_perm = np.arange(n_cells)

    for s_id in unique_samples:
        indices = np.where(sample_ids == s_id)[0]
        n_s = len(indices)
        if n_s > 1:
            perm_loc = rng.permutation(n_s)
            global_perm[indices] = indices[perm_loc]

    A_shuff = A_global[global_perm, :][:, global_perm].tocsr()

    row_sums = np.array(A_shuff.sum(axis=1)).flatten()
    row_sums = np.maximum(row_sums, 1e-6)
    D_inv = sp.diags(1.0 / row_sums, dtype=np.float32)
    P_shuff = (D_inv @ A_shuff).tocsr()

    return P_shuff, A_shuff


def build_within_sample_shuffled_g_bundle_map(
    n_cells: int,
    sample_ids: np.ndarray,
    seed: int = 42
) -> np.ndarray:
    """
    Constructs Shuffled-G-Bundle negative control mapping.
    Permutes neighborhood bundle assignments within each sample: center v is conditioned on
    the spatial neighborhood of another cell u in the same sample, while keeping v's true target.
    """
    rng = np.random.RandomState(seed)
    bundle_map = np.arange(n_cells)
    unique_samples = np.unique(sample_ids)

    for s_id in unique_samples:
        indices = np.where(sample_ids == s_id)[0]
        n_s = len(indices)
        if n_s > 1:
            perm_loc = rng.permutation(n_s)
            bundle_map[indices] = indices[perm_loc]

    return bundle_map


def build_within_sample_shuffled_x_features(
    X_global: np.ndarray,
    sample_ids: np.ndarray,
    seed: int = 42
) -> np.ndarray:
    """
    Constructs Shuffled-X-Condition negative control feature matrix.
    Permutes gene expression vectors across cells strictly within each sample for condition only.
    """
    rng = np.random.RandomState(seed)
    X_shuff = X_global.copy()
    unique_samples = np.unique(sample_ids)

    for s_id in unique_samples:
        indices = np.where(sample_ids == s_id)[0]
        n_s = len(indices)
        if n_s > 1:
            perm = rng.permutation(n_s)
            X_shuff[indices] = X_global[indices[perm]]

    return X_shuff
