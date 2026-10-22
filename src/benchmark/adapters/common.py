from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[3]


def dense_float32(matrix: object) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def prepare_hvg_adata(
    source: ad.AnnData,
    *,
    n_top_genes: int = 3000,
    scale: bool = False,
) -> ad.AnnData:
    """Paper-aligned raw-count HVG selection followed by normalize/log1p."""
    data = source.copy()
    if "highly_variable" in data.var:
        del data.var["highly_variable"]
    sc.pp.highly_variable_genes(
        data, flavor="seurat_v3", n_top_genes=n_top_genes, subset=False
    )
    selected = data[:, data.var["highly_variable"].to_numpy()].copy()
    sc.pp.normalize_total(selected, target_sum=1e4)
    sc.pp.log1p(selected)
    if scale:
        sc.pp.scale(selected, zero_center=False, max_value=10)
    return selected


def prepare_banksy_dlpfc_adata(
    source: ad.AnnData, *, n_top_genes: int = 2000
) -> ad.AnnData:
    """BANKSY DLPFC preprocessing from the Nature Genetics protocol."""
    data = source.copy()
    totals = np.asarray(data.X.sum(axis=1)).reshape(-1)
    positive = totals[totals > 0]
    if positive.size == 0:
        raise ValueError("BANKSY input has no positive library sizes")
    sc.pp.normalize_total(data, target_sum=float(np.median(positive)))
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(
        data, flavor="seurat", n_top_genes=n_top_genes, subset=False
    )
    return data[:, data.var["highly_variable"].to_numpy()].copy()


def controlled_gmm_labels(
    embedding: np.ndarray, n_clusters: int, seed: int
) -> np.ndarray:
    values = np.asarray(embedding, dtype=np.float64)
    return GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=seed,
    ).fit_predict(values)


def canonical_mclust_R(
    embedding: np.ndarray,
    n_clusters: int,
    seed: int = 42,
    model_names: str = "EEE",
) -> np.ndarray:
    """Executes true R mclust package via Rscript subprocess with complete isolation."""
    import tempfile
    import subprocess
    import pandas as pd

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f_in, \
         tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f_out:
        in_path = f_in.name
        out_path = f_out.name

    df = pd.DataFrame(np.asarray(embedding, dtype=np.float64))
    df.to_csv(in_path, index=False, header=False)
    r_code = f"""
    suppressMessages(library(mclust))
    mat <- as.matrix(read.csv('{in_path}', header=FALSE))
    set.seed({seed})
    res <- Mclust(mat, G={n_clusters}, modelNames='{model_names}', verbose=FALSE)
    write.csv(res$classification, '{out_path}', row.names=FALSE)
    """
    proc = subprocess.run(["Rscript", "-e", r_code], check=False, capture_output=True, text=True)
    if proc.returncode == 0 and Path(out_path).is_file():
        preds = pd.read_csv(out_path).iloc[:, 0].to_numpy() - 1
        Path(in_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)
        return preds.astype(np.int64)
    else:
        Path(in_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)
        raise RuntimeError(f"R mclust execution failed:\n{proc.stderr}\n{proc.stdout}")


def git_commit(repository: Path) -> str:
    # Do not accidentally report the parent SpaPEFT commit for vendored sources
    # whose original .git metadata was not retained.
    if not (repository / ".git").exists():
        return "vendored_source_without_embedded_git_metadata"
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def source_tree_sha256(repository: Path) -> str:
    """Stable fingerprint for vendored baseline source and protocol files."""
    digest = hashlib.sha256()
    candidates = sorted(
        path for path in repository.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix.lower() in {".py", ".md", ".rst", ".txt", ".yml", ".yaml"}
    )
    if not candidates:
        raise FileNotFoundError(f"No source files found under {repository}")
    for path in candidates:
        digest.update(path.relative_to(repository).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def require_spatial(adata: ad.AnnData) -> None:
    if "spatial" not in adata.obsm:
        raise KeyError("Canonical adapter requires adata.obsm['spatial']")
    spatial = np.asarray(adata.obsm["spatial"])
    if spatial.shape != (adata.n_obs, 2) or not np.isfinite(spatial).all():
        raise ValueError(f"Invalid spatial coordinates: {spatial.shape}")


def full_resolution_visium_coordinates(adata: ad.AnnData) -> tuple[np.ndarray, str]:
    """Recover approximate full-resolution Visium pixels for radius methods."""
    require_spatial(adata)
    coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    nearest = NearestNeighbors(n_neighbors=2).fit(coordinates)
    median_pitch = float(np.median(nearest.kneighbors(coordinates)[0][:, 1]))
    if median_pitch >= 50:
        return coordinates, "obsm_spatial_already_full_resolution"

    spatial_metadata = adata.uns.get("spatial", {})
    if isinstance(spatial_metadata, dict):
        for library in spatial_metadata.values():
            if not isinstance(library, dict):
                continue
            scale = library.get("scalefactors", {}).get("tissue_hires_scalef")
            if scale is not None and float(scale) > 0:
                recovered = coordinates / float(scale)
                return recovered.astype(np.float32), "obsm_spatial_divided_by_tissue_hires_scalef"

    # Visium center-to-center pitch is approximately 100 full-resolution pixels.
    recovered = coordinates * (100.0 / max(median_pitch, 1e-6))
    return recovered.astype(np.float32), "adapter_pitch_calibration_to_100_pixels"


def refine_knn_majority(
    labels: np.ndarray, coordinates: np.ndarray, *, n_neighbors: int = 50
) -> np.ndarray:
    labels = np.asarray(labels)
    count = min(n_neighbors + 1, len(labels))
    indices = NearestNeighbors(n_neighbors=count).fit(coordinates).kneighbors(
        return_distance=False
    )
    refined = []
    for row, center in zip(indices, np.arange(len(labels)), strict=True):
        neighbors = row[row != center][:n_neighbors]
        values, counts = np.unique(labels[neighbors], return_counts=True)
        refined.append(values[np.argmax(counts)])
    return np.asarray(refined)


def build_sample_isolated_spatial_graph(
    adata: ad.AnnData,
    *,
    sample_key: str = "sample_id",
    radius: float | None = 50.0,
    n_neighbors: int | None = None,
    include_self: bool = True,
    allow_single_sample: bool = False,
) -> sp.csr_matrix:
    """Builds a provably block-diagonal spatial graph strictly isolated by physical sample/slice.

    Scientific Contract:
    - Never connects cells across different physical slices or FOVs (sample_key).
    - If sample_key is missing from adata.obs and allow_single_sample=False, raises KeyError (Fail-Closed).
    - Assembles a block-diagonal scipy.sparse.csr_matrix matching original adata index ordering via vectorized COO arrays.
    """
    require_spatial(adata)
    n_cells = adata.n_obs

    # Strict Fail-Closed check on sample_key
    if sample_key not in adata.obs.columns:
        if not allow_single_sample:
            raise KeyError(
                f"Required sample isolation key '{sample_key}' is missing in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}. "
                f"Pass allow_single_sample=True only if the data is confirmed to be a single physical slice."
            )
        groups = np.zeros(n_cells, dtype=int)
    else:
        groups = adata.obs[sample_key].to_numpy()

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    unique_groups = pd.unique(groups)

    orig_indices = np.arange(n_cells)
    all_rows = []
    all_cols = []
    all_data = []

    # Process each physical slice independently
    for grp in unique_groups:
        mask = (groups == grp)
        sub_coords = coords[mask]
        sub_n = sub_coords.shape[0]

        if sub_n == 0:
            continue

        if radius is not None:
            nn = NearestNeighbors(radius=radius, metric="euclidean").fit(sub_coords)
            sub_adj = nn.radius_neighbors_graph(sub_coords, mode="connectivity")
        elif n_neighbors is not None:
            k = min(n_neighbors + (1 if include_self else 0), sub_n)
            nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(sub_coords)
            sub_adj = nn.kneighbors_graph(sub_coords, mode="connectivity")
        else:
            raise ValueError("Either radius or n_neighbors must be specified.")

        sub_adj = sub_adj.tocsr()
        if include_self:
            sub_adj.setdiag(1)
        else:
            sub_adj.setdiag(0)
        sub_adj.eliminate_zeros()

        sub_coo = sub_adj.tocoo()
        idx_sub = orig_indices[mask]
        all_rows.append(idx_sub[sub_coo.row])
        all_cols.append(idx_sub[sub_coo.col])
        all_data.append(sub_coo.data)

    # Vectorized single-shot assembly via COO array concatenation
    if all_rows:
        concat_rows = np.concatenate(all_rows)
        concat_cols = np.concatenate(all_cols)
        concat_data = np.concatenate(all_data)
    else:
        concat_rows = np.array([], dtype=np.int64)
        concat_cols = np.array([], dtype=np.int64)
        concat_data = np.array([], dtype=np.float32)

    global_csr = sp.csr_matrix(
        (concat_data, (concat_rows, concat_cols)),
        shape=(n_cells, n_cells),
        dtype=np.float32,
    )
    global_csr.sum_duplicates()
    global_csr.eliminate_zeros()

    # Zero cross-sample leakage assertion
    if len(unique_groups) > 1:
        rows, cols = global_csr.nonzero()
        cross_edges = (groups[rows] != groups[cols]).sum()
        if cross_edges > 0:
            raise RuntimeError(f"FATAL: Detected {cross_edges} illegal cross-sample spatial connections!")

    adata.obsp["spatial_connectivities"] = global_csr
    return global_csr
