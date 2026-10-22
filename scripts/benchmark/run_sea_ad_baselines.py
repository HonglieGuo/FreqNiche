"""
Stage 5 Unified Baseline Runner for SEA-AD MTG MERFISH on Remote Server.

Supported Methods (in strict priority order):
1. raw_pca: Standard 50-d PCA on normalized expression (Non-spatial lower bound).
2. px_resdiff: 50-d PCA on spatially smoothed PX and high-pass residual H (Spatial diffusion lower bound).
3. banksy: Official BANKSY-controlled spatial feature augmentation (lambda=0.2, pca_dim=20).
4. cellcharter: Official CellCharter-PCA-Controlled neighborhood aggregation.
5. cellniche: Official CellNiche microenvironment contrastive representation.
6. stagate: Spatial Graph Attention Autoencoder (128-d).

Unified Evaluation Protocol:
- Same Train / Validation / Test split from sea_ad_mtg_merfish.h5ad;
- Same 10,000 Validation & 10,000 Test evaluation centers;
- Unsupervised stability resolution r* selected strictly on Validation 10k centers (K in [5, 25]);
- Single-pass Test blind evaluation at locked r*;
- Outputs standardized cell assignments CSV (7 columns) and benchmark summary CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.freqniche_st.graph import build_global_spatial_graph
from src.freqniche_st.runtime_paths import resolve_run_output, resolve_baseline_output

DATASET_PATH = WORKSPACE_ROOT / "datasets" / "h5ad" / "sea_ad" / "sea_ad_mtg_merfish.h5ad"
BASE_DIR = resolve_run_output("outputs/runs/freqniche/sea_ad/2026-09-15/20260915_sea_ad_final")
BASELINES_DIR = resolve_baseline_output("baselines/Baselines_outputs/runs/sea_ad/2026-09-15/20260915_sea_ad_baselines")
EMB_DIR = BASELINES_DIR / "embeddings"
ASSIGN_DIR = BASELINES_DIR / "assignments"
RES_DIR = BASELINES_DIR / "results"

VAL_CENTERS_PATH = BASE_DIR / "val_centers.json"
VAL_FOLDS_PATH = BASE_DIR / "val_bootstrap_folds.json"
TEST_CENTERS_PATH = BASE_DIR / "test_centers.json"

FROZEN_RESOLUTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
K_MIN = 5
K_MAX = 25
SEEDS = [42, 123, 2024]


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def select_unsupervised_resolution(
    val_emb: np.ndarray,
    bootstrap_folds: Dict[str, Any],
    resolutions: List[float] = FROZEN_RESOLUTIONS,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    seed: int = 42
) -> Tuple[float, Dict[str, Any]]:
    """Selects optimal Leiden resolution r* for a single seed via unsupervised stability on 10 folds."""
    fold_names = [f"fold_{i}" for i in range(10)]
    fold_indices_list = [np.array(bootstrap_folds[fname]["indices_in_val_centers"], dtype=np.int64) for fname in fold_names]
    n_bootstraps = len(fold_indices_list)

    scores_per_res: Dict[float, float] = {}
    mean_k_per_res: Dict[float, float] = {}

    for res in resolutions:
        clusterings: List[np.ndarray] = []
        for b in range(n_bootstraps):
            sub_idx = fold_indices_list[b]
            z_sub = val_emb[sub_idx]

            adata_sub = ad.AnnData(X=z_sub.copy())
            sc.pp.neighbors(adata_sub, n_neighbors=15, use_rep="X")
            sc.tl.leiden(adata_sub, resolution=res, key_added="leiden", random_state=seed + b)
            clusterings.append(adata_sub.obs["leiden"].to_numpy().astype(int))

        n_clusters_list = [len(np.unique(c)) for c in clusterings]
        mean_k = float(np.mean(n_clusters_list))
        mean_k_per_res[res] = mean_k

        pair_aris: List[float] = []
        for i in range(n_bootstraps):
            for j in range(i + 1, n_bootstraps):
                idx_i = fold_indices_list[i]
                idx_j = fold_indices_list[j]
                common_cells, pos_i, pos_j = np.intersect1d(idx_i, idx_j, return_indices=True)
                if len(common_cells) > 10:
                    c_i = clusterings[i][pos_i]
                    c_j = clusterings[j][pos_j]
                    pair_aris.append(float(adjusted_rand_score(c_i, c_j)))

        scores_per_res[res] = float(np.mean(pair_aris)) if pair_aris else 0.0

    valid_res = [r for r in resolutions if k_min <= mean_k_per_res[r] <= k_max]
    if valid_res:
        best_res = max(valid_res, key=lambda r: (scores_per_res[r], -r))
    else:
        def dist_to_interval(r: float) -> float:
            k = mean_k_per_res[r]
            if k < k_min:
                return k_min - k
            if k > k_max:
                return k - k_max
            return 0.0
        best_res = min(resolutions, key=lambda r: (dist_to_interval(r), -scores_per_res[r]))

    report = {
        "optimal_r_star": float(best_res),
        "stability_scores": {str(k): float(v) for k, v in scores_per_res.items()},
        "mean_clusters": {str(k): float(v) for k, v in mean_k_per_res.items()},
        "best_stability_score": float(scores_per_res[best_res]),
        "valid_resolutions": valid_res
    }
    return float(best_res), report


def evaluate_test_clustering(
    test_emb: np.ndarray,
    frozen_r_star: float,
    test_cohort: Dict[str, Any],
    adata: ad.AnnData,
    method_name: str,
    seed: int
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluates Test embedding at frozen r*, exports assignments and donor metrics."""
    test_global_indices = test_cohort["global_indices"]
    test_barcodes = test_cohort["cell_barcodes"]
    test_sample_ids = adata.obs["sample_id"].iloc[test_global_indices].to_numpy()
    test_donors = adata.obs["donor_id"].iloc[test_global_indices].to_numpy()
    test_true_labels = adata.obs["niche"].iloc[test_global_indices].to_numpy()

    adata_t = ad.AnnData(X=test_emb.copy())
    sc.pp.neighbors(adata_t, n_neighbors=15, use_rep="X", random_state=seed)
    sc.tl.leiden(adata_t, resolution=frozen_r_star, key_added="pred", random_state=seed)
    pred_test = adata_t.obs["pred"].to_numpy().astype(int)

    # 1. Assignments DataFrame
    df_assignments = pd.DataFrame({
        "cell_barcode": test_barcodes,
        "donor_id": test_donors,
        "sample_id": test_sample_ids,
        "niche": test_true_labels,
        "predicted_cluster": pred_test,
        "method": method_name,
        "seed": seed,
        "r_star": frozen_r_star
    })

    # 2. Donor-level metrics
    unique_donors = np.unique(test_donors)
    donor_aris = []
    donor_nmis = []
    donor_ari_dict = {}
    donor_nmi_dict = {}

    for d in unique_donors:
        mask = (test_donors == d)
        y_true = test_true_labels[mask]
        y_pred = pred_test[mask]
        ari = float(adjusted_rand_score(y_true, y_pred))
        nmi = float(normalized_mutual_info_score(y_true, y_pred))
        donor_aris.append(ari)
        donor_nmis.append(nmi)
        donor_ari_dict[str(d)] = ari
        donor_nmi_dict[str(d)] = nmi

    macro_ari = float(np.mean(donor_aris))
    macro_nmi = float(np.mean(donor_nmis))
    worst_ari = float(np.min(donor_aris))
    asw = float(silhouette_score(test_emb, test_true_labels, metric="euclidean"))

    # 3. 1,000-Iteration Donor Bootstrap
    rng = np.random.RandomState(seed)
    boot_aris = []
    boot_nmis = []
    boot_worsts = []
    for _ in range(1000):
        b_donors = rng.choice(unique_donors, size=len(unique_donors), replace=True)
        b_aris = [donor_ari_dict[str(d)] for d in b_donors]
        b_nmis = [donor_nmi_dict[str(d)] for d in b_donors]
        boot_aris.append(float(np.mean(b_aris)))
        boot_nmis.append(float(np.mean(b_nmis)))
        boot_worsts.append(float(np.min(b_aris)))

    ci_macro_ari = (float(np.percentile(boot_aris, 2.5)), float(np.percentile(boot_aris, 97.5)))
    ci_macro_nmi = (float(np.percentile(boot_nmis, 2.5)), float(np.percentile(boot_nmis, 97.5)))
    ci_worst_ari = (float(np.percentile(boot_worsts, 2.5)), float(np.percentile(boot_worsts, 97.5)))

    metrics = {
        "method": method_name,
        "seed": seed,
        "r_star": frozen_r_star,
        "n_clusters": int(len(np.unique(pred_test))),
        "macro_ari": macro_ari,
        "macro_nmi": macro_nmi,
        "worst_donor_ari": worst_ari,
        "asw": asw,
        "donor_aris": donor_ari_dict,
        "donor_nmis": donor_nmi_dict,
        "ci_95_macro_ari": ci_macro_ari,
        "ci_95_macro_nmi": ci_macro_nmi,
        "ci_95_worst_ari": ci_worst_ari
    }
    return metrics, df_assignments


# ==============================================================================
# BASELINE FEATURE EXTRACTORS
# ==============================================================================

def run_raw_pca(
    adata: ad.AnnData,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    n_components: int = 50,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Method 1: Raw PCA on standardized expression (normalize_total(1e4) + log1p)."""
    print(f"[{time.strftime('%H:%M:%S')}] Running Raw PCA (n_components={n_components}, seed={seed}) ...")
    ad_pca = adata.copy()
    sc.pp.normalize_total(ad_pca, target_sum=1e4)
    sc.pp.log1p(ad_pca)

    X = ad_pca.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    pca = PCA(n_components=n_components, random_state=seed)
    # Fit PCA on Train + Val context to avoid any test leakage
    train_val_mask = (ad_pca.obs["split"] != "test").to_numpy()
    pca.fit(X[train_val_mask])

    val_emb = pca.transform(X[val_indices]).astype(np.float32)
    test_emb = pca.transform(X[test_indices]).astype(np.float32)
    return val_emb, test_emb


_CACHED_MULTI_FREQ = None

def run_px_resdiff(
    adata: ad.AnnData,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    n_components: int = 50,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Method 2: PX / ResDiff baseline (Spatial convolution PX + high-pass residual H)."""
    global _CACHED_MULTI_FREQ
    print(f"[{time.strftime('%H:%M:%S')}] Running PX / ResDiff (R=50um spatial smoothing, seed={seed}) ...")
    
    if _CACHED_MULTI_FREQ is None:
        print(f"[{time.strftime('%H:%M:%S')}] Computing spatial diffusion PX and high-pass residual H on full cohort ...")
        ad_work = adata.copy()
        sc.pp.normalize_total(ad_work, target_sum=1e4)
        sc.pp.log1p(ad_work)

        coords = ad_work.obsm["spatial"].astype(np.float32)
        sample_ids = ad_work.obs["sample_id"].to_numpy()

        # Build sample-isolated spatial graph
        P, A = build_global_spatial_graph(coords=coords, sample_ids=sample_ids, radius=50.0, max_neighbors=None)

        X = ad_work.X
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        # Compute PX (low-pass smoothing)
        PX = P.dot(X).astype(np.float32)
        # Residual H = X - PX
        H = X - PX

        # Concatenate [PX, H] as complete multi-frequency feature representation
        _CACHED_MULTI_FREQ = np.concatenate([PX, H], axis=1)
        print(f"[{time.strftime('%H:%M:%S')}] Cached multi-frequency features: shape={_CACHED_MULTI_FREQ.shape}")

    multi_freq_features = _CACHED_MULTI_FREQ
    train_val_mask = (adata.obs["split"] != "test").to_numpy()
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(multi_freq_features[train_val_mask])

    val_emb = pca.transform(multi_freq_features[val_indices]).astype(np.float32)
    test_emb = pca.transform(multi_freq_features[test_indices]).astype(np.float32)
    return val_emb, test_emb


def run_banksy(
    adata: ad.AnnData,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Method 3: Official BANKSY-Controlled adapter."""
    print(f"[{time.strftime('%H:%M:%S')}] Running BANKSY adapter (seed={seed}) ...")
    from src.benchmark.adapters.banksy import OfficialBANKSYAdapter

    adapter = OfficialBANKSYAdapter(
        protocol="controlled",
        lambda_param=0.2,
        pca_dim=20,
        return_diagnostics=False
    )
    res = adapter.fit_transform(adata, seed=seed, embedding_only=True)
    full_emb = np.asarray(res.embedding, dtype=np.float32)
    val_emb = full_emb[val_indices]
    test_emb = full_emb[test_indices]
    return val_emb, test_emb


def run_cellcharter(
    adata: ad.AnnData,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Method 4: Official CellCharter-PCA-Controlled adapter."""
    print(f"[{time.strftime('%H:%M:%S')}] Running CellCharter-PCA-Controlled adapter (seed={seed}) ...")
    from src.benchmark.adapters.cellcharter import ControlledCellCharterAdapter

    adapter = ControlledCellCharterAdapter(
        protocol="controlled",
        sample_key="sample_id",
        pca_dim=50,
        n_layers=3,
        radius=50.0
    )
    res = adapter.fit_transform(adata, seed=seed, embedding_only=True)
    full_emb = np.asarray(res.embedding, dtype=np.float32)
    val_emb = full_emb[val_indices]
    test_emb = full_emb[test_indices]
    return val_emb, test_emb


def run_cellniche(
    adata: ad.AnnData,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Method 5: Official CellNiche adapter."""
    print(f"[{time.strftime('%H:%M:%S')}] Running CellNiche adapter (seed={seed}) ...")
    from src.benchmark.adapters.cellniche import OfficialCellNicheAdapter

    adapter = OfficialCellNicheAdapter(
        protocol="controlled",
        sample_key="sample_id",
        radius=50.0,
        epochs=5,
        batch_size=512
    )
    res = adapter.fit_transform(adata, seed=seed, embedding_only=True)
    full_emb = np.asarray(res.embedding, dtype=np.float32)
    val_emb = full_emb[val_indices]
    test_emb = full_emb[test_indices]
    return val_emb, test_emb


def run_stagate(
    adata: ad.AnnData,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Method 7: Official STAGATE adapter."""
    print(f"[{time.strftime('%H:%M:%S')}] Running STAGATE adapter (seed={seed}) ...")
    from src.benchmark.adapters.stagate import OfficialSTAGATEAdapter

    adapter = OfficialSTAGATEAdapter(
        protocol="controlled",
        sample_key="sample_id",
        radius=50.0
    )
    res = adapter.fit_transform(adata, seed=seed, embedding_only=True)
    full_emb = np.asarray(res.embedding, dtype=np.float32)
    val_emb = full_emb[val_indices]
    test_emb = full_emb[test_indices]
    return val_emb, test_emb


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

METHOD_RUNNERS = {
    "raw_pca": run_raw_pca,
    "px_resdiff": run_px_resdiff,
    "banksy": run_banksy,
    "cellcharter": run_cellcharter,
    "cellniche": run_cellniche,
    "stagate": run_stagate
}


def main():
    parser = argparse.ArgumentParser(description="Stage 5 SEA-AD Baseline Benchmark")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["raw_pca", "px_resdiff", "banksy", "cellcharter", "cellniche"],
        help="List of methods to run: raw_pca, px_resdiff, banksy, cellcharter, cellniche, stagate"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024], help="Random seeds")
    args = parser.parse_args()

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    ASSIGN_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 5: SEA-AD BASELINE BENCHMARK EXECUTION")
    print(f"Methods to run: {args.methods}")
    print(f"Seeds: {args.seeds}")
    print(f"Output Directory: {BASELINES_DIR}")
    print("=" * 80)

    # 1. Load Datasets and Cohort Manifests
    print(f"Loading AnnData from {DATASET_PATH} ...")
    adata = ad.read_h5ad(DATASET_PATH)
    print(f"Loaded AnnData: shape={adata.shape}, genes={adata.n_vars}")

    with open(VAL_CENTERS_PATH, "r", encoding="utf-8") as f:
        val_cohort = json.load(f)
    with open(VAL_FOLDS_PATH, "r", encoding="utf-8") as f:
        val_folds = json.load(f)
    with open(TEST_CENTERS_PATH, "r", encoding="utf-8") as f:
        test_cohort = json.load(f)

    val_global_indices = np.array(val_cohort["global_indices"], dtype=np.int64)
    test_global_indices = np.array(test_cohort["global_indices"], dtype=np.int64)
    assert len(val_global_indices) == 10000
    assert len(test_global_indices) == 10000
    print("Cohort centers verified: 10,000 Validation & 10,000 Test centers.")

    all_benchmark_records = []

    for method in args.methods:
        if method not in METHOD_RUNNERS:
            print(f"[SKIP] Unknown method: {method}")
            continue

        runner_fn = METHOD_RUNNERS[method]
        method_seed_metrics = []

        print("\n" + "#" * 80)
        print(f"STARTING BASELINE: {method.upper()}")
        print("#" * 80)

        for seed in args.seeds:
            print(f"\n--- Method: {method.upper()} | Seed: {seed} ---")
            t_start = time.time()

            val_emb_path = EMB_DIR / f"{method}_seed_{seed}_val_emb.npy"
            test_emb_path = EMB_DIR / f"{method}_seed_{seed}_test_emb.npy"

            # 1. Feature Extraction / Embedding
            if val_emb_path.exists() and test_emb_path.exists():
                print(f"Loading existing embeddings from {EMB_DIR.name} ...")
                val_emb = np.load(val_emb_path)
                test_emb = np.load(test_emb_path)
            else:
                val_emb, test_emb = runner_fn(
                    adata=adata,
                    val_indices=val_global_indices,
                    test_indices=test_global_indices,
                    seed=seed
                )
                np.save(val_emb_path, val_emb)
                np.save(test_emb_path, test_emb)
                print(f"Saved embeddings to {EMB_DIR.name} (val: {val_emb.shape}, test: {test_emb.shape})")

            # 2. Validation Unsupervised Stability Sweep (K in [5, 25])
            print(f"Running Validation stability selection on 10 folds for {method} (seed={seed}) ...")
            r_star, stab_report = select_unsupervised_resolution(
                val_emb=val_emb,
                bootstrap_folds=val_folds,
                resolutions=FROZEN_RESOLUTIONS,
                k_min=K_MIN,
                k_max=K_MAX,
                seed=seed
            )
            print(f" -> Locked r* = {r_star} (Stability Score: {stab_report['best_stability_score']:.4f})")

            # 3. Single-Pass Test Blind Evaluation
            print(f"Evaluating Test 10k centers at locked r*={r_star} ...")
            metrics, df_assignments = evaluate_test_clustering(
                test_emb=test_emb,
                frozen_r_star=r_star,
                test_cohort=test_cohort,
                adata=adata,
                method_name=method,
                seed=seed
            )

            # Export assignments CSV
            assign_csv_path = ASSIGN_DIR / f"{method}_seed_{seed}_test_assignments.csv"
            df_assignments.to_csv(assign_csv_path, index=False)
            print(f"Saved cell assignments to {assign_csv_path.name}")

            # Export per-seed JSON
            res_json_path = RES_DIR / f"{method}_seed_{seed}_results.json"
            metrics["elapsed_seconds"] = float(time.time() - t_start)
            metrics["stability_selection"] = stab_report
            with open(res_json_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            method_seed_metrics.append(metrics)
            print(f"[{method.upper()} Seed {seed} Result] Macro ARI: {metrics['macro_ari']:.4f}, Macro NMI: {metrics['macro_nmi']:.4f}, Worst-Donor ARI: {metrics['worst_donor_ari']:.4f}, ASW: {metrics['asw']:.4f}, K: {metrics['n_clusters']}")

    # Aggregate all completed methods to build official Master Table 5 and refresh Manifest
    from scripts.benchmark.export_sea_ad_table5 import export_table5, export_manifest
    df_table5 = export_table5()
    manifest = export_manifest()

    print("\n" + "=" * 110)
    print("STAGE 5 BASELINE BENCHMARK MASTER TABLE (WITH PARADIGMS):")
    print("=" * 110)
    print(df_table5.to_string(index=False))
    print("=" * 110)
    print(f"Table 5 exported to: {BASELINES_DIR / 'table5_sea_ad_baseline_benchmark.csv'}")
    print(f"Manifest exported to: {BASELINES_DIR / 'sea_ad_baselines_manifest.json'}")


if __name__ == "__main__":
    main()
