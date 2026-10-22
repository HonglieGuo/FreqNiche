"""Run STAGATE-Controlled Benchmark on Human Skin MERFISH.

Executes official STAGATE-pyG pipeline:
1. Spatial_Net graph with sample isolation (0 cross-sample edges) and R=50.0 um.
2. Max epochs truncated to 100 with early stopping to prevent compute explosion.
3. Official mclust/Louvain clustering as secondary readout.
4. Common Leiden Readout on Test 8,000 centers with r* selected on Validation 8,000 centers.
Runs 3 Seeds (42, 123, 2024).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmark.adapters.stagate import OfficialSTAGATEAdapter
from src.benchmark.evaluation_w24 import (
    evaluate_common_leiden_readout,
    evaluate_official_labels_readout,
    select_unsupervised_resolution_r_star,
)
from src.freqniche_st.runtime_paths import resolve_run_output

DATASET_PATH = ROOT / "datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad"
VAL_BARCODES_PATH = ROOT / "outputs/contracts/human_skin/validation_barcodes_8k.csv"
TEST_CENTERS_PATH = ROOT / "outputs/contracts/human_skin/test_centers_8k.csv"

OUT_DIR = resolve_run_output("baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final")
RES_DIR = OUT_DIR / "results"
EMB_DIR = OUT_DIR / "embeddings"


def compute_file_sha256(filepath: Path | str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="STAGATE Human Skin benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    args = parser.parse_args()
    print("=" * 80)
    print(">>> STAGATE-Controlled W2.4 Benchmark Runner")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    RES_DIR.mkdir(parents=True, exist_ok=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    val_df = pd.read_csv(VAL_BARCODES_PATH)
    test_df = pd.read_csv(TEST_CENTERS_PATH)
    assert len(val_df) == 8000
    assert len(test_df) == 8000
    test_barcodes = list(test_df["cell_barcode"])

    print(f"Loading AnnData from {DATASET_PATH} ...")
    adata = ad.read_h5ad(DATASET_PATH)
    print(f"AnnData shape: {adata.shape}")

    obs_names = list(adata.obs_names)
    bc_to_idx = {bc: idx for idx, bc in enumerate(obs_names)}
    val_indices = np.array([bc_to_idx[bc] for bc in val_df["cell_barcode"]], dtype=np.int64)
    test_indices = np.array([bc_to_idx[bc] for bc in test_barcodes], dtype=np.int64)

    seeds = args.seeds
    for seed in seeds:
        print(f"\n--- Running STAGATE-Controlled Seed {seed} ---")
        t_start = time.time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        adapter = OfficialSTAGATEAdapter(
            protocol="controlled",
            radius_cutoff=50.0,
            epochs=100,  # Truncated per protocol to avoid 47-hour explosion
        )

        res = adapter.fit_transform(
            adata,
            n_clusters=10,
            seed=seed,
            device=device,
        )
        elapsed = time.time() - t_start

        meth_gpu = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
        peak_rss = float(psutil.Process().memory_info().rss / (1024 * 1024))

        full_emb = res.embedding
        val_emb = full_emb[val_indices]
        test_emb = full_emb[test_indices]
        assert val_emb.shape[0] == 8000
        assert test_emb.shape[0] == 8000

        # Save val and test embeddings
        val_emb_path = EMB_DIR / f"stagate_controlled_seed_{seed}_val_z.npy"
        test_emb_path = EMB_DIR / f"stagate_controlled_seed_{seed}_test_z.npy"
        np.save(val_emb_path, val_emb)
        np.save(test_emb_path, test_emb)
        val_emb_sha = compute_file_sha256(val_emb_path)
        test_emb_sha = compute_file_sha256(test_emb_path)

        # Common Leiden Readout
        best_r, stability_report = select_unsupervised_resolution_r_star(
            val_emb,
            resolutions=[0.2, 0.4, 0.6, 0.8, 1.0],
            k_min=5,
            k_max=25,
            seed=42,
        )

        common_eval = evaluate_common_leiden_readout(
            test_embeddings=test_emb,
            test_metadata=test_df,
            frozen_r_star=best_r,
            seed=seed,
            compute_asw=True,
        )

        official_eval = None
        if res.canonical_labels is not None:
            test_official_labels = res.canonical_labels[test_indices]
            official_eval = evaluate_official_labels_readout(
                pred_labels=test_official_labels,
                test_metadata=test_df,
                label_source="stagate_official_clustering",
            )

        out_data = {
            "benchmark_type": "primary",
            "method": "STAGATE-Controlled",
            "seed": seed,
            "paradigm": "Transductive",
            "status": "PASS",
            "elapsed_sec": elapsed,
            "method_gpu_allocated_mb": meth_gpu,
            "system_gpu_used_peak_mb": None,
            "peak_rss_mb": peak_rss,
            "common_readout": {
                "stability_selection": stability_report,
                "test_evaluation": common_eval,
            },
            "official_readout": official_eval,
            "embedding_file": str(test_emb_path.relative_to(ROOT)),
            "embedding_sha256": test_emb_sha,
            "val_embedding_file": str(val_emb_path.relative_to(ROOT)),
            "val_embedding_sha256": val_emb_sha,
        }

        res_path = RES_DIR / f"stagate_controlled_seed_{seed}.json"
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)

        print(f"[OK] Seed {seed} Common Leiden Test Donor-Macro ARI: {common_eval['test_donor_macro_ari']:.4f}")
        print(f" -> Worst-Donor ARI : {common_eval['worst_donor_ari']:.4f}")
        print(f" -> Donor-Macro NMI : {common_eval['test_donor_macro_nmi']:.4f}")
        print(f" -> Wrote result to {res_path}")

    print("\n" + "=" * 80)
    print(">>> STAGATE-Controlled Benchmark Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
