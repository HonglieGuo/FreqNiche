"""
scripts/benchmark/prepare_full_slice_manifests.py

Generates:
1. preexisting_assets_manifest.json (SHA-256 signatures of existing figures/tables/provenances)
2. smoke_1000_cells_manifest.json (Deterministic spatially contiguous 1,000-cell patch with ordered/sorted/file SHA)
3. bootstrap_manifest_11293_10fold.json (Strictly bound 10-fold 80% subsample manifest with barcode binding & self-hash)
"""

import hashlib
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "baselines/Baselines_outputs/runs/human_skin/2026-09-21/20260921_human_skin_full_slice_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = ROOT / "datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad"
TARGET_SAMPLE = "202309281027_AutopsyBFFPE092823_VMSC02301_region_1"

PREEXISTING_FILES = [
    ROOT / "Paper/FreqNiche-zh/figures/fig_human_skin_spatial_domain_umap.pdf",
    ROOT / "Paper/FreqNiche-zh/figures/fig_human_skin_spatial_domain_umap.png",
    ROOT / "Paper/FreqNiche-zh/figures/fig_human_skin_spatial_domain_umap.svg",
    ROOT / "Paper/FreqNiche-zh/figures/fig_human_skin_spatial_domain_umap_provenance.json",
    ROOT / "Paper/FreqNiche-zh/figures/fig_supp_all_methods_spatial_domain.pdf",
    ROOT / "Paper/FreqNiche-zh/figures/fig_supp_all_methods_spatial_domain.png",
    ROOT / "Paper/FreqNiche-zh/figures/fig_supp_all_methods_spatial_domain.svg",
    ROOT / "Paper/FreqNiche-zh/figures/fig_supp_all_methods_spatial_domain_provenance.json",
    ROOT / "Paper/FreqNiche-zh/figures/fig_supp_all_methods_umap_comparison.pdf",
    ROOT / "Paper/FreqNiche-zh/figures/fig_supp_all_methods_umap_comparison.png",
    ROOT / "Paper/FreqNiche-zh/figures/fig_supp_all_methods_umap_comparison.svg",
    ROOT / "Paper/FreqNiche-zh/figures/fig_supp_all_methods_umap_comparison_provenance.json",
    ROOT / "Paper/FreqNiche-zh/figures/fig5_xenium_large_scale_benchmark.pdf",
    ROOT / "Paper/FreqNiche-zh/figures/fig5_xenium_large_scale_benchmark.png",
    ROOT / "Paper/FreqNiche-zh/figures/fig5_xenium_large_scale_benchmark.svg",
    ROOT / "Paper/FreqNiche-zh/figures/fig5_xenium_large_scale_benchmark_provenance.json",
    ROOT / "baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final/full_slice/d028_region1_full_11293_z.npy",
    ROOT / "baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final/full_slice/d028_region1_full_11293_pred.csv",
    ROOT / "baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final/full_slice/d028_region1_full_11293_provenance.json",
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_barcode_hashes(barcodes: list[str]) -> tuple[str, str]:
    # Delimiter "\n" prevents concatenation ambiguity
    ordered_str = "\n".join(barcodes) + "\n"
    sorted_str = "\n".join(sorted(barcodes)) + "\n"
    return sha256_bytes(ordered_str.encode("utf-8")), sha256_bytes(sorted_str.encode("utf-8"))


def generate_preexisting_manifest():
    manifest = {}
    print(">>> Generating preexisting assets SHA-256 manifest...")
    for f in PREEXISTING_FILES:
        if f.exists():
            h = sha256_file(f)
            rel = str(f.relative_to(ROOT)).replace("\\", "/")
            manifest[rel] = {
                "size_bytes": f.stat().st_size,
                "sha256": h,
            }
            print(f"  [LOCKED] {rel} ({h[:16]}...)")
        else:
            print(f"  [WARNING] File missing: {f}")

    out_path = OUT_DIR / "preexisting_assets_manifest.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"[OK] Preexisting manifest saved: {out_path} ({len(manifest)} files locked)\n")


def generate_manifests():
    print(f">>> Loading AnnData from {DATASET_PATH}...")
    adata = ad.read_h5ad(DATASET_PATH, backed="r")
    mask = adata.obs["sample_id"] == TARGET_SAMPLE
    assert mask.sum() == 11293, f"Expected 11293 cells, got {mask.sum()}"

    obs_slice = adata.obs.loc[mask]
    barcodes_all = obs_slice.index.tolist()
    coords = adata.obsm["spatial"][mask.values]

    full_ordered_sha, full_sorted_sha = compute_barcode_hashes(barcodes_all)

    # 1. Smoke 1000 cells manifest
    center_x = float(np.median(coords[:, 0]))
    center_y = float(np.median(coords[:, 1]))
    dists = np.sqrt((coords[:, 0] - center_x) ** 2 + (coords[:, 1] - center_y) ** 2)

    smoke_idx = np.argsort(dists)[:1000]
    smoke_idx_sorted = np.sort(smoke_idx)
    smoke_barcodes = [barcodes_all[i] for i in smoke_idx_sorted]
    smoke_coords = coords[smoke_idx_sorted]

    smoke_ordered_sha, smoke_sorted_sha = compute_barcode_hashes(smoke_barcodes)

    smoke_manifest = {
        "dataset": "human_skin_merfish",
        "sample_id": TARGET_SAMPLE,
        "n_slice_cells": 11293,
        "n_smoke_cells": len(smoke_barcodes),
        "spatial_center": [center_x, center_y],
        "max_distance_to_center_um": float(np.max(dists[smoke_idx])),
        "ordered_barcodes_sha256": smoke_ordered_sha,
        "sorted_barcodes_sha256": smoke_sorted_sha,
        "full_slice_ordered_barcodes_sha256": full_ordered_sha,
        "full_slice_sorted_barcodes_sha256": full_sorted_sha,
        "barcodes": smoke_barcodes,
        "coords_bbox": {
            "x_min": float(np.min(smoke_coords[:, 0])),
            "x_max": float(np.max(smoke_coords[:, 0])),
            "y_min": float(np.min(smoke_coords[:, 1])),
            "y_max": float(np.max(smoke_coords[:, 1])),
        },
    }

    payload_smoke_str = json.dumps(smoke_manifest, sort_keys=True)
    smoke_manifest["manifest_self_sha256"] = sha256_bytes(payload_smoke_str.encode("utf-8"))

    smoke_out_path = OUT_DIR / "smoke_1000_cells_manifest.json"
    with open(smoke_out_path, "w", encoding="utf-8") as fp:
        json.dump(smoke_manifest, fp, indent=2)
    print(f"[OK] Smoke 1000 cells manifest saved: {smoke_out_path}")
    print(f"  Smoke Ordered Barcode SHA: {smoke_ordered_sha}")
    print(f"  Smoke Manifest Self-Hash: {smoke_manifest['manifest_self_sha256']}")

    # 2. Bootstrap manifest (11,293 cells, 10-fold, 80% subsample, Seed 42)
    seed = 42
    n_bootstraps = 10
    subsample_ratio = 0.80
    n_total_cells = 11293
    n_subsample = int(n_total_cells * subsample_ratio)

    rng = np.random.RandomState(seed)
    folds = {}
    for i in range(n_bootstraps):
        indices = rng.choice(n_total_cells, size=n_subsample, replace=False).tolist()
        assert len(indices) == len(set(indices)), f"Fold {i} has duplicate indices!"
        assert all(0 <= idx < n_total_cells for idx in indices), f"Fold {i} has out-of-range index!"
        folds[f"fold_{i}"] = sorted(indices)

    bootstrap_data = {
        "dataset": "human_skin_merfish",
        "sample_id": TARGET_SAMPLE,
        "n_total_cells": n_total_cells,
        "subsample_ratio": subsample_ratio,
        "n_subsample": n_subsample,
        "n_bootstraps": n_bootstraps,
        "seed": seed,
        "full_slice_ordered_barcodes_sha256": full_ordered_sha,
        "full_slice_sorted_barcodes_sha256": full_sorted_sha,
        "folds": folds,
    }

    # Generate self-hash
    payload_str = json.dumps(bootstrap_data, sort_keys=True)
    bootstrap_data["manifest_self_sha256"] = sha256_bytes(payload_str.encode("utf-8"))

    bs_out_path = OUT_DIR / "bootstrap_manifest_11293_10fold.json"
    with open(bs_out_path, "w", encoding="utf-8") as fp:
        json.dump(bootstrap_data, fp, indent=2)
    print(f"[OK] Bootstrap manifest saved: {bs_out_path}")
    print(f"  Full Slice Ordered Barcode SHA: {full_ordered_sha}")
    print(f"  Bootstrap Manifest Self-Hash: {bootstrap_data['manifest_self_sha256']}")


if __name__ == "__main__":
    generate_preexisting_manifest()
    generate_manifests()
