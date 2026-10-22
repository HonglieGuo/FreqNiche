"""
Builds and cryptographically freezes the W2.7 Xenium Data Audit Manifest.

Performs read-only inspection of:
1. Primary multi-donor evaluation cohort: 6 NSCLC 5K samples from GSE311609.
2. Intra-section spatial case study cohort: 3 disjoint 10k ROIs from Xenium_Prime5K_Lung.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import anndata as ad
from src.freqniche_st.runtime_paths import resolve_run_output
from src.freqniche_st.model_config import load_model_config
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1048576):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    workspace_root = Path(__file__).resolve().parents[2]
    h5ad_root = workspace_root / "datasets" / "h5ad"
    gse_dir = h5ad_root / "xenium_ffpe_lungbreast_gse311609" / "samples"
    roi_dir = h5ad_root / "Xenium_Prime5K_Lung"
    output_dir = resolve_run_output("outputs/runs/freqniche/xenium_prime5k/2026-09-18/20260918_xenium_prime5k_final")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = load_model_config(output_dir=output_dir)
    manifest_out = output_dir / "xenium_data_audit_manifest.json"

    gse_samples_meta = [
        ("GSM9509134_NSCLC_5k_L1_L1.h5ad", "L1", "test"),
        ("GSM9509135_NSCLC_5k_L3_L3.h5ad", "L3", "train"),
        ("GSM9509136_NSCLC_5k_L5_L5.h5ad", "L5", "train"),
        ("GSM9509137_NSCLC_5k_L6_L6.h5ad", "L6", "validation"),
        ("GSM9509138_NSCLC_5k_L7_L7.h5ad", "L7", "train"),
        ("GSM9509139_NSCLC_5k_L8_L8.h5ad", "L8", "train"),
    ]

    roi_samples_meta = [
        ("xenium_prime5k_lung_roi1.h5ad", "roi1"),
        ("xenium_prime5k_lung_roi2.h5ad", "roi2"),
        ("xenium_prime5k_lung_roi3.h5ad", "roi3"),
    ]

    print("[1/3] Auditing primary 6-sample NSCLC 5K cohort ...")
    primary_details = {}
    reference_genes = None

    for fname, donor, split in gse_samples_meta:
        p = gse_dir / fname
        sha = compute_file_sha256(p)
        a = ad.read_h5ad(p, backed="r")
        if reference_genes is None:
            reference_genes = list(a.var_names)
        else:
            assert list(a.var_names) == reference_genes, f"Gene order mismatch in {fname}"

        coords = np.asarray(a.obsm["spatial"], dtype=np.float32)
        tree = cKDTree(coords)
        dists, _ = tree.query(coords, k=2)

        id_valid = int(a.obs["identity_valid"].sum()) if "identity_valid" in a.obs else 0
        niche_valid = int(a.obs["niche_valid"].sum()) if "niche_valid" in a.obs else 0

        primary_details[fname] = {
            "file_sha256": sha,
            "sample_id": fname.replace(".h5ad", ""),
            "donor_id": donor,
            "split": split,
            "n_cells": int(a.n_obs),
            "n_genes": int(a.n_vars),
            "x_min": float(coords[:, 0].min()),
            "x_max": float(coords[:, 0].max()),
            "y_min": float(coords[:, 1].min()),
            "y_max": float(coords[:, 1].max()),
            "nn_median_dist_um": round(float(np.median(dists[:, 1])), 2),
            "nn_dist_scope": "full_population_exact",
            "cell_type_valid_count": id_valid,
            "niche_valid_count": niche_valid,
        }
        print(f"  - {fname}: {a.shape}, split={split}, sha256={sha[:16]}..., nn_median={primary_details[fname]['nn_median_dist_um']} um")

    print("\n[2/3] Auditing optional 3-ROI spatial case study cohort ...")
    roi_details = {}
    available_roi_files = [(roi_dir / fname).is_file() for fname, _ in roi_samples_meta]
    if any(available_roi_files) and not all(available_roi_files):
        raise FileNotFoundError(
            "Xenium optional ROI cohort is incomplete; provide all three ROI H5AD files or none"
        )
    if not any(available_roi_files):
        print("  - Optional ROI H5AD files absent; primary six-sample training is unaffected.")
    for fname, roi_name in (roi_samples_meta if all(available_roi_files) else []):
        p = roi_dir / fname
        sha = compute_file_sha256(p)
        a = ad.read_h5ad(p)
        assert list(a.var_names) == reference_genes, f"Gene order mismatch in {fname}"

        coords = np.asarray(a.obsm["spatial"], dtype=np.float32)
        tree = cKDTree(coords)
        dists, _ = tree.query(coords, k=2)

        n_pseudo = int(len(a.obs["xenium_cluster"].unique())) if "xenium_cluster" in a.obs else None
        roi_details[fname] = {
            "file_sha256": sha,
            "roi_name": roi_name,
            "role": "intra_section_spatial_visualization_and_case_study",
            "n_cells": int(a.n_obs),
            "n_genes": int(a.n_vars),
            "x_min": float(coords[:, 0].min()),
            "x_max": float(coords[:, 0].max()),
            "y_min": float(coords[:, 1].min()),
            "y_max": float(coords[:, 1].max()),
            "nn_median_dist_um": round(float(np.median(dists[:, 1])), 2),
            "nn_dist_scope": "full_population_exact",
            "n_pseudo_clusters": n_pseudo,
        }
        print(f"  - {fname}: {a.shape}, roi={roi_name}, sha256={sha[:16]}..., nn_median={roi_details[fname]['nn_median_dist_um']} um")

    ordered_genes_sha = hashlib.sha256(("\n".join(reference_genes)).encode("utf-8")).hexdigest()

    manifest = {
        "manifest_version": "1.0.0",
        "created_at": "2026-09-17",
        "stage": "W2.7",
        "dataset_title": "10x Genomics Xenium Prime 5K Human NSCLC Benchmark",
        "gene_panel": {
            "panel_name": "Xenium Prime 5K Human Pan-Tissue & Pathways",
            "n_genes": len(reference_genes),
            "ordered_genes_sha256": ordered_genes_sha,
        },
        "coordinate_contracts": {
            "obsm_key": "spatial",
            "unit": "micron",
            "scale_factor_applied": 1.0,
            "neighbor_radius_um": 50.0,
            "gaussian_sigma_um": 16.6667,
        },
        "primary_cohort": {
            "cohort_name": "GSE311609_NSCLC_5K_Cohort",
            "role": "Multi-donor model training, validation tuning, and external generalization evaluation",
            "total_cells": sum(d["n_cells"] for d in primary_details.values()),
            "n_donors": 6,
            "n_samples": 6,
            "splits_summary": {
                "train": {
                    "donors": ["L3", "L5", "L7", "L8"],
                    "n_cells": sum(d["n_cells"] for d in primary_details.values() if d["split"] == "train"),
                    "samples": [k for k, d in primary_details.items() if d["split"] == "train"],
                },
                "validation": {
                    "donors": ["L6"],
                    "n_cells": sum(d["n_cells"] for d in primary_details.values() if d["split"] == "validation"),
                    "samples": [k for k, d in primary_details.items() if d["split"] == "validation"],
                },
                "test": {
                    "donors": ["L1"],
                    "n_cells": sum(d["n_cells"] for d in primary_details.values() if d["split"] == "test"),
                    "samples": [k for k, d in primary_details.items() if d["split"] == "test"],
                },
            },
            "samples": primary_details,
        },
        "visualization_roi_cohort": {
            "cohort_name": "Xenium_Prime5K_Lung_ROIs",
            "role": "Intra-section spatial domain continuity, marker enrichment and visualization case studies",
            "total_cells": sum(d["n_cells"] for d in roi_details.values()),
            "n_rois": len(roi_details),
            "available": bool(roi_details),
            "rois": roi_details,
        },
        "model_and_whitener_contract": {
            "architecture": (
                f"FreqNiche-ST (d_model={model_config['d_model']}, "
                f"n_enc={model_config['n_enc']}, n_dit={model_config['n_dit']}, "
                f"n_heads={model_config['n_heads']}, timesteps={model_config['timesteps']})"
            ),
            "champion_source": (
                str((output_dir / "model_config.json").relative_to(workspace_root))
                if os.environ.get("FREQNICHE_RUN_DIR")
                else "outputs/runs/freqniche/human_skin/2026-09-08/20260908_human_skin_training_final/champion_selection.json"
            ),
            "frozen_hyperparameters": {
                "trial_id": 24,
                "desc": "Run-specific model configuration",
                "d_model": model_config["d_model"],
                "n_enc": model_config["n_enc"],
                "n_dit": model_config["n_dit"],
                "n_heads": model_config["n_heads"],
                "timesteps": model_config["timesteps"],
                "radius": 50.0,
                "sigma": 16.6667,
                "k_max": model_config["k_max"],
                "rho": model_config["rho"],
                "p_center_mask": model_config["p_center_mask"],
                "m_max": model_config["m_max"],
                "lr": model_config["lr"],
                "wd": model_config["wd"],
                "lambda_cov": model_config["lambda_cov"]
            },
            "adaptation_mode": "Retrain on Xenium 5K Train Donors (Zero-shot across gene panels is impossible due to 5,001 vs 960/562/180 dimension mismatch)",
            "whitener_spec": {
                "input_dim": 5001,
                "output_dim": 32,
                "fit_scope": "train_only (strictly fit on Train donors L3, L5, L7, L8)",
                "matrix_shape": [5001, 32],
                "type": "PCA/ZCA Dimensionality Reduction Whitener",
            },
        },
        "evaluation_protocol": {
            "paradigm": "Unsupervised Non-ARI Spatial Biology Protocol (No ground-truth niche annotations exist)",
            "endpoints": [
                "1. Local Spatial Homogeneity / Moran's I with Spatial Shuffle Permutation Negative Control",
                "2. Cross-ROI Niche Topographic Replicability (Cosine Similarity of Niche Profiles across ROIs)",
                "3. Exploratory Intra-Cell-Type Niche Diversity (Requires independent external cell type reference)",
                "4. Anatomical Marker Gene Spatial Enrichment (Biological attribution clues, not standalone ground truth)",
                "5. Computational Scalability (Throughput in cells/sec, Peak VRAM, Peak RAM on 5,001 genes)",
            ],
        },
    }

    print("\n[3/3] Writing audit manifest ...")
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[OK] Successfully wrote {manifest_out}")


if __name__ == "__main__":
    main()
