"""
experiments_analysis/infer_full_slice/infer_d028_region1_full_slice.py

Performs full-slice inductive forward inference on D028 region_1 (11,293 cells)
using frozen FreqNiche-ST Seed 42 checkpoint.

Adheres strictly to the Execution Contracts:
1. Whitener fitted strictly on Training donors only (never on D028).
2. Exact checkpoint SHA-256 verified: a3e99c7b42b5ae9aeab08f02b0e8741b2a883e7747e67625bba5c9ec23f24848.
3. Plan B: Unsupervised Leiden clustering on the full slice at frozen r*=0.8, random_state=42.
4. Ground truth niche is NEVER used in embedding or clustering (strictly post-hoc reference only).
"""

import os
import sys
import json
import hashlib
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.freqniche_st.dataset import SubgraphDataset, collate_subgraphs
from src.freqniche_st.graph import build_global_spatial_graph
from src.freqniche_st.models import FreqNicheDiTModel
from src.freqniche_st.whitening import TrainOnlyWhitening

DATASET_PATH = ROOT / "datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad"
SPLITS_PATH = ROOT / "datasets/h5ad/human_skin_merfish/donor_splits.csv"
MODEL_PATH = ROOT / "outputs/runs/freqniche/human_skin/2026-09-08/20260908_human_skin_training_final/trial_24_seed_42/best_val_model.pt"
EXPECTED_CKPT_SHA256 = "a3e99c7b42b5ae9aeab08f02b0e8741b2a883e7747e67625bba5c9ec23f24848"

OUT_DIR = ROOT / "baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final/full_slice"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_sha256(filepath: Path | str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print(">>> D028 region_1 Full-Slice (11,293 Cells) Inductive Forward Inference")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Verify Checkpoint SHA-256
    assert MODEL_PATH.exists(), f"Missing model checkpoint: {MODEL_PATH}"
    actual_ckpt_hash = compute_sha256(MODEL_PATH)
    assert actual_ckpt_hash == EXPECTED_CKPT_SHA256, (
        f"Checkpoint hash mismatch!\nExpected: {EXPECTED_CKPT_SHA256}\nActual:   {actual_ckpt_hash}"
    )
    print(f"[OK] Checkpoint verified: {MODEL_PATH.name} (SHA-256: {actual_ckpt_hash[:16]}...)")

    # 2. Load AnnData & Standard Preprocessing
    print(f"Loading AnnData from {DATASET_PATH} ...")
    adata = ad.read_h5ad(DATASET_PATH)
    print(f"AnnData shape: {adata.shape}")

    print("Applying normalize_total(1e4) and log1p() ...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    X_global = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    coords_global = adata.obsm["spatial"]
    sample_ids = adata.obs["sample_id"].to_numpy()

    # 3. Identify all cells in D028 region_1
    slice_mask = (adata.obs["donor_id"] == "D028") & (adata.obs["sample_id"].str.contains("region_1"))
    n_slice_cells = int(slice_mask.sum())
    assert n_slice_cells == 11293, f"Expected 11293 cells in D028 region_1, got {n_slice_cells}"
    slice_indices = np.where(slice_mask.to_numpy())[0].astype(np.int64)
    print(f"[OK] Target slice D028 region_1 identified: exactly {n_slice_cells:,} cells.")

    # 4. Build Spatial Graph
    print("Building global spatial graph with R=50.0 um ...")
    P_global, A_global = build_global_spatial_graph(
        coords=coords_global,
        sample_ids=sample_ids,
        radius=50.0,
    )

    # 5. Fit Whitener strictly on Training Donors only
    print("Fitting whitener strictly on Training Donors (never on D028) ...")
    splits_df = pd.read_csv(SPLITS_PATH)
    train_donors = set(splits_df[splits_df["split"] == "train"]["donor_id"])
    assert "D028" not in train_donors, "D028 is a test donor and must NOT be in train_donors"
    train_mask = adata.obs["donor_id"].isin(train_donors).to_numpy()
    print(f"Fitting whitener on {train_mask.sum():,} train cells ...")
    whitener = TrainOnlyWhitening(k_components=32)
    whitener.fit(X_global[train_mask])

    # 6. Setup DataLoader for All 11,293 Cells
    print("Setting up SubgraphDataset for all 11,293 cells ...")
    slice_dataset = SubgraphDataset(
        center_indices=slice_indices,
        A_global=A_global,
        P_global=P_global,
        X_global=X_global,
        coords_global=coords_global,
        whitener=whitener,
        k_max=64,
        m_max=32,
        is_train=False,
        is_val_objective=False,
    )
    slice_loader = DataLoader(
        slice_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=collate_subgraphs,
        pin_memory=(device.type == "cuda"),
        num_workers=4,
    )

    # 7. Load Model & Forward Inference
    print("Loading FreqNiche-ST model and running forward inference ...")
    model = FreqNicheDiTModel(
        in_features=X_global.shape[1],
        d_model=128,
        n_enc_layers=3,
        n_dit_layers=2,
        n_heads=8,
        timesteps=50,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model.eval()

    embeddings_list = []
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(slice_loader):
            X_vis = batch["X_vis"].to(device)
            delta_s_vis = batch["delta_s_vis"].to(device)
            vis_pad_mask = batch["vis_pad_mask"].to(device)
            L_vis_cond = batch["L_vis_cond"].to(device)
            H_vis_cond = batch["H_vis_cond"].to(device)
            emb = model.encode_niche(
                X_vis=X_vis,
                delta_s_vis=delta_s_vis,
                vis_pad_mask=vis_pad_mask,
                L_vis_cond=L_vis_cond,
                H_vis_cond=H_vis_cond,
            )
            embeddings_list.append(emb.cpu().numpy())
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(slice_loader):
                print(f"  Processed {min((batch_idx + 1) * 128, n_slice_cells):,}/{n_slice_cells:,} cells ...")

    Z_full = np.concatenate(embeddings_list, axis=0)
    assert Z_full.shape == (n_slice_cells, 128), f"Expected ({n_slice_cells}, 128), got {Z_full.shape}"
    print(f"[OK] Full-slice embeddings extracted in {time.time() - t0:.2f}s, shape: {Z_full.shape}")

    # Save full-slice embeddings
    z_out_path = OUT_DIR / "d028_region1_full_11293_z.npy"
    np.save(z_out_path, Z_full)
    z_hash = compute_sha256(z_out_path)
    print(f"[OK] Saved full-slice embeddings to {z_out_path} (SHA-256: {z_hash})")

    # 8. Plan B: Unsupervised Leiden Clustering at Frozen r*=0.8
    print("\nExecuting Plan B: Unsupervised Leiden on full slice (k=15, frozen r*=0.8, seed=42) ...")
    adata_slice = ad.AnnData(X=Z_full)
    sc.pp.neighbors(adata_slice, n_neighbors=15, use_rep="X", random_state=42)
    sc.tl.leiden(adata_slice, resolution=0.8, key_added="cluster", random_state=42)
    pred_clusters = adata_slice.obs["cluster"].to_numpy().astype(int)
    n_unique_clusters = len(np.unique(pred_clusters))
    print(f"[OK] Plan B completed: {n_unique_clusters} clusters obtained on full slice D028 region_1.")

    # 9. Assemble and Save Prediction Dataframe
    slice_obs = adata.obs.iloc[slice_indices].copy()
    slice_df = pd.DataFrame({
        "cell_barcode": slice_obs.index.values,
        "center_x": slice_obs["center_x"].values,
        "center_y": slice_obs["center_y"].values,
        "niche": slice_obs["niche"].values,
        "cell_type": slice_obs["cell_type"].values,
        "pred_cluster": pred_clusters,
    })
    csv_out_path = OUT_DIR / "d028_region1_full_11293_pred.csv"
    slice_df.to_csv(csv_out_path, index=False)
    print(f"[OK] Saved prediction table to {csv_out_path}")

    # Save Provenance
    prov = {
        "task": "D028 region_1 Full-Slice Inductive Inference",
        "sample_id": "202309281027_AutopsyBFFPE092823_VMSC02301_region_1",
        "donor_id": "D028",
        "n_cells": n_slice_cells,
        "checkpoint": {
            "path": str(MODEL_PATH.relative_to(ROOT)),
            "sha256": actual_ckpt_hash
        },
        "whitener": {
            "fit_cohort": "Train donors only",
            "k_components": 32,
            "d028_included": False
        },
        "plan": "Plan B (Unsupervised Leiden at frozen r*=0.8)",
        "readout": {
            "k_neighbors": 15,
            "resolution": 0.8,
            "random_state": 42,
            "n_clusters_obtained": n_unique_clusters
        },
        "output_embeddings": {
            "filename": z_out_path.name,
            "shape": list(Z_full.shape),
            "sha256": z_hash
        }
    }
    prov_path = OUT_DIR / "d028_region1_full_11293_provenance.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved provenance metadata to {prov_path}")
    print("\n[SUCCESS] Full-slice inductive forward inference completed cleanly!")

if __name__ == "__main__":
    main()
