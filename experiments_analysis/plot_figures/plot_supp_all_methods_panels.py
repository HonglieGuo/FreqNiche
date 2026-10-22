"""
experiments_analysis/plot_figures/plot_supp_all_methods_panels.py

Generates:
1. Supplementary Figure S1 (2x3 Grid):
   All-Method Slice Spatial Domain Identification (D028 region_1, 669 cells)
   - A: Ground Truth Niche
   - B: FreqNiche-ST (Ours, Inductive, r*=0.8)
   - C: CellCharter (Transductive, r*=0.2)
   - D: BANKSY (Transductive, r*=0.2)
   - E: STAGATE-Controlled (Transductive, r*=0.2)
   - F: CellNiche-Controlled (Transductive, r*=0.2)
   Output: Paper/FreqNiche-ST-zh/figures/fig_supp_all_methods_spatial_domain.pdf/png/json

2. Supplementary Figure S2 (2x3 Grid):
   All-Method Latent Space UMAP Comparison (8,000 cells)
   All 6 panels colored by GROUND TRUTH NICHE with a unified color palette:
   - A: FreqNiche-ST (Ours, 128D, Inductive)
   - B: CellCharter (80D, Transductive)
   - C: BANKSY (20D, Transductive)
   - D: STAGATE-Controlled (30D, Transductive)
   - E: CellNiche-Controlled (128D, Transductive)
   - F: Raw PCA (50D Baseline, fit on Val 8k, transform on Test 8k)
   Output: Paper/FreqNiche-ST-zh/figures/fig_supp_all_methods_umap_comparison.pdf/png/json
"""

import os
from pathlib import Path
import sys
import json
import hashlib
import time
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from experiments_analysis.figure_style import set_publication_style, save_publication_figure

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def run_common_leiden(z: np.ndarray, resolution: float, seed: int = 42) -> np.ndarray:
    """Standardized Common Readout Leiden clustering matching evaluation_w24.py."""
    adata_temp = ad.AnnData(X=z)
    sc.pp.neighbors(adata_temp, n_neighbors=15, use_rep="X", random_state=seed)
    sc.tl.leiden(adata_temp, resolution=resolution, key_added="cluster", random_state=seed)
    return adata_temp.obs['cluster'].to_numpy().astype(int)

def compute_umap(z: np.ndarray, seed: int = 42) -> np.ndarray:
    """Standardized UMAP projection with fixed hyper-parameters."""
    adata_temp = ad.AnnData(X=z)
    sc.pp.neighbors(adata_temp, n_neighbors=30, random_state=seed, use_rep="X")
    sc.tl.umap(adata_temp, min_dist=0.3, random_state=seed)
    return adata_temp.obsm['X_umap']

def main():
    print("=" * 80)
    print("PLOTTING ALL-METHOD SUPPLEMENTARY FIGURES S1 (SPATIAL) & S2 (UMAP)")
    print("=" * 80)
    
    base_dir = str(Path(__file__).resolve().parents[2])
    if not os.path.exists(base_dir):
        base_dir = "d:/code/research-projects/SpaPEFT"
        
    out_dir = f"{base_dir}/Paper/FreqNiche-zh/figures"
    os.makedirs(out_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Load Metadata & Strict Row Order Assertions
    # -------------------------------------------------------------
    mapping_path = f"{base_dir}/outputs/freqniche_wave_a_diagnostics/wave_a_main/test_centers_mapping.csv"
    val_bcs_path = f"{base_dir}/outputs/freqniche_wave_a_diagnostics/original_val_barcodes.csv"
    h5ad_path = f"{base_dir}/datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad"
    
    print("Loading metadata...")
    test_df = pd.read_csv(mapping_path)
    val_df = pd.read_csv(val_bcs_path)
    
    assert len(test_df) == 8000, f"Expected 8000 test centers, got {len(test_df)}"
    assert np.all(test_df['embedding_row'].values == np.arange(8000)), "embedding_row must strictly match 0..7999"
    assert len(val_df) == 8000, f"Expected 8000 val centers, got {len(val_df)}"
    print(f"[OK] Metadata verified: 8,000 test centers, strictly 0..7999 aligned.")
    
    # Check expected donors in test set
    expected_donors = ['D028', 'D082', 'D145', 'D151']
    assert sorted(test_df['donor_id'].unique().tolist()) == expected_donors, "Donors mismatch"
    for d in expected_donors:
        n_d = (test_df['donor_id'] == d).sum()
        assert n_d == 2000, f"Expected 2000 cells for donor {d}, got {n_d}"
    print(f"[OK] 4 Test donors balanced: 2,000 cells each.")
    
    # -------------------------------------------------------------
    # 2. Load 5 Baseline Embeddings & Verify Official SHA-256 Hashes
    # -------------------------------------------------------------
    emb_files = {
        'FreqNiche-ST': {
            'path': f"{base_dir}/baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final/embeddings/freqniche_st_v1_trial24_seed_42_test_z.npy",
            'sha256': "a9fca8e45ae20e22bc0dccdce63fdf86a6f7a3d96135240c1562894f9d541bb7",
            'r_star': 0.8,
            'expected_clusters': 16,
            'paradigm': 'Inductive'
        },
        'CellCharter': {
            'path': f"{base_dir}/baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final/embeddings/cellcharter_canonical_seed_42_test_z.npy",
            'sha256': "86a91b9e9dfffd7655c08e5df04f37f41acf07adb16aaad561b5ebc1bf6d4d43",
            'r_star': 0.2,
            'expected_clusters': 6,
            'paradigm': 'Transductive'
        },
        'BANKSY': {
            'path': f"{base_dir}/baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final/embeddings/banksy_controlled_test_z.npy",
            'sha256': "71ed8f7722aa040e0d259f0bf5f995ef3e3109aab5e60ed5906b76ef6738ac92",
            'r_star': 0.2,
            'expected_clusters': 8,
            'paradigm': 'Transductive'
        },
        'STAGATE': {
            'path': f"{base_dir}/baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final/embeddings/stagate_controlled_seed_42_test_z.npy",
            'sha256': "af4a1298a1e1cbe59f65a3d9a44d627afea2fc5a33bba6a19e4ae5e5b5247294",
            'r_star': 0.2,
            'expected_clusters': 9,
            'paradigm': 'Transductive'
        },
        'CellNiche': {
            'path': f"{base_dir}/baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final/embeddings/cellniche_controlled_seed_42_test_z.npy",
            'sha256': "01ddb0a83e57a38a1534877de049a912ca5ff47eeef58517bdf27b58743178ac",
            'r_star': 0.2,
            'expected_clusters': 12,
            'paradigm': 'Transductive'
        }
    }
    
    embeddings = {}
    clusters = {}
    print("\nVerifying embedding SHA-256 and shapes...")
    for method, info in emb_files.items():
        actual_hash = compute_sha256(info['path'])
        assert actual_hash == info['sha256'], f"SHA-256 mismatch for {method}: expected {info['sha256']}, got {actual_hash}"
        z = np.load(info['path'])
        assert z.shape[0] == 8000, f"Expected 8000 rows for {method}, got {z.shape[0]}"
        embeddings[method] = z
        print(f" - {method:15s}: SHA-256 matched, Shape {z.shape}")
        
        # Execute standardized Common Leiden with frozen r*
        cl = run_common_leiden(z, resolution=info['r_star'], seed=42)
        n_cl = len(np.unique(cl))
        assert n_cl == info['expected_clusters'], f"Cluster count mismatch for {method}: expected {info['expected_clusters']}, got {n_cl}"
        clusters[method] = cl
        test_df[f'pred_{method}'] = cl
        print(f"   Common Readout r*={info['r_star']}: exactly {n_cl} clusters confirmed.")
        
    # -------------------------------------------------------------
    # 3. Compute Raw PCA (50D, fit on Val 8k, transform Test 8k)
    # -------------------------------------------------------------
    print("\nExtracting expression and computing Raw PCA (fit on Val 8k, transform on Test 8k)...")
    adata = sc.read_h5ad(h5ad_path, backed='r')
    
    val_bcs = val_df['cell_barcode'].tolist()
    test_bcs = test_df['cell_barcode'].tolist()
    
    # Load physical coordinates for test centers
    coords_df = adata.obs.loc[test_bcs, ['center_x', 'center_y']].copy()
    test_df['center_x'] = coords_df['center_x'].values
    test_df['center_y'] = coords_df['center_y'].values
    
    X_val_raw = adata[val_bcs, :].X
    if hasattr(X_val_raw, 'toarray'):
        X_val_raw = X_val_raw.toarray()
    else:
        X_val_raw = np.array(X_val_raw)
        
    X_test_raw = adata[test_bcs, :].X
    if hasattr(X_test_raw, 'toarray'):
        X_test_raw = X_test_raw.toarray()
    else:
        X_test_raw = np.array(X_test_raw)
        
    pca = PCA(n_components=50, random_state=42)
    pca.fit(X_val_raw)  # Strictly fit on validation set
    z_raw_pca = pca.transform(X_test_raw)
    assert z_raw_pca.shape == (8000, 50), f"Expected (8000, 50), got {z_raw_pca.shape}"
    embeddings['Raw PCA'] = z_raw_pca
    print(f"[OK] Raw PCA 50D computed: {z_raw_pca.shape}")
    
    # -------------------------------------------------------------
    # 4. Compute 2D UMAP for All 6 Methods
    # -------------------------------------------------------------
    umaps = {}
    methods_for_umap = ['FreqNiche-ST', 'CellCharter', 'BANKSY', 'STAGATE', 'CellNiche', 'Raw PCA']
    print("\nComputing 2D UMAP for all 6 methods (fixed n_neighbors=30, min_dist=0.3, seed=42)...")
    t0 = time.time()
    for method in methods_for_umap:
        print(f" - Computing UMAP for {method}...")
        u = compute_umap(embeddings[method], seed=42)
        assert u.shape == (8000, 2)
        umaps[method] = u
    print(f"All 6 UMAPs computed in {time.time()-t0:.2f}s.")
    
    # -------------------------------------------------------------
    # 5. Extract Single Physical Slice: D028 region_1 (669 cells)
    # -------------------------------------------------------------
    slice_mask = (test_df['donor_id'] == 'D028') & (test_df['sample_id'].str.contains('region_1'))
    slice_df = test_df[slice_mask].copy()
    n_slice_cells = len(slice_df)
    assert n_slice_cells == 669, f"Expected 669 cells in D028 region_1, got {n_slice_cells}"
    print(f"[OK] Selected single physical slice: D028 region_1 with {n_slice_cells} cells.")
    
    # Ground Truth Niche categories and global palette (shared across all UMAPs and Ground Truth slice)
    niche_cats = sorted(test_df['niche'].unique())
    niche_palette = sns.color_palette("tab10", len(niche_cats))
    niche2color = {cat: niche_palette[i] for i, cat in enumerate(niche_cats)}
    
    # Load full physical slice for histological background context (11,293 cells)
    full_slice_pred_path = f"{base_dir}/baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final/full_slice/d028_region1_full_11293_pred.csv"
    assert os.path.exists(full_slice_pred_path), f"Missing full slice: {full_slice_pred_path}"
    full_slice_df = pd.read_csv(full_slice_pred_path)
    print(f"Loaded full slice background: {len(full_slice_df)} cells.")
    
    # -------------------------------------------------------------
    # 6. PLOT Supplementary Figure S1: 2x3 Slice Spatial Domain Maps
    # -------------------------------------------------------------
    set_publication_style()
    fig_s1, axes_s1 = plt.subplots(2, 3, figsize=(15.6, 9.8))
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.97, hspace=0.30, wspace=0.25)
    
    # (A) Ground Truth
    axA = axes_s1[0, 0]
    # Draw full-slice gray background context
    axA.scatter(full_slice_df['center_x'], full_slice_df['center_y'], c='#dcdcdc', s=4, alpha=0.45, edgecolors='none', rasterized=True)
    for cat in niche_cats:
        sub = slice_df[slice_df['niche'] == cat]
        if len(sub) > 0:
            axA.scatter(sub['center_x'], sub['center_y'], c=[niche2color[cat]], s=16, alpha=0.92, edgecolors='none', rasterized=True, label=cat)
    axA.set_title("A  True Niches (D028 region_1, N=669 centers)", loc='left', fontweight='bold', pad=8)
    axA.set_xlabel("Spatial X (um)")
    axA.set_ylabel("Spatial Y (um)")
    axA.legend(loc='lower left', bbox_to_anchor=(0.0, -0.02), ncol=2, fontsize=7.5, frameon=True, framealpha=0.88, handletextpad=0.2, markerscale=1.5)
    
    # Method configurations for B-F: (panel_letter, method_key, display_name, paradigm, r_star, ax, pal_name)
    method_slots = [
        ('B', 'FreqNiche-ST', 'FreqNiche', 'Ours, Inductive', 0.8, axes_s1[0, 1], 'tab20'),
        ('C', 'CellCharter', 'CellCharter', 'Transductive', 0.2, axes_s1[0, 2], 'Set2'),
        ('D', 'BANKSY', 'BANKSY', 'Transductive', 0.2, axes_s1[1, 0], 'Dark2'),
        ('E', 'STAGATE', 'STAGATE', 'Transductive', 0.2, axes_s1[1, 1], 'Paired'),
        ('F', 'CellNiche', 'CellNiche', 'Transductive', 0.2, axes_s1[1, 2], 'tab20b')
    ]
    
    for panel_letter, method, display_name, paradigm, r_star, ax, pal_name in method_slots:
        # Draw full-slice gray background context
        ax.scatter(full_slice_df['center_x'], full_slice_df['center_y'], c='#dcdcdc', s=4, alpha=0.45, edgecolors='none', rasterized=True)
        preds = slice_df[f'pred_{method}'].values
        unique_preds = sorted(np.unique(preds))
        n_c = emb_files[method]['expected_clusters']
        pal = sns.color_palette(pal_name, n_c)
        
        for cat in unique_preds:
            sub = slice_df[slice_df[f'pred_{method}'] == cat]
            ax.scatter(sub['center_x'], sub['center_y'], c=[pal[cat % len(pal)]], s=16, alpha=0.92, edgecolors='none', rasterized=True, label=f"D{cat}")
            
        ax.set_title(f"{panel_letter}  {display_name} ({paradigm}, r*={r_star})", loc='left', fontweight='bold', pad=8)
        ax.set_xlabel("Spatial X (um)")
        ax.set_ylabel("Spatial Y (um)")
        ncol = 4 if len(unique_preds) > 10 else 3
        ax.legend(loc='lower left', bbox_to_anchor=(0.0, -0.02), ncol=ncol, fontsize=7.2, frameon=True, framealpha=0.85, handletextpad=0.2, markerscale=1.5)
        
    s1_prov = {
        "figure": "Supplementary Figure S1: All-Method Physical Slice Spatial Domain Identification (2x3)",
        "source_method_name": "FreqNiche-ST v1.0 Final",
        "display_method_name": "FreqNiche",
        "representative_seed": 42,
        "physical_slice": {
            "donor_id": "D028",
            "sample_id": "202309281027_AutopsyBFFPE092823_VMSC02301_region_1",
            "n_evaluation_centers": 669,
            "n_background_cells": len(full_slice_df),
            "nature": "Single physical tissue region (largest among D028 regions) with full slice background"
        },
        "readout_paradigm": "Strict Common Readout (Leiden on latent graph with frozen r*)",
        "methods": {
            method: {
                "paradigm": emb_files[method]['paradigm'],
                "frozen_r_star": emb_files[method]['r_star'],
                "seed_42_common_readout_total_clusters": emb_files[method]['expected_clusters'],
                "slice_active_clusters": int(len(np.unique(slice_df[f'pred_{method}'])))
            } for method in emb_files
        },
        "embedding_sha256_verified": {
            method: emb_files[method]['sha256'] for method in emb_files
        },
        "scope_note": "Includes all methods evaluated on the Human Skin full-cohort primary benchmark."
    }
    s1_basename = "fig_supp_all_methods_spatial_domain"
    save_publication_figure(fig_s1, s1_basename, out_dir, metadata=s1_prov)
    
    # -------------------------------------------------------------
    # 7. PLOT Supplementary Figure S2: 2x3 All-Method UMAP Comparison
    # -------------------------------------------------------------
    fig_s2, axes_s2 = plt.subplots(2, 3, figsize=(15.6, 9.8))
    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.06, right=0.97, hspace=0.30, wspace=0.25)
    
    umap_slots = [
        ('A', 'FreqNiche-ST', 'FreqNiche', '128D, Inductive', axes_s2[0, 0]),
        ('B', 'CellCharter', 'CellCharter', '80D, Transductive', axes_s2[0, 1]),
        ('C', 'BANKSY', 'BANKSY', '20D, Transductive', axes_s2[0, 2]),
        ('D', 'STAGATE', 'STAGATE', '30D, Transductive', axes_s2[1, 0]),
        ('E', 'CellNiche', 'CellNiche', '128D, Transductive', axes_s2[1, 1]),
        ('F', 'Raw PCA', 'Raw PCA', '50D, Validation-fitted', axes_s2[1, 2])
    ]
    
    umap_metrics = {}
    for panel_letter, method, display_name, desc, ax in umap_slots:
        u = umaps[method]
        
        # Calculate NMI and ARI vs Ground Truth Niche for embedding Leiden clusters if available
        if method in clusters:
            ari_m = float(adjusted_rand_score(test_df['niche'], clusters[method]))
            nmi_m = float(normalized_mutual_info_score(test_df['niche'], clusters[method]))
            umap_metrics[method] = {'ari_vs_niche': ari_m, 'nmi_vs_niche': nmi_m}
        else:
            umap_metrics[method] = {'note': 'Linear PCA baseline (Validation-fitted)'}
            
        for cat in niche_cats:
            mask = (test_df['niche'] == cat)
            ax.scatter(u[mask, 0], u[mask, 1], c=[niche2color[cat]], s=7, alpha=0.75, edgecolors='none', rasterized=True, label=cat)
            
        ax.set_title(f"{panel_letter}  {display_name} ({desc})", loc='left', fontweight='bold', pad=8)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        
    # Put shared global legend at the bottom
    handles, labels = axes_s2[0, 0].get_legend_handles_labels()
    fig_s2.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.51, 0.02), ncol=5, fontsize=8.2, frameon=True, framealpha=0.9, markerscale=2.0)
    
    s2_prov = {
        "figure": "Supplementary Figure S2: All-Method Latent Space UMAP Comparison Colored by Ground Truth Niche (2x3)",
        "source_method_name": "FreqNiche-ST v1.0 Final",
        "display_method_name": "FreqNiche",
        "representative_seed": 42,
        "test_cohort": {
            "n_cells_total": 8000,
            "donors": ["D028", "D082", "D145", "D151"]
        },
        "color_palette": "Global shared tab10 palette for 10 Ground Truth Niches",
        "embedding_sha256_verified": {
            method: emb_files[method]['sha256'] for method in emb_files
        },
        "metrics_seed_42_common_readout": umap_metrics,
        "metrics_source_note": "NMI and ARI metrics reported in provenance are computed from frozen Common Leiden clustering on the original high-dimensional embeddings, not from 2D UMAP coordinates.",
        "raw_pca_contract": "Strictly fit PCA(n_components=50, random_state=42) on Validation 8,000 centers, transformed on Test 8,000 centers. Zero test data fitting.",
        "scope_note": "Includes all methods evaluated on the Human Skin full-cohort primary benchmark + Raw PCA baseline."
    }
    s2_basename = "fig_supp_all_methods_umap_comparison"
    save_publication_figure(fig_s2, s2_basename, out_dir, metadata=s2_prov)
    
    print("\n[ALL DONE] Successfully generated both Supplementary Figure S1 and S2 with strict contract verification!")

if __name__ == "__main__":
    main()
