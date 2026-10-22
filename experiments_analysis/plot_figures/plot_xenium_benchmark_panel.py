"""
experiments_analysis/plot_figures/plot_xenium_benchmark_panel.py

Figure 5: 10x Xenium 5K (5,000 Genes, 67,705 Test Cells) Benchmark Panel & Empirical Hardware Profile
- Subplot A: Spatial Autocorrelation (Moran's I: Top-5 PCs & Cluster Indicator Fields)
- Subplot B: Local Microenvironment Purity (R=50um) & Cross-Seed ARI
- Subplot C: Empirical Peak VRAM across Scales (1k, 5k, 10k Cells) for Baseline Smoke Tests
  * Strictly adheres to user audit adjudication:
  * No continuous VRAM trip curves; discrete empirical scale-peak bars.
  * CellNiche marked as 10k PASS (8,416 MiB) and pre-excluded from full 67k test due to steep empirical VRAM scaling to avoid OOM risk.
  * NO invented or non-identical experimental bars for FreqNiche-ST in 1k/5k/10k smoke ladder.
- Subplot D: Effective Embedding Rank & Dimensional Capacity (SVD Entropy)
  * Measures resistance to dimensional collapse; NO unfounded claims of 'no over-smoothing'.
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from experiments_analysis.figure_style import set_publication_style, save_publication_figure, PALETTE_METHODS

def parse_mean_std(val_str):
    if pd.isna(val_str):
        return 0.0, 0.0
    val_str = str(val_str).strip()
    if '±' in val_str:
        parts = val_str.split('±')
        return float(parts[0].strip()), float(parts[1].strip())
    try:
        return float(val_str), 0.0
    except ValueError:
        return 0.0, 0.0

def main():
    print("=" * 70)
    print("Plotting Xenium 5K Benchmark & Hardware Profile Panel (Refined)...")
    print("=" * 70)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--xenium-table', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    data_dir = str(Path(__file__).resolve().parents[2])
    figures_out_dir = str(args.output_dir or Path(data_dir) / "Paper/FreqNiche-zh/figures")
    
    # 1. Load Table 7
    t7_path = str(args.xenium_table or Path(data_dir) / "outputs/runs/freqniche/xenium_prime5k/2026-09-18/20260918_xenium_prime5k_final/table7_benchmark_synthesis.csv")
    df7 = pd.read_csv(t7_path)
    
    # 2. Load Smoke Telemetry
    smoke_path = f"{data_dir}/baselines/Baselines_outputs/runs/xenium_prime5k/2026-09-18/20260918_xenium_prime5k_baselines/smoke/smoke_telemetry.json"
    with open(smoke_path, 'r') as f:
        smoke_data = json.load(f)
        
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.5))
    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.08, right=0.96, hspace=0.36, wspace=0.26)
    
    methods = df7['Method'].tolist()
    clean_labels = [m.replace(' (Ours)', '').replace('-Controlled', '').replace(' (px_resdiff)', '').replace('FreqNiche-ST', 'FreqNiche').replace('CellCharter-PCA', 'CellCharter').replace('CellCharter-Canonical', 'CellCharter') for m in methods]
    x_m = np.arange(len(methods))
    width = 0.36
    
    # --- Subplot A: Spatial Autocorrelation (Moran's I) ---
    axA = axes[0, 0]
    moran_pc = [parse_mean_std(v) for v in df7['Moran_I_Top5_PCs']]
    moran_cl = [parse_mean_std(v) for v in df7['Moran_I_Clusters']]
    
    pc_m, pc_s = [p[0] for p in moran_pc], [p[1] for p in moran_pc]
    cl_m, cl_s = [p[0] for p in moran_cl], [p[1] for p in moran_cl]
    
    axA.bar(x_m - width/2, pc_m, width, yerr=pc_s, label="Top-5 PCs Moran's I", color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=0.6, capsize=3)
    axA.bar(x_m + width/2, cl_m, width, yerr=cl_s, label="Cluster Fields Moran's I", color='#2ca02c', alpha=0.85, edgecolor='black', linewidth=0.6, capsize=3)
    
    axA.set_title("A  Spatial Representation Coherence on Xenium 5K (67,705 Cells)", fontweight='bold', loc='left', pad=10)
    axA.set_xticks(x_m)
    axA.set_xticklabels(clean_labels, rotation=25, ha='right', fontsize=8.0)
    axA.set_ylabel("Moran's I Autocorrelation")
    axA.set_ylim(0, 1.05)
    axA.legend(loc='upper left', frameon=True, fontsize=8.0)
    
    # --- Subplot B: Local Microenvironment Purity (R=50um) & Cross-Seed ARI ---
    axB = axes[0, 1]
    purity = [parse_mean_std(v) for v in df7['Local_Purity_R50um']]
    cross_ari = [parse_mean_std(v) for v in df7['Cross_Seed_ARI']]
    
    pur_m, pur_s = [p[0] for p in purity], [p[1] for p in purity]
    ari_m, ari_s = [p[0] for p in cross_ari], [p[1] for p in cross_ari]
    
    axB.bar(x_m - width/2, pur_m, width, yerr=pur_s, label=r"Local Purity ($R=50\,\mu\mathrm{m}$)", color='#9467bd', alpha=0.85, edgecolor='black', linewidth=0.6, capsize=3)
    axB.bar(x_m + width/2, ari_m, width, yerr=ari_s, label="Cross-Seed Clustering ARI", color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=0.6, capsize=3)
    
    axB.set_title(r"B  Local Microenvironment Purity ($R=50\,\mu\mathrm{m}$) & Clustering Stability", fontweight='bold', loc='left', pad=10)
    axB.set_xticks(x_m)
    axB.set_xticklabels(clean_labels, rotation=25, ha='right', fontsize=8.0)
    axB.set_ylabel("Metric Score")
    axB.set_ylim(0, 1.05)
    axB.legend(loc='upper left', frameon=True, fontsize=8.0)

    # --- Subplot C: Empirical Peak VRAM for Baseline Smoke Tests ---
    axC = axes[1, 0]
    scales = [1000, 5000, 10000]
    
    # Strictly extract peak VRAM for verified smoke runs
    stagate_vram = [smoke_data['results'][f'stagate@{s}']['peak_vram_mb'] for s in scales]
    cellniche_vram = [smoke_data['results'][f'cellniche@{s}']['peak_vram_mb'] for s in scales]
    
    x_s = np.arange(len(scales))
    w_s = 0.32
    
    axC.bar(x_s - w_s/2, cellniche_vram, w_s, label='CellNiche (Transductive GNN)', color='#d62728', alpha=0.85, edgecolor='black', linewidth=0.6)
    axC.bar(x_s + w_s/2, stagate_vram, w_s, label='STAGATE (Transductive GAE)', color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=0.6)
    
    # Annotations for CellNiche
    axC.text(x_s[0] - w_s/2, cellniche_vram[0] + 150, "502 MiB", ha='center', va='bottom', fontsize=7.2)
    axC.text(x_s[1] - w_s/2, cellniche_vram[1] + 150, "1,596 MiB", ha='center', va='bottom', fontsize=7.2)
    axC.text(x_s[2] - w_s/2, 8416.0 + 150, "8,416 MiB\n(PASS, 10k)", ha='center', va='bottom', fontsize=7.2, fontweight='bold', color='#a00000')
    
    # Annotations for STAGATE
    axC.text(x_s[0] + w_s/2, stagate_vram[0] + 150, "202 MiB", ha='center', va='bottom', fontsize=7.2)
    axC.text(x_s[1] + w_s/2, stagate_vram[1] + 150, "700 MiB", ha='center', va='bottom', fontsize=7.2)
    axC.text(x_s[2] + w_s/2, stagate_vram[2] + 150, "1,354 MiB", ha='center', va='bottom', fontsize=7.2, fontweight='bold')
    
    axC.set_title("C  Empirical Peak VRAM in Baseline Smoke Test Regimes", fontweight='bold', loc='left', pad=10)
    axC.set_xticks(x_s)
    axC.set_xticklabels(['1,000 Cells', '5,000 Cells', '10,000 Cells'], fontsize=8.0)
    axC.set_ylabel("Peak VRAM Allocation (MiB)")
    axC.set_ylim(0, 10500)
    axC.legend(loc='upper right', frameon=True, fontsize=8.0)
    
    # Note on exclusion in footnote

    # --- Subplot D: Effective Embedding Rank Retention Rate (Rank / Nominal Dim) ---
    axD = axes[1, 1]
    rank_parsed = [parse_mean_std(v) for v in df7['Effective_Rank']]
    rank_m = np.array([p[0] for p in rank_parsed])
    rank_s = np.array([p[1] for p in rank_parsed])
    dims = np.array([int(v) for v in df7['Dim']])
    
    # Retention rate as percentage of nominal dimension
    retention_rate = (rank_m / dims) * 100.0
    retention_err = (rank_s / dims) * 100.0
    
    colors_d = ['#1f77b4' if 'FreqNiche' in m else '#7f7f7f' for m in methods]
    bars = axD.bar(x_m, retention_rate, width=0.55, yerr=retention_err, color=colors_d, alpha=0.85, edgecolor='black', linewidth=0.6, capsize=3)
    
    for i, (rm, d, r_pct) in enumerate(zip(rank_m, dims, retention_rate)):
        axD.text(i, r_pct + 2.0, f"{rm:.1f}/{d}D\n({r_pct:.1f}%)", ha='center', va='bottom', fontsize=7.0, fontweight='bold')
        
    axD.set_title("D  Effective Rank Retention Rate (Rank / Nominal Dimension)", fontweight='bold', loc='left', pad=10)
    axD.set_xticks(x_m)
    axD.set_xticklabels(clean_labels, rotation=25, ha='right', fontsize=8.0)
    axD.set_ylabel("Rank Retention Rate (% of Nominal Dim)")
    axD.set_ylim(0, 120)
    
    # Objective statement on rank retention: capacity utilization, not downstream representation quality ranking

    metadata = {
        'task': 'Figure 5: 10x Xenium 5K Large-Scale Benchmark & Hardware Profile',
        'source_method_name': 'FreqNiche fixed 014' if args.xenium_table else 'FreqNiche-ST v1.0 Final',
        'display_method_name': 'FreqNiche',
        'freqniche_version': '014' if args.xenium_table else 'historical v1.0',
        'clarification': 'Effective rank retention rate measures latent space capacity utilization across nominal dimension budgets and does not constitute a ranking of biological representation quality.',
        'sources': {
            'xenium_table7': t7_path,
            'smoke_telemetry': smoke_path
        },
        'source_sha256': {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (t7_path, smoke_path)},
        'hardware_audit_facts': {
            'cellniche_status': '10k PASS with 8,416 MiB peak VRAM; pre-excluded from full 67k test due to steep empirical VRAM scaling to avoid OOM risk',
            'stagate_status': 'Completed full 67k evaluation',
            'freqniche_hardware_note': 'No FreqNiche 1k/5k/10k bars; historical baseline telemetry retained unchanged. No new 014 GPU peak measurement is inferred.'
        }
    }
    
    save_publication_figure(fig, "fig5_xenium_large_scale_benchmark", figures_out_dir, metadata=metadata)
    print("=" * 70)
    print("FIGURE 5 GENERATED SUCCESSFULLY WITH PROVENANCE & CLEAN LAYOUT!")
    print("=" * 70)

if __name__ == '__main__':
    main()
