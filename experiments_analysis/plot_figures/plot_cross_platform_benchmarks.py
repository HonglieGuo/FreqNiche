"""
experiments_analysis/plot_figures/plot_cross_platform_benchmarks.py

Figure 4: Cross-Platform Spatial Niche Benchmarks across Human Skin, SEA-AD, and CosMx
- Human Skin MERFISH (W2.4 Primary Full Cohort Table 1)
- SEA-AD MTG MERFISH (W2.5 5 Cortical Layers: L1, L2/3, L4, L5, L6, Table 5)
- CosMx NSCLC (W2.6 Cross-Platform Benchmark Table 6)
Strictly reads from archived CSV assets with full provenance and automated report generation.
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
    print("Plotting Cross-Platform Benchmarks (Human Skin, SEA-AD, CosMx)...")
    print("=" * 70)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sea-ad-table', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    data_dir = str(Path(__file__).resolve().parents[2])
    figures_out_dir = str(args.output_dir or Path(data_dir) / "Paper/FreqNiche-zh/figures")
    
    # 1. Human Skin Table 1
    skin_t1_path = f"{data_dir}/baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final/table1_full_cohort_primary.csv"
    skin_df = pd.read_csv(skin_t1_path)
    
    # 2. SEA-AD Table 5
    sea_t5_path = str(args.sea_ad_table or Path(data_dir) / "baselines/Baselines_outputs/runs/sea_ad/2026-09-15/20260915_sea_ad_baselines/table5_sea_ad_baseline_benchmark.csv")
    sea_df = pd.read_csv(sea_t5_path)
    
    # 3. CosMx Table 6
    cosmx_t6_path = f"{data_dir}/outputs/runs/freqniche/cosmx_nsclc/2026-09-17/20260917_cosmx_nsclc_final/table6_cosmx_full_benchmark.csv"
    cosmx_df = pd.read_csv(cosmx_t6_path)
    
    # --- Aggregation for Human Skin Table 1 ---
    # Primary evaluation: Test_Donor_Macro_ARI, Test_Donor_Macro_NMI, Worst_Donor_ARI
    skin_agg = skin_df.groupby('Method').agg({
        'Test_Donor_Macro_ARI': ['mean', 'std'],
        'Test_Donor_Macro_NMI': ['mean', 'std'],
        'Worst_Donor_ARI': ['min', 'mean', 'std']
    })
    skin_agg.columns = ['ari_mean', 'ari_std', 'nmi_mean', 'nmi_std', 'worst_ari_min', 'worst_ari_mean', 'worst_ari_std']
    skin_agg = skin_agg.sort_values('ari_mean', ascending=False)
    
    skin_methods = skin_agg.index.tolist()
    s_ari_mean = skin_agg['ari_mean'].values
    s_ari_std = np.nan_to_num(skin_agg['ari_std'].values)
    s_nmi_mean = skin_agg['nmi_mean'].values
    s_nmi_std = np.nan_to_num(skin_agg['nmi_std'].values)
    worst_ari = skin_agg['worst_ari_min'].values
    
    # Print automated verification report for Human Skin
    print("\n" + "=" * 70)
    print("FIGURE 4A/B AUTOMATED REPORT (HUMAN SKIN TABLE 1):")
    print("=" * 70)
    print("| Method | Macro ARI (Mean ± Std) | Macro NMI (Mean ± Std) | Min Worst-Donor ARI |")
    print("|---|---|---|---|")
    for m in skin_methods:
        row = skin_agg.loc[m]
        print(f"| {m} | {row['ari_mean']:.4f} ± {row['ari_std']:.4f} | {row['nmi_mean']:.4f} ± {row['nmi_std']:.4f} | {row['worst_ari_min']:.4f} |")
    print("=" * 70 + "\n")
    
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.5))
    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.08, right=0.96, hspace=0.36, wspace=0.26)
    
    width = 0.36
    
    # --- Subplot A: Human Skin Macro ARI & NMI ---
    axA = axes[0, 0]
    x_s = np.arange(len(skin_methods))
    axA.bar(x_s - width/2, s_ari_mean, width, yerr=s_ari_std, label='Macro ARI', color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=0.6, capsize=3)
    axA.bar(x_s + width/2, s_nmi_mean, width, yerr=s_nmi_std, label='Macro NMI', color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=0.6, capsize=3)
    
    axA.set_title("A  Human Skin MERFISH Benchmark (Full Cohort, N=3 Seeds)", fontweight='bold', loc='left', pad=10)
    axA.set_xticks(x_s)
    clean_skin_labels = [m.replace('-Controlled', '').replace('-Canonical', '').replace(' v1.0 Final', '').replace('-Multisample', '').replace('FreqNiche-ST', 'FreqNiche').replace('CellCharter-PCA', 'CellCharter').replace('CellCharter-Canonical', 'CellCharter') for m in skin_methods]
    axA.set_xticklabels(clean_skin_labels, rotation=25, ha='right', fontsize=8.0)
    axA.set_ylabel("Score (Mean ± Std)")
    axA.set_ylim(0, 0.70)
    axA.legend(loc='upper right', frameon=True, fontsize=8.0)
    
    # --- Subplot B: Human Skin Worst-Donor ARI ---
    axB = axes[0, 1]
    colors_b = [PALETTE_METHODS.get(m, '#7f7f7f') for m in skin_methods]
    axB.bar(x_s, worst_ari, width=0.55, color=colors_b, alpha=0.85, edgecolor='black', linewidth=0.6)
    axB.set_title("B  Human Skin Worst-Case Generalization (Min ARI across 3 Seeds x 4 Donors)", fontweight='bold', loc='left', pad=10)
    axB.set_xticks(x_s)
    axB.set_xticklabels(clean_skin_labels, rotation=25, ha='right', fontsize=8.0)
    axB.set_ylabel("Min Worst-Donor ARI")
    axB.set_ylim(0, max(worst_ari) * 1.35)
    for i, v in enumerate(worst_ari):
        axB.text(i, v + 0.010, f"{v:.3f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    axB.text(0.04, 0.88, "Min ARI = min_{seed, donor} ARI\n(Strictest unseen donor stress test)", 
             transform=axB.transAxes, fontsize=7.2, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", lw=0.5))
        
    # --- Subplot C: SEA-AD Cortical Layers (L1, L2/3, L4, L5, L6) ---
    axC = axes[1, 0]
    sea_methods = sea_df['Method'].tolist()
    sea_ari_parsed = [parse_mean_std(v) for v in sea_df['Macro ARI']]
    sea_nmi_parsed = [parse_mean_std(v) for v in sea_df['Macro NMI']]
    
    sea_ari_mean = [p[0] for p in sea_ari_parsed]
    sea_ari_std = [p[1] for p in sea_ari_parsed]
    sea_nmi_mean = [p[0] for p in sea_nmi_parsed]
    sea_nmi_std = [p[1] for p in sea_nmi_parsed]
    
    x_sea = np.arange(len(sea_methods))
    axC.bar(x_sea - width/2, sea_ari_mean, width, yerr=sea_ari_std, label='Macro ARI', color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=0.6, capsize=3)
    axC.bar(x_sea + width/2, sea_nmi_mean, width, yerr=sea_nmi_std, label='Macro NMI', color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=0.6, capsize=3)
    
    axC.set_title("C  SEA-AD MTG MERFISH (5 Layers: L1, L2/3, L4, L5, L6)", fontweight='bold', loc='left', pad=10)
    axC.set_xticks(x_sea)
    clean_sea_labels = []
    for m in sea_methods:
        lbl = m.replace(' v1.0 (Ours)', '').replace('-[PX,H]-PCA', '').replace('FreqNiche-ST', 'FreqNiche').replace('CellCharter-PCA', 'CellCharter').replace('CellCharter-Canonical', 'CellCharter')
        if 'CellNiche' in m:
            lbl += ' (Transd.)*'
        clean_sea_labels.append(lbl)
    axC.set_xticklabels(clean_sea_labels, rotation=25, ha='right', fontsize=8.0)
    axC.set_ylabel("Layer Concordance Score")
    axC.set_ylim(0, 0.72)
    axC.legend(loc='upper left', frameon=True, fontsize=8.0)
    

    # --- Subplot D: CosMx NSCLC Benchmark (Table 6) ---
    axD = axes[1, 1]
    cosmx_methods = cosmx_df['Method_variant'].tolist()
    cosmx_ari_parsed = [parse_mean_std(v) for v in cosmx_df['ARI']]
    cosmx_nmi_parsed = [parse_mean_std(v) for v in cosmx_df['NMI']]
    
    cosmx_ari_mean = [p[0] for p in cosmx_ari_parsed]
    cosmx_ari_std = [p[1] for p in cosmx_ari_parsed]
    cosmx_nmi_mean = [p[0] for p in cosmx_nmi_parsed]
    cosmx_nmi_std = [p[1] for p in cosmx_nmi_parsed]
    
    x_cos = np.arange(len(cosmx_methods))
    axD.bar(x_cos - width/2, cosmx_ari_mean, width, yerr=cosmx_ari_std, label='Macro ARI', color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=0.6, capsize=3)
    axD.bar(x_cos + width/2, cosmx_nmi_mean, width, yerr=cosmx_nmi_std, label='Macro NMI', color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=0.6, capsize=3)
    
    axD.set_title("D  CosMx Human NSCLC Subcellular Benchmark", fontweight='bold', loc='left', pad=10)
    axD.set_xticks(x_cos)
    clean_cosmx_labels = [m.replace('-Controlled', '').replace(' Final', '').replace(' (px_resdiff)', '').replace('FreqNiche-ST', 'FreqNiche').replace('CellCharter-PCA', 'CellCharter').replace('CellCharter-Canonical', 'CellCharter') for m in cosmx_methods]
    axD.set_xticklabels(clean_cosmx_labels, rotation=25, ha='right', fontsize=8.0)
    axD.set_ylabel("Score (Mean ± Std)")
    axD.set_ylim(0, 0.55)
    axD.legend(loc='upper right', frameon=True, fontsize=8.0)

    metadata = {
        'task': 'Figure 4: Cross-Platform Spatial Niche Benchmarks',
        'source_method_name': 'Mixed dataset-specific versions; see freqniche_versions' if args.sea_ad_table else 'FreqNiche-ST v1.0 Final',
        'display_method_name': 'FreqNiche',
        'freqniche_versions': {'human_skin': 'historical v1.0', 'cosmx': 'historical v1.0', 'sea_ad': '014' if args.sea_ad_table else 'historical v1.0'},
        'sources': {
            'human_skin_table1': skin_t1_path,
            'sea_ad_table5': sea_t5_path,
            'cosmx_table6': cosmx_t6_path
        },
        'source_sha256': {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (skin_t1_path, sea_t5_path, cosmx_t6_path)},
        'sea_ad_layers': ['L1', 'L2/3', 'L4', 'L5', 'L6'],
        'methods_included': {
            'human_skin': skin_methods,
            'sea_ad': sea_methods,
            'cosmx': cosmx_methods
        },
        'extracted_values': {
            'freqniche_skin_ari': f"{skin_agg.loc['FreqNiche-ST v1.0 Final', 'ari_mean']:.4f} ± {skin_agg.loc['FreqNiche-ST v1.0 Final', 'ari_std']:.4f}",
            'freqniche_skin_nmi': f"{skin_agg.loc['FreqNiche-ST v1.0 Final', 'nmi_mean']:.4f} ± {skin_agg.loc['FreqNiche-ST v1.0 Final', 'nmi_std']:.4f}",
            'freqniche_skin_min_worst_ari': f"{skin_agg.loc['FreqNiche-ST v1.0 Final', 'worst_ari_min']:.4f}"
        }
    }
    
    save_publication_figure(fig, "fig4_cross_platform_benchmarks", figures_out_dir, metadata=metadata)
    print("=" * 70)
    print("FIGURE 4 GENERATED SUCCESSFULLY WITH PROVENANCE & CLEAN LAYOUT!")
    print("=" * 70)

if __name__ == '__main__':
    main()
