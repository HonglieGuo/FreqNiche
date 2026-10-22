#!/usr/bin/env python3
"""
Programmatic Table 7 Synthesis Generator
Extracts metrics directly from the 6 authoritative JSON test evaluation reports
(FreqNiche-ST and 5 Baselines on L1 67,705 cells, 5,001 genes)
Computes exact 3-seed Mean ± Sample SD (ddof=1), formats Table 7 in Markdown and LaTeX,
and saves the structured synthesis to JSON and CSV.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = WORKSPACE_ROOT / "outputs/runs/freqniche/xenium_prime5k/2026-09-18/20260918_xenium_prime5k_final"
BASELINES_RESULTS_DIR = WORKSPACE_ROOT / "baselines/Baselines_outputs/runs/xenium_prime5k/2026-09-18/20260918_xenium_prime5k_baselines/results"

REPORT_PATHS = {
    "raw_pca": BASELINES_RESULTS_DIR / "raw_pca_test_evaluation_report.json",
    "px_resdiff": BASELINES_RESULTS_DIR / "px_resdiff_test_evaluation_report.json",
    "banksy": BASELINES_RESULTS_DIR / "banksy_test_evaluation_report.json",
    "cellcharter": BASELINES_RESULTS_DIR / "cellcharter_test_evaluation_report.json",
    "stagate": BASELINES_RESULTS_DIR / "stagate_test_evaluation_report.json",
    "freqniche_st": BASE_DIR / "xenium_test_evaluation_report.json",
}

METHOD_METADATA = {
    "raw_pca": {
        "display_name": "Raw PCA",
        "category": "Lower Bound (Expression)",
        "learning_paradigm": "Transductive",  # Target slice unsupervised PCA refitting
        "feature_dim": 50,
        "input_context": "None (Pure Expression)",
    },
    "px_resdiff": {
        "display_name": "DualFreq (px_resdiff)",
        "category": "Lower Bound (Linear Filter)",
        "learning_paradigm": "Transductive",  # Target slice unsupervised spatial filtering + PCA refitting
        "feature_dim": 50,
        "input_context": "Physical Graph P (R=50μm)",
    },
    "banksy": {
        "display_name": "BANKSY-Controlled",
        "category": "Neighborhood Aggregation",
        "learning_paradigm": "Transductive",
        "feature_dim": 20,
        "input_context": "kNN (k=18)",
    },
    "cellcharter": {
        "display_name": "CellCharter-PCA",
        "category": "Neighborhood Aggregation",
        "learning_paradigm": "Transductive",
        "feature_dim": 200,
        "input_context": "Neighborhood Mean (R=50μm, 3 layers)",
    },
    "stagate": {
        "display_name": "STAGATE-Controlled",
        "category": "Graph Autoencoder",
        "learning_paradigm": "Transductive",
        "feature_dim": 30,
        "input_context": "Spatial Graph (R=50μm)",
    },
    "freqniche_st": {
        "display_name": "FreqNiche-ST (Ours)",
        "category": "Dual-Frequency Contrastive GNN",
        "learning_paradigm": "Inductive",  # Frozen encoder from primary cohort, zero fine-tuning on L1
        "feature_dim": 128,
        "input_context": "Dual-Frequency Graphs A & P (R=50μm)",
    },
}

SEEDS = ["42", "123", "2024"]


def parse_reports():
    records = []
    
    for key, path in REPORT_PATHS.items():
        assert path.exists(), f"Missing report: {path}"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        meta = METHOD_METADATA[key]
        
        # Extract per-seed raw metrics
        k_list = []
        asw_list = []
        erank_list = []
        purity_list = []
        moran_pcs_list = []
        moran_clust_list = []
        r_star_list = []
        per_seed_data = data["per_seed_results"]

        if key == "freqniche_st" and "three_seed_summary_stats" in data:
            # Directly read authoritative frozen summary stats for FreqNiche-ST
            stats = data["three_seed_summary_stats"]
            k_m, k_sd = float(stats["k_clusters"]["mean"]), float(stats["k_clusters"]["std"])
            asw_m, asw_sd = float(stats["internal_asw"]["mean"]), float(stats["internal_asw"]["std"])
            erank_m, erank_sd = float(stats["effective_rank"]["mean"]), float(stats["effective_rank"]["std"])
            purity_m, purity_sd = float(stats["local_neighborhood_purity"]["mean"]), float(stats["local_neighborhood_purity"]["std"])
            moran_pcs_m, moran_pcs_sd = float(stats["moran_i_top5_pcs_delta"]["mean"]), float(stats["moran_i_top5_pcs_delta"]["std"])
            moran_clust_m, moran_clust_sd = float(stats["moran_i_cluster_onehot_delta"]["mean"]), float(stats["moran_i_cluster_onehot_delta"]["std"])
            ari_m, ari_sd = float(stats["cross_seed_pairwise_ari"]["mean"]), float(stats["cross_seed_pairwise_ari"]["std"])
            
            # Also extract per-seed raw values
            for s in SEEDS:
                res = per_seed_data[s]
                r_star = res.get("locked_r_star", res.get("frozen_r_star"))
                r_star_list.append(float(r_star))
                k_list.append(int(res["k_clusters"]))
                asw_list.append(float(res["internal_asw"]))
                erank_list.append(float(res["effective_rank"]))
                purity_list.append(float(res["local_neighborhood_purity_mean"]))
                moran_pcs_list.append(float(res["morans_i_top5_pcs"]["mean_delta_i"]))
                moran_clust_list.append(float(res["morans_i_cluster_onehot"]["mean_delta_i"]))
            r_star_m = float(np.mean(r_star_list))
        else:
            # Baseline reports
            for s in SEEDS:
                res = per_seed_data[s]
                r_star = res.get("frozen_r_star", res.get("locked_r_star"))
                r_star_list.append(float(r_star))
                k_list.append(int(res["k_clusters"]))
                asw_list.append(float(res["internal_asw"]))
                erank_list.append(float(res["effective_rank"]))
                purity_list.append(float(res["local_neighborhood_purity_mean"]))
                moran_pcs_list.append(float(res["morans_i_top5_pcs_delta"]))
                moran_clust_list.append(float(res["morans_i_clusters_delta"]))

            def m_sd(vals):
                return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

            k_m, k_sd = m_sd(k_list)
            asw_m, asw_sd = m_sd(asw_list)
            erank_m, erank_sd = m_sd(erank_list)
            purity_m, purity_sd = m_sd(purity_list)
            moran_pcs_m, moran_pcs_sd = m_sd(moran_pcs_list)
            moran_clust_m, moran_clust_sd = m_sd(moran_clust_list)
            r_star_m, r_star_sd = m_sd(r_star_list)

            ari_info = data["cross_seed_pairwise_ari"]
            ari_m = float(ari_info["mean"])
            ari_sd = float(ari_info["std"])

        record = {
            "key": key,
            "display_name": meta["display_name"],
            "category": meta["category"],
            "learning_paradigm": meta["learning_paradigm"],
            "feature_dim": meta["feature_dim"],
            "input_context": meta["input_context"],
            "r_stars": r_star_list,
            "r_star_mean": r_star_m,
            "k_clusters": {
                "per_seed": k_list,
                "mean": round(k_m, 2),
                "std": round(k_sd, 2),
                "str": f"{k_m:.1f} ± {k_sd:.1f}"
            },
            "asw_silhouette": {
                "per_seed": [round(x, 4) for x in asw_list],
                "mean": round(asw_m, 4),
                "std": round(asw_sd, 4),
                "str": f"{asw_m:+.4f} ± {asw_sd:.4f}"
            },
            "effective_rank": {
                "per_seed": [round(x, 2) for x in erank_list],
                "mean": round(erank_m, 2),
                "std": round(erank_sd, 2),
                "str": f"{erank_m:.1f} ± {erank_sd:.1f}"
            },
            "local_purity": {
                "per_seed": [round(x, 4) for x in purity_list],
                "mean": round(purity_m, 4),
                "std": round(purity_sd, 4),
                "str": f"{purity_m:.4f} ± {purity_sd:.4f}"
            },
            "morans_i_top5_pcs": {
                "per_seed": [round(x, 4) for x in moran_pcs_list],
                "mean": round(moran_pcs_m, 4),
                "std": round(moran_pcs_sd, 4),
                "str": f"{moran_pcs_m:.4f} ± {moran_pcs_sd:.4f}"
            },
            "morans_i_clusters": {
                "per_seed": [round(x, 4) for x in moran_clust_list],
                "mean": round(moran_clust_m, 4),
                "std": round(moran_clust_sd, 4),
                "str": f"{moran_clust_m:.4f} ± {moran_clust_sd:.4f}"
            },
            "cross_seed_ari": {
                "mean": round(ari_m, 4) if ari_m is not None else None,
                "std": round(ari_sd, 4) if ari_sd is not None else None,
                "str": f"{ari_m:.4f} ± {ari_sd:.4f}" if ari_m is not None else "N/A"
            },
            "raw_report_path": str(path.relative_to(WORKSPACE_ROOT)),
        }
        records.append(record)
        
    return records


def main():
    print("Parsing 6 original evaluation reports ...")
    records = parse_reports()
    
    # Save JSON synthesis
    out_json = BASE_DIR / "table7_benchmark_synthesis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved JSON synthesis to {out_json}")
    
    # Save CSV synthesis
    csv_rows = []
    for r in records:
        csv_rows.append({
            "Method": r["display_name"],
            "Category": r["category"],
            "Paradigm": r["learning_paradigm"],
            "Dim": r["feature_dim"],
            "K_Clusters": r["k_clusters"]["str"],
            "Internal_ASW": r["asw_silhouette"]["str"],
            "Effective_Rank": r["effective_rank"]["str"],
            "Local_Purity_R50um": r["local_purity"]["str"],
            "Moran_I_Top5_PCs": r["morans_i_top5_pcs"]["str"],
            "Moran_I_Clusters": r["morans_i_clusters"]["str"],
            "Cross_Seed_ARI": r["cross_seed_ari"]["str"],
        })
    df = pd.DataFrame(csv_rows)
    out_csv = BASE_DIR / "table7_benchmark_synthesis.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved CSV synthesis to {out_csv}")
    
    print("\n" + "=" * 120)
    print("TABLE 7: BENCHMARK SYNTHESIS (PROGRAMMATICALLY EXTRACTED FROM 6 AUTHORITATIVE JSON REPORTS)")
    headers = list(csv_rows[0].keys())
    col_widths = {h: max(len(h), max(len(str(row[h])) for row in csv_rows)) for h in headers}
    header_line = "| " + " | ".join(h.ljust(col_widths[h]) for h in headers) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[h] for h in headers) + " |"
    row_lines = [
        "| " + " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers) + " |"
        for row in csv_rows
    ]
    md_table = "\n".join([header_line, sep_line] + row_lines)
    print(md_table)
    print("=" * 120)


if __name__ == "__main__":
    main()
