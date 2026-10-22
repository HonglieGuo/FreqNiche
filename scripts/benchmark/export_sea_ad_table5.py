"""
Official Master Table 5 and Baseline Manifest Exporter for SEA-AD MTG MERFISH.

Preserves full method paradigm metadata and dynamically loads results without hardcoding:
- Raw PCA (Train+Val Fit, Inductive)
- DualFreq-[PX,H]-PCA (Train+Val Fit, Inductive)
- BANKSY-Controlled (Transductive)
- CellCharter-PCA-Controlled (Transductive)
- CellNiche-Expr-Controlled (Transductive, 5 epochs)
- FreqNiche-ST v1.0 (Inductive, 3-Seed)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(WORKSPACE_ROOT))
from src.freqniche_st.runtime_paths import resolve_baseline_output, resolve_run_output
SEA_AD_OUTPUT_DIR = resolve_run_output("outputs/runs/freqniche/sea_ad/2026-09-15/20260915_sea_ad_final")
BASELINES_DIR = resolve_baseline_output("baselines/Baselines_outputs/runs/sea_ad/2026-09-15/20260915_sea_ad_baselines")
ASSIGN_DIR = BASELINES_DIR / "assignments"
EMB_DIR = BASELINES_DIR / "embeddings"
RES_DIR = BASELINES_DIR / "results"

TABLE5_PATH = BASELINES_DIR / "table5_sea_ad_baseline_benchmark.csv"
MANIFEST_PATH = BASELINES_DIR / "sea_ad_baselines_manifest.json"
FREQNICHE_TABLE4_PATH = SEA_AD_OUTPUT_DIR / "table4_sea_ad_freqniche_results.csv"
FREQNICHE_RES_JSON_PATH = SEA_AD_OUTPUT_DIR / "test_evaluation_results.json"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_baseline_metrics(method_key: str, seeds: List[int] = (42, 123, 2024)) -> Dict[str, Any]:
    seed_files = [RES_DIR / f"{method_key}_seed_{s}_results.json" for s in seeds]
    for sf in seed_files:
        if not sf.exists():
            raise FileNotFoundError(f"Missing baseline result json: {sf}")

    metrics_list = []
    for sf in seed_files:
        with open(sf, "r", encoding="utf-8") as f:
            metrics_list.append(json.load(f))

    k_vals = [m["n_clusters"] for m in metrics_list]
    m_aris = [m["macro_ari"] for m in metrics_list]
    m_nmis = [m["macro_nmi"] for m in metrics_list]
    w_aris = [m["worst_donor_ari"] for m in metrics_list]
    asw_vals = [m["asw"] for m in metrics_list]
    r_stars = [str(m["r_star"]) for m in metrics_list]

    return {
        "clusters_k": f"{np.mean(k_vals):.1f} ± {np.std(k_vals, ddof=1):.1f}",
        "macro_ari": f"{np.mean(m_aris):.4f} ± {np.std(m_aris, ddof=1):.4f}",
        "macro_nmi": f"{np.mean(m_nmis):.4f} ± {np.std(m_nmis, ddof=1):.4f}",
        "worst_donor_ari": f"{np.mean(w_aris):.4f} ± {np.std(w_aris, ddof=1):.4f}",
        "asw": f"{np.mean(asw_vals):.4f} ± {np.std(asw_vals, ddof=1):.4f}",
        "optimal_r_star": "/".join(r_stars),
    }


def load_freqniche_metrics() -> Dict[str, Any]:
    if FREQNICHE_RES_JSON_PATH.exists():
        with open(FREQNICHE_RES_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Support both structures robustly
        if "per_seed_results" in data:
            k_vals = [m["n_clusters"] for m in data["per_seed_results"]]
            r_stars = [str(m["r_star"]) for m in data["per_seed_results"]]
            return {
                "clusters_k": f"{data['n_clusters']['mean']:.1f} ± {data['n_clusters']['std']:.1f}",
                "macro_ari": f"{data['macro_ari']['mean']:.4f} ± {data['macro_ari']['std']:.4f}",
                "macro_nmi": f"{data['macro_nmi']['mean']:.4f} ± {data['macro_nmi']['std']:.4f}",
                "worst_donor_ari": f"{data['worst_donor_ari']['mean']:.4f} ± {data['worst_donor_ari']['std']:.4f}",
                "asw": f"{data['asw']['mean']:.4f} ± {data['asw']['std']:.4f}",
                "optimal_r_star": "/".join(r_stars),
            }
        elif "seed_evaluations" in data:
            summary = data.get("3_seed_summary", {})
            k_vals = [m["n_clusters"] for m in data["seed_evaluations"].values()]
            r_stars = [str(m["frozen_r_star"]) for m in data["seed_evaluations"].values()]
            std_key = "std_ddof1" if "std_ddof1" in summary.get("macro_ari", {}) else "std"
            return {
                "clusters_k": f"{np.mean(k_vals):.1f} ± {np.std(k_vals, ddof=1):.1f}",
                "macro_ari": f"{summary['macro_ari']['mean']:.4f} ± {summary['macro_ari'][std_key]:.4f}",
                "macro_nmi": f"{summary['macro_nmi']['mean']:.4f} ± {summary['macro_nmi'][std_key]:.4f}",
                "worst_donor_ari": f"{summary['worst_donor_ari']['mean']:.4f} ± {summary['worst_donor_ari'][std_key]:.4f}",
                "asw": f"{summary['asw']['mean']:.4f} ± {summary['asw'][std_key]:.4f}",
                "optimal_r_star": "/".join(r_stars),
            }
    elif FREQNICHE_TABLE4_PATH.exists():
        df4 = pd.read_csv(FREQNICHE_TABLE4_PATH)
        seeds_df = df4[df4["Seed"].astype(str).str.isnumeric()]
        k_vals = seeds_df["Clusters_K"].astype(float).tolist()
        r_stars = seeds_df["Frozen_r_star"].astype(str).tolist()
        summary_row = df4[df4["Seed"] == "Summary_ddof1"].iloc[0]
        return {
            "clusters_k": f"{np.mean(k_vals):.1f} ± {np.std(k_vals, ddof=1):.1f}",
            "macro_ari": str(summary_row["Macro_ARI"]),
            "macro_nmi": str(summary_row["Macro_NMI"]),
            "worst_donor_ari": str(summary_row["Worst_Donor_ARI"]),
            "asw": str(summary_row["ASW"]),
            "optimal_r_star": "/".join(r_stars),
        }
    else:
        raise FileNotFoundError(f"Neither {FREQNICHE_RES_JSON_PATH} nor {FREQNICHE_TABLE4_PATH} exists.")


def export_table5() -> pd.DataFrame:
    method_configs = [
        {
            "key": "raw_pca",
            "name": "Raw PCA",
            "variant": "Raw PCA (Train+Val Fit)",
            "paradigm": "Inductive",
            "fit_scope": "Train + Validation (unseen Test)",
            "epochs": "N/A (Closed-form SVD)",
            "dim": 50,
        },
        {
            "key": "px_resdiff",
            "name": "DualFreq-[PX,H]-PCA",
            "variant": "DualFreq-[PX,H]-PCA (Train+Val Fit)",
            "paradigm": "Inductive",
            "fit_scope": "Train + Validation (unseen Test)",
            "epochs": "N/A (Closed-form SVD)",
            "dim": 50,
        },
        {
            "key": "banksy",
            "name": "BANKSY",
            "variant": "BANKSY-Controlled (Transductive)",
            "paradigm": "Transductive",
            "fit_scope": "Full Cohort (Slice-isolated graphs)",
            "epochs": "N/A (PCA)",
            "dim": 20,
        },
        {
            "key": "cellcharter",
            "name": "CellCharter",
            "variant": "CellCharter-PCA-Controlled (Transductive)",
            "paradigm": "Transductive",
            "fit_scope": "Full Cohort (Slice-isolated graphs)",
            "epochs": "N/A (Feature aggregation)",
            "dim": 200,
        },
        {
            "key": "cellniche",
            "name": "CellNiche",
            "variant": "CellNiche-Expr-Controlled (Transductive, 5 epochs)",
            "paradigm": "Transductive",
            "fit_scope": "Full Cohort (Unlabeled Test expression & graph seen)",
            "epochs": "5",
            "dim": 128,
        },
    ]

    records = []
    for cfg in method_configs:
        m = load_baseline_metrics(cfg["key"])
        records.append({
            "Method": cfg["name"],
            "Method_variant": cfg["variant"],
            "Paradigm": cfg["paradigm"],
            "Fit_data_scope": cfg["fit_scope"],
            "Training_epochs": cfg["epochs"],
            "Embedding_dim": cfg["dim"],
            "Clusters (K)": m["clusters_k"],
            "Macro ARI": m["macro_ari"],
            "Macro NMI": m["macro_nmi"],
            "Worst-Donor ARI": m["worst_donor_ari"],
            "ASW": m["asw"],
            "Optimal_r_star": m["optimal_r_star"],
        })

    fn_m = load_freqniche_metrics()
    records.append({
        "Method": "FreqNiche-ST v1.0 (Ours)",
        "Method_variant": "FreqNiche-ST v1.0 (Inductive, 3-Seed)",
        "Paradigm": "Inductive",
        "Fit_data_scope": "Train Donors only (unseen Test donors)",
        "Training_epochs": "50 (Budgeted)",
        "Embedding_dim": 128,
        "Clusters (K)": fn_m["clusters_k"],
        "Macro ARI": fn_m["macro_ari"],
        "Macro NMI": fn_m["macro_nmi"],
        "Worst-Donor ARI": fn_m["worst_donor_ari"],
        "ASW": fn_m["asw"],
        "Optimal_r_star": fn_m["optimal_r_star"],
    })

    df = pd.DataFrame(records)
    df.to_csv(TABLE5_PATH, index=False)
    return df


def export_manifest() -> Dict[str, Any]:
    manifest = {
        "benchmark_dataset": "SEA-AD MTG MERFISH (1.88M cells)",
        "cohorts": {
            "validation_centers": 10000,
            "test_centers": 10000,
            "test_donors": 5,
        },
        "evaluation_protocol": {
            "resolution_grid": [0.2, 0.4, 0.6, 0.8, 1.0],
            "cluster_range_constraint": [5, 25],
            "bootstrap_replicates": 1000,
            "variance_estimator": "sample_variance_ddof_1",
        },
        "methods": {
            "Raw PCA": {"dim": 50, "paradigm": "Inductive"},
            "DualFreq-[PX,H]-PCA": {"dim": 50, "paradigm": "Inductive"},
            "BANKSY": {"dim": 20, "paradigm": "Transductive"},
            "CellCharter": {"dim": 200, "paradigm": "Transductive"},
            "CellNiche": {"dim": 128, "paradigm": "Transductive"},
            "FreqNiche-ST v1.0": {"dim": 128, "paradigm": "Inductive"},
        },
        "files": {},
    }

    manifest["files"]["table5_sea_ad_baseline_benchmark.csv"] = {
        "rel_path": "table5_sea_ad_baseline_benchmark.csv",
        "sha256": compute_sha256(TABLE5_PATH),
        "size_bytes": TABLE5_PATH.stat().st_size,
    }

    for f in sorted(ASSIGN_DIR.glob("*.csv")):
        manifest["files"][f"assignments/{f.name}"] = {
            "rel_path": f"assignments/{f.name}",
            "sha256": compute_sha256(f),
            "size_bytes": f.stat().st_size,
            "rows": sum(1 for _ in open(f, "r", encoding="utf-8")) - 1,
        }

    for f in sorted(EMB_DIR.glob("*.npy")):
        emb = np.load(f)
        manifest["files"][f"embeddings/{f.name}"] = {
            "rel_path": f"embeddings/{f.name}",
            "sha256": compute_sha256(f),
            "size_bytes": f.stat().st_size,
            "shape": list(emb.shape),
            "is_finite": bool(np.isfinite(emb).all()),
        }

    for f in sorted(RES_DIR.glob("*.json")):
        manifest["files"][f"results/{f.name}"] = {
            "rel_path": f"results/{f.name}",
            "sha256": compute_sha256(f),
            "size_bytes": f.stat().st_size,
        }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    print(f"[{Path(__file__).name}] Exporting Table 5 ...")
    df = export_table5()
    print(f"[OK] Table 5 exported to: {TABLE5_PATH}")
    print(f"[{Path(__file__).name}] Exporting Manifest ...")
    m = export_manifest()
    print(f"[OK] Manifest ({len(m['files'])} files) exported to: {MANIFEST_PATH}")
    print("=" * 110)
    print(df.to_string(index=False))
    print("=" * 110)


if __name__ == "__main__":
    main()
