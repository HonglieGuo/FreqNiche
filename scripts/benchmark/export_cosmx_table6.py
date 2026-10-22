"""
Master Table 6 and Baseline Manifest Exporter for CosMx Human NSCLC (W2.6).

Aggregates:
- FreqNiche-ST v1.0 Final (Inductive, train_only, 3-Seed)
- Raw PCA (Inductive, train_val_unsupervised, 3-Seed)
- DualFreq (px_resdiff) (Inductive, train_val_unsupervised, 3-Seed)
- BANKSY-Controlled (Transductive, full_cohort_unsupervised, 3-Seed)
- CellCharter-PCA-Controlled (Transductive, full_cohort_unsupervised, 3-Seed)
- CellNiche-Controlled (Transductive, full_cohort_unsupervised, 3-Seed)
- (Optional) STAGATE-Controlled

Strict Invariant:
- By default, FAILS CLOSED if ANY of the 5 core baseline methods is missing, preventing incomplete master tables.
- Requires --allow-partial to export intermediate/partial results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(WORKSPACE_ROOT))
from src.freqniche_st.runtime_paths import resolve_baseline_output, resolve_run_output
COSMX_OUTPUT_DIR = resolve_run_output("outputs/runs/freqniche/cosmx_nsclc/2026-09-17/20260917_cosmx_nsclc_final")
BASELINES_DIR = resolve_baseline_output("baselines/Baselines_outputs/runs/cosmx_nsclc/2026-09-17/20260917_cosmx_nsclc_baselines")
RES_DIR = BASELINES_DIR / "results"
ASSIGN_DIR = BASELINES_DIR / "assignments"
EMB_DIR = BASELINES_DIR / "embeddings"

TABLE6_PATH = COSMX_OUTPUT_DIR / "table6_cosmx_full_benchmark.csv"
MANIFEST_PATH = BASELINES_DIR / "cosmx_baselines_manifest.json"
FREQNICHE_REPORT_PATH = COSMX_OUTPUT_DIR / "cosmx_test_evaluation_report.json"

CORE_METHODS = ["raw_pca", "px_resdiff", "banksy", "cellcharter", "cellniche"]


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_freqniche_row() -> Dict[str, Any]:
    if not FREQNICHE_REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing FreqNiche report: {FREQNICHE_REPORT_PATH}")

    with open(FREQNICHE_REPORT_PATH, "r", encoding="utf-8") as f:
        rep = json.load(f)

    r_stars = [str(rep["per_seed_results"][s]["r_star"]) for s in ["42", "123", "2024"]]
    return {
        "Method_variant": "FreqNiche-ST v1.0 Final",
        "Paradigm": "Inductive",
        "Fit_data_scope": "train_only",
        "Target_Donor": "Lung9",
        "K_Clusters": f"{rep['mean_clusters']:.1f} ± {rep['std_clusters']:.1f}",
        "ARI": f"{rep['mean_ari']:.4f} ± {rep['std_ari']:.4f}",
        "NMI": f"{rep['mean_nmi']:.4f} ± {rep['std_nmi']:.4f}",
        "Reference-Niche ASW": f"{rep['mean_asw']:.4f} ± {rep['std_asw']:.4f}",
        "Locked_r_stars": "/".join(r_stars)
    }


def load_baseline_row(method_name: str) -> Dict[str, Any] | None:
    summary_file = RES_DIR / f"{method_name}_benchmark_summary.json"
    if not summary_file.exists():
        return None

    with open(summary_file, "r", encoding="utf-8") as f:
        d = json.load(f)

    r_stars = [str(d["locked_r_stars"][str(s)]) for s in [42, 123, 2024]]
    return {
        "Method_variant": d["method_variant"],
        "Paradigm": d["paradigm"],
        "Fit_data_scope": d["fit_data_scope"],
        "Target_Donor": "Lung9",
        "K_Clusters": f"{d['mean_clusters']:.1f} ± {d['std_clusters']:.1f}",
        "ARI": f"{d['mean_ari']:.4f} ± {d['std_ari']:.4f}",
        "NMI": f"{d['mean_nmi']:.4f} ± {d['std_nmi']:.4f}",
        "Reference-Niche ASW": f"{d['mean_reference_niche_asw']:.4f} ± {d['std_reference_niche_asw']:.4f}",
        "Locked_r_stars": "/".join(r_stars)
    }


def export_table6(allow_partial: bool = False):
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    # 1. FreqNiche-ST
    rows.append(load_freqniche_row())

    # 2. Check completeness of core methods
    missing_methods = []
    for m in CORE_METHODS:
        summary_file = RES_DIR / f"{m}_benchmark_summary.json"
        if not summary_file.exists():
            missing_methods.append(m)

    if missing_methods and not allow_partial:
        raise RuntimeError(
            f"Cannot export master Table 6: Missing results for core methods {missing_methods}! "
            "Execution refused by default (Fail Closed). Specify --allow-partial to export incomplete table."
        )

    # 3. Baselines in standard order
    all_methods = CORE_METHODS + ["stagate"]
    manifest_assets = {}

    for m in all_methods:
        row = load_baseline_row(m)
        if row is not None:
            rows.append(row)
            summary_file = RES_DIR / f"{m}_benchmark_summary.json"
            manifest_assets[m] = {
                "summary_sha256": compute_sha256(summary_file),
                "assignments": {
                    s: compute_sha256(ASSIGN_DIR / f"{m}_test_assignments_seed_{s}.csv")
                    for s in [42, 123, 2024]
                    if (ASSIGN_DIR / f"{m}_test_assignments_seed_{s}.csv").exists()
                },
                "embeddings": {
                    f"{split}_seed_{s}": compute_sha256(EMB_DIR / f"{m}_{split}_seed_{s}.npy")
                    for split in ["val", "test"]
                    for s in [42, 123, 2024]
                    if (EMB_DIR / f"{m}_{split}_seed_{s}.npy").exists()
                }
            }

    df_table6 = pd.DataFrame(rows)
    tmp_table6 = TABLE6_PATH.with_suffix(".tmp.csv")
    df_table6.to_csv(tmp_table6, index=False)
    os.replace(tmp_table6, TABLE6_PATH)

    # Save manifest
    manifest_data = {
        "dataset": "cosmx_human_nsclc",
        "description": "W2.6 CosMx Baseline Benchmark Master Manifest",
        "table6_sha256": compute_sha256(TABLE6_PATH),
        "is_partial": bool(missing_methods),
        "methods_recorded": list(manifest_assets.keys()),
        "assets": manifest_assets
    }
    tmp_manifest = MANIFEST_PATH.with_suffix(".tmp.json")
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    os.replace(tmp_manifest, MANIFEST_PATH)

    print("\n" + "=" * 90)
    print(f"MASTER TABLE 6 (CosMx NSCLC Benchmark | Partial={bool(missing_methods)}):")
    try:
        print(df_table6.to_markdown(index=False))
    except Exception:
        print(df_table6.to_string(index=False))
    print("=" * 90)
    print(f"Exported to: {TABLE6_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Export Table 6 for CosMx NSCLC Benchmark")
    parser.add_argument("--allow-partial", action="store_true", help="Allow exporting partial table if some methods are missing")
    args = parser.parse_args()
    export_table6(allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
