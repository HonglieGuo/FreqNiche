"""
scripts/benchmark/export_full_slice_summary.py

Exports Table:
1. baselines/Baselines_outputs/runs/human_skin/2026-09-21/20260921_human_skin_full_slice_comparison/table_supp_full_slice_domain_summary.csv
2. Paper/FreqNiche-zh/tables/table_supp_full_slice_domain_summary.tex
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "baselines/Baselines_outputs/runs/human_skin/2026-09-21/20260921_human_skin_full_slice_comparison"
READOUT_META_PATH = BASE_DIR / "readouts/resolution_selection.json"
FULL_EXEC_SUMMARY = BASE_DIR / "full_execution_summary.json"

CSV_OUT = BASE_DIR / "table_supp_full_slice_domain_summary.csv"
TEX_OUT = ROOT / "Paper/FreqNiche-zh/tables/table_supp_full_slice_domain_summary.tex"
TEX_OUT.parent.mkdir(parents=True, exist_ok=True)

METHOD_DISPLAY_NAMES = {
    "gt": ("Ground Truth niche", "Reference", "N/A", "N/A"),
    "freqniche": ("FreqNiche", "Inductive", 128, "Frozen"),
    "cellcharter": ("CellCharter-Canonical", "Transductive", 20, "Slice-native"),
    "banksy": ("BANKSY-Controlled", "Transductive", 20, "Deterministic"),
    "stagate": ("STAGATE-Controlled", "Transductive", 30, "Slice-native"),
    "cellniche": ("CellNiche-Controlled", "Transductive", 128, "Slice-native"),
}


def main():
    print("=" * 80)
    print(">>> Exporting Full Slice Domain Summary Table (CSV & LaTeX)")
    print("=" * 80)

    if not READOUT_META_PATH.exists():
        print(f"[WARNING] Missing readout metadata: {READOUT_META_PATH}. Waiting for execution.")
        return

    with open(READOUT_META_PATH, "r", encoding="utf-8") as fp:
        readout_meta = json.load(fp)

    exec_meta = {}
    if FULL_EXEC_SUMMARY.exists():
        with open(FULL_EXEC_SUMMARY, "r", encoding="utf-8") as fp:
            exec_meta = json.load(fp)

    rows = []
    # 1. Ground Truth
    rows.append({
        "Method": "Ground Truth niche",
        "Paradigm": "Reference",
        "Embedding Dim": "N/A",
        "Execution": "Official Annotations",
        "Slice r*": "N/A",
        "Final K": 10,  # D028 region_1 GT active niches count
        "Stability (ARI)": 1.000,
        "Post-hoc ARI": 1.000,
        "Post-hoc NMI": 1.000,
        "Runtime (s)": "N/A",
        "Peak VRAM (MB)": "N/A",
    })

    # 2. Methods
    for key, (display_name, paradigm, dim, exec_type) in METHOD_DISPLAY_NAMES.items():
        if key == "gt":
            continue
        if key in readout_meta:
            rm = readout_meta[key]
            em = exec_meta.get(key, {})
            rows.append({
                "Method": display_name,
                "Paradigm": paradigm,
                "Embedding Dim": dim,
                "Execution": exec_type,
                "Slice r*": rm.get("r_star_slice", "N/A"),
                "Final K": rm.get("final_k", "N/A"),
                "Stability (ARI)": rm.get("stability_ari", "N/A"),
                "Post-hoc ARI": rm.get("post_hoc_ari", "N/A"),
                "Post-hoc NMI": rm.get("post_hoc_nmi", "N/A"),
                "Runtime (s)": em.get("wall_clock_seconds", "N/A"),
                "Peak VRAM (MB)": em.get("peak_vram_mb", "N/A"),
            })

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)
    print(f"[OK] Saved CSV summary table: {CSV_OUT}")

    # Generate LaTeX Table
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Summary of native full-slice spatial-domain identification across all methods on the Human Skin physical section D028 \texttt{region\_1} (11,293 cells). All methods produce embeddings for exactly the same 11,293 cells and use the same unlabeled stability-based resolution selection protocol: Leiden candidate grid $[0.05, 1.00]$, $k\text{NN}=15$, $10\times 80\%$ subsampling, and a target mean cluster-count constraint of $5 \le K \le 25$. Post-hoc ARI and NMI are descriptive metrics computed after parameters are frozen; they are never used for model fitting or resolution selection. The final full-slice cluster count $K$ is reported regardless of whether it falls within the target range.}",
        r"\label{tab:tab_supp_full_slice_domain_summary}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Paradigm} & \textbf{Dim} & $r^*_{\text{slice}}$ & \textbf{Final } $K$ & \textbf{Stability (ARI)} & \textbf{Post-hoc ARI} & \textbf{Post-hoc NMI} & \textbf{Runtime (s)} \\",
        r"\midrule",
    ]

    for _, r in df.iterrows():
        row_str = f"{r['Method']} & {r['Paradigm']} & {r['Embedding Dim']} & {r['Slice r*']} & {r['Final K']} & {r['Stability (ARI)']} & {r['Post-hoc ARI']} & {r['Post-hoc NMI']} & {r['Runtime (s)']} \\\\"
        tex_lines.append(row_str)

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(TEX_OUT, "w", encoding="utf-8") as fp:
        fp.write("\n".join(tex_lines) + "\n")
    print(f"[OK] Saved LaTeX summary table: {TEX_OUT}")


if __name__ == "__main__":
    main()
