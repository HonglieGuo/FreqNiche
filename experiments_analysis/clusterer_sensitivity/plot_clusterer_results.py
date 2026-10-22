"""Render append-only figures from the verified combined result table with atomic publishing and provenance signing."""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import clusterer_common as common
from merge_clusterer_results import COMBINED_DIR


ORDER = ["kmeans", "mclust", "louvain", "leiden"]
DISPLAY_LABELS = ["K-means", "mclust-EEE\n(PCA20)", "Louvain", "Leiden"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot clusterer-sensitivity results")
    parser.add_argument("--output-dir", type=Path, default=COMBINED_DIR / "figures")
    parser.add_argument("--dry-run", action="store_true", help="Perform checks and print provenance without rendering")
    args = parser.parse_args()

    combined_report_path = COMBINED_DIR / "combined_report.json"
    summary_path = COMBINED_DIR / "summary.csv"
    all_runs_path = COMBINED_DIR / "all_runs.csv"
    script_path = Path(__file__).resolve()
    common_path = Path(common.__file__).resolve()

    report = common.load_signed_json(combined_report_path)
    summary_sha = common.compute_sha256(summary_path)
    if summary_sha != report["outputs"]["summary.csv"]:
        raise RuntimeError("Combined summary SHA mismatch")
    all_runs_sha = common.compute_sha256(all_runs_path)
    if all_runs_sha != report["outputs"]["all_runs.csv"]:
        raise RuntimeError("Combined all_runs SHA mismatch")

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite figure directory: {args.output_dir}")

    summary = pd.read_csv(summary_path)
    if set(summary["track"]) != {"validation_frozen", "oracle_true_k"}:
        raise RuntimeError(f"Unexpected tracks: {set(summary['track'])}")
    if set(summary["clusterer"]) != set(ORDER):
        raise RuntimeError(f"Unexpected clusterers: {set(summary['clusterer'])}")

    summary = summary.loc[~summary["method"].eq("FreqNiche-Transductive")].copy()
    primary = summary[summary["track"].eq("validation_frozen")].copy()
    oracle = summary[summary["track"].eq("oracle_true_k")].copy()

    if len(primary) != 20 or len(oracle) != 20:
        raise RuntimeError(f"Unexpected row counts: primary={len(primary)}, oracle={len(oracle)}")
    if set(primary["method"]) != set(oracle["method"]):
        raise RuntimeError("Mismatch in methods between primary and oracle tracks")

    methods = sorted(primary["method"].unique())
    heatmap_vmax = max(
        float(primary["mean_donor_macro_ari"].max()),
        float(oracle["mean_donor_macro_ari"].max()),
    )

    if args.dry_run:
        print("[DRY RUN] All input assertions and structural metadata checks passed.")
        print(f"[DRY RUN] Target output directory: {args.output_dir}")
        print(f"[DRY RUN] Methods ({len(methods)}): {methods}")
        print(f"[DRY RUN] Clusterers ({len(ORDER)}): {ORDER}")
        print(f"[DRY RUN] Primary rows: {len(primary)}, Oracle rows: {len(oracle)}")
        print(f"[DRY RUN] Shared heatmap vmax: {heatmap_vmax:.4f}")
        return

    staging_dir = args.output_dir.parent / f".combined_figures.tmp.{os.getpid()}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    try:
        sns.set_theme(style="whitegrid", context="paper")

        # 1. Inductive FreqNiche clusterer sensitivity reported in the manuscript
        freq = primary[primary["method"].str.startswith("FreqNiche")].copy()
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), constrained_layout=True)
        palette = {"FreqNiche-Inductive": "#3B6FB6"}
        for ax, metric, ylabel in (
            (axes[0], "donor_macro_ari", "Donor-macro ARI"),
            (axes[1], "donor_macro_nmi", "Donor-macro NMI"),
        ):
            sns.barplot(
                data=freq, x="clusterer", y=f"mean_{metric}", hue="method",
                order=ORDER, hue_order=list(palette), palette=palette,
                ax=ax, errorbar=None,
            )
            for method_index, method in enumerate(palette):
                lookup = freq[freq["method"].eq(method)].set_index("clusterer")
                container = ax.containers[method_index]
                errors = [lookup.loc[c, f"sample_sd_{metric}"] for c in ORDER]
                centers = [patch.get_x() + patch.get_width() / 2 for patch in container]
                heights = [patch.get_height() for patch in container]
                ax.errorbar(centers, heights, yerr=errors, fmt="none", color="black", capsize=2, linewidth=0.8)
            ax.set_xlabel("Clusterer")
            ax.set_ylabel(ylabel)
            ax.set_xticks(range(len(DISPLAY_LABELS)))
            ax.set_xticklabels(DISPLAY_LABELS)

        handles, labels = axes[0].get_legend_handles_labels()
        for ax in axes:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        fig.legend(
            handles,
            labels,
            title="Method",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=2,
            frameon=False,
        )
        fig.suptitle(
            "FreqNiche Clusterer Sensitivity (Validation-stability-selected)",
            fontsize=11,
            y=1.12,
        )
        fig.text(
            0.5, -0.02,
            "Note: mclust-EEE evaluated on Validation-fitted PCA20. Error bars show sample standard deviation (N=3 seeds).",
            ha="center", fontsize=8, color="#555555",
        )
        for suffix in ("pdf", "png"):
            fig.savefig(staging_dir / f"freqniche_clusterer_sensitivity.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 2. All methods heatmap (Validation-stability-selected vs Oracle target-K diagnostic)
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), constrained_layout=True)
        for ax, frame, title in (
            (axes[0], primary, "Validation-stability-selected (primary readout)"),
            (axes[1], oracle, "Oracle target-K diagnostic"),
        ):
            matrix = frame.pivot(index="method", columns="clusterer", values="mean_donor_macro_ari")
            matrix = matrix.reindex(index=methods, columns=ORDER)
            sns.heatmap(
                matrix, annot=True, fmt=".3f", cmap="viridis",
                vmin=0, vmax=heatmap_vmax, ax=ax, cbar_kws={"label": "ARI"},
            )
            ax.set_title(title)
            ax.set_xlabel("Clusterer")
            ax.set_ylabel("")
            ax.set_xticklabels(DISPLAY_LABELS, rotation=0)
            ax.set_yticklabels([name.replace("CellNiche-Controlled", "CellNiche").replace("BANKSY-Controlled", "BANKSY").replace("CellCharter-Canonical", "CellCharter").replace("CellCharter-PCA", "CellCharter") for name in methods], rotation=0)
        heatmap_footnote = (
            "All panels use common clustering readouts applied to each method's frozen representation; they are not native end-to-end outputs.\n"
            "Oracle target-K uses the scalar Test reference class count K=10; label identities and ARI/NMI are excluded from selection, "
            "and exact K is not guaranteed for mclust/Louvain/Leiden."
        )
        fig.text(
            0.5, -0.04,
            heatmap_footnote,
            ha="center", fontsize=8, color="#444444",
        )
        for suffix in ("pdf", "png"):
            fig.savefig(staging_dir / f"all_methods_clusterer_heatmap.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 3. Compute figure hashes and provenance
        figure_files = [
            "freqniche_clusterer_sensitivity.pdf",
            "freqniche_clusterer_sensitivity.png",
            "all_methods_clusterer_heatmap.pdf",
            "all_methods_clusterer_heatmap.png",
        ]
        outputs_sha = {}
        for fname in figure_files:
            fpath = staging_dir / fname
            if not fpath.is_file() or fpath.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty figure file: {fname}")
            outputs_sha[fname] = common.compute_sha256(fpath)

        provenance = {
            "provenance_version": "1.0",
            "analysis": "Human Skin clusterer-sensitivity figure provenance",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "plot_script": {
                    "path": common.relative_path(script_path),
                    "sha256": common.compute_sha256(script_path),
                },
                "clusterer_common": {
                    "path": common.relative_path(common_path),
                    "sha256": common.compute_sha256(common_path),
                },
                "combined_report": {
                    "path": common.relative_path(combined_report_path),
                    "sha256": common.compute_sha256(combined_report_path),
                },
                "summary_csv": {
                    "path": common.relative_path(summary_path),
                    "sha256": summary_sha,
                },
                "all_runs_csv": {
                    "path": common.relative_path(all_runs_path),
                    "sha256": all_runs_sha,
                },
            },
            "figure_outputs": outputs_sha,
            "figure_metadata": {
                "clusterer_order": list(ORDER),
                "clusterer_display_labels": [label.replace("\n", " ") for label in DISPLAY_LABELS],
                "methods": list(methods),
                "error_bar_definition": "sample standard deviation across 3 seeds",
                "mclust_specification": "mclust-EEE evaluated on frozen Validation-fitted PCA20",
                "readout_nature": "All panels use common clustering readouts applied to each method's frozen representation; they are not native end-to-end outputs.",
                "oracle_target_k_boundary": "diagnostic readout using the scalar Test reference class count K=10; cell-level label identities and ARI/NMI are excluded from parameter selection; exact K is not guaranteed for mclust/Louvain/Leiden due to empty-component shrinkage or resolution search step.",
                "upstream_model_selection_boundary": "Clusterer hyperparameters are selected without niche identities, but upstream model/embedding candidates were previously selected using Validation metrics.",
                "heatmap_color_scale": {
                    "vmin": 0.0,
                    "vmax": heatmap_vmax,
                    "shared_across_tracks": True,
                },
                "track_semantics": {
                    "validation_stability_selected": "primary readout selected solely by Validation 10-fold bootstrap stability without test labels",
                    "oracle_target_k_diagnostic": "diagnostic readout using the scalar Test reference class count K=10; cell-level label identities and ARI/NMI are excluded from parameter selection",
                },
            },
        }

        provenance_path = staging_dir / "clusterer_figures_provenance.json"
        common.atomic_write_json(provenance_path, provenance, signed=True)

        # 4. Atomic publish
        os.replace(staging_dir, args.output_dir)
        print(f"Figures and provenance atomically published to: {args.output_dir}")

    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
