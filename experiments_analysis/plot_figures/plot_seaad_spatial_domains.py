"""Plot saved Table-1 clusters on all SEA-AD test physical sections, without refitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/h5ad/sea_ad/sea_ad_mtg_merfish.h5ad"
FREQ = ROOT / "outputs/runs/freqniche/sea_ad/2026-10-15/20261015-192535-868567_sea_ad_014-test"
BASE = ROOT / "baselines/Baselines_outputs/runs/sea_ad/2026-09-15/20260915_sea_ad_baselines"
ADDED = ROOT / "baselines/Baselines_outputs/additional_spatial_assets_20261016/sea_ad/additional_spatial"
ORDER = ["Reference cortical layers", "FreqNiche", "GraphST", "SEDR",
         "CellCharter", "SpaGCN", "Raw PCA", "BANKSY"]
OLD_METHODS = {"CellCharter": "cellcharter",
               "Raw PCA": "raw_pca", "BANKSY": "banksy"}
LAYER_COLORS = {"L1": "#8c6bb1", "L2/3": "#41b6c4", "L4": "#a1d76a",
                "L5": "#fdae61", "L6": "#d73027"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def sorted_labels(values):
    return sorted(set(values), key=lambda x: (0, int(x)) if x.isdigit() else (1, x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Paper/image-gen/sea_ad_spatial_domains_20261017")
    parser.add_argument("--main-only", action="store_true", help="Redraw manuscript Figure 2 only; preserve candidate atlas and data files")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for folder in [Path("/mnt/c/Windows/Fonts"), Path("C:/Windows/Fonts")]:
        for filename in ["times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"]:
            path = folder / filename
            if path.is_file():
                font_manager.fontManager.addfont(str(path))
    font_manager.findfont(font_manager.FontProperties(family="Times New Roman"), fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.linewidth": 0.5, "savefig.facecolor": "white"})
    seed = args.seed
    expected = read_csv(FREQ / f"test_assignments_seed_{seed}.csv")
    ids = expected.cell_barcode.tolist()
    if len(ids) != 10000 or len(set(ids)) != len(ids):
        raise ValueError("Unexpected frozen evaluation population")
    data = ad.read_h5ad(DATA, backed="r")
    try:
        obs = data.obs
        indices = obs.index.get_indexer(ids)
        assert (indices >= 0).all()
        metadata = obs.iloc[indices].copy()
        coords = np.asarray(data.obsm["spatial"], dtype=float)
        assert np.isfinite(coords[indices]).all()
        assert (metadata.spatial_unit.astype(str) == "micron").all()
        for key, column in [("donor_id", "donor_id"), ("sample_id", "sample_id"), ("niche", "niche")]:
            assert np.array_equal(metadata[key].astype(str), expected[column])
        for key in ["donor_id", "sample_id", "niche"]:
            expected[key] = expected[key].astype(str)
        expected["x_um"], expected["y_um"] = coords[indices].T
        source_records, frames, resolutions, metric_records = {}, {}, {}, {}
        freq_metrics = json.loads((FREQ / "test_evaluation_results.json").read_text())
        freq_metric = next(v for v in freq_metrics["per_seed_results"] if v["seed"] == seed)

        def register(name, frame, paths, metric, scope, resolution):
            if frame.cell_barcode.duplicated().any() or set(frame.cell_barcode) != set(ids):
                raise ValueError(f"Cell population mismatch: {name}")
            frame = frame.set_index("cell_barcode").loc[ids].reset_index()
            for key in ["donor_id", "sample_id"]:
                assert np.array_equal(frame[key], expected[key]), (name, key)
            label_column = "reference_label" if "reference_label" in frame else "niche"
            assert np.array_equal(frame[label_column], expected.niche), name
            frames[name] = frame
            resolutions[name] = resolution
            donor_values = {}
            for donor, block in frame.groupby("donor_id"):
                mask = expected.donor_id.eq(donor).to_numpy()
                truth = expected.loc[mask, "niche"]
                pred = block.predicted_cluster.to_numpy()
                # groupby preserves row ordering within each donor.
                ari, nmi = adjusted_rand_score(truth, pred), normalized_mutual_info_score(truth, pred)
                original = metric["per_donor"][donor] if "per_donor" in metric else {
                    "ari": metric["donor_aris"][donor], "nmi": metric["donor_nmis"][donor]}
                assert np.isclose(ari, original["ari"], atol=1e-12, rtol=0), (name, donor, ari, original)
                assert np.isclose(nmi, original["nmi"], atol=1e-12, rtol=0), (name, donor, nmi, original)
                donor_values[donor] = {"ari": float(ari), "nmi": float(nmi)}
            source_records[name] = {"files": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in paths],
                                    "readout_scope": scope, "frozen_resolution": resolution,
                                    "verified_seed_donor_metrics": donor_values}

        register("FreqNiche", expected.copy(), [FREQ / f"test_assignments_seed_{seed}.csv", FREQ / "test_evaluation_results.json"],
                 freq_metric, "Saved clustering on the complete 10k evaluation population; cropped by section only for display", freq_metric["r_star"])
        for name, method in OLD_METHODS.items():
            csv = BASE / "assignments" / f"{method}_seed_{seed}_test_assignments.csv"
            result = BASE / "results" / f"{method}_seed_{seed}_results.json"
            metric = json.loads(result.read_text())
            register(name, read_csv(csv), [csv, result], metric,
                     "Saved clustering on the complete 10k evaluation population; cropped by section only for display", metric["r_star"])
        for method, name in [("graphst", "GraphST"), ("sedr", "SEDR"), ("spagcn", "SpaGCN")]:
            directory = ADDED / method / f"seed_{seed}"
            result = directory / "test_result.json"
            metric = json.loads(result.read_text())
            paths = sorted((directory / "assets/test").glob("*/assignments.csv"))
            for p in paths:
                manifest = json.loads((p.parent / "artifact_manifest.json").read_text())
                assert sha(p) == manifest["files"]["assignments.csv"]["sha256"]
            register(name, pd.concat([read_csv(p) for p in paths], ignore_index=True), [result, *paths], metric,
                     "Saved donor-wise clustering on 2000 evaluation centers per donor; cropped by section only for display", metric["leiden_resolution"])
        assert set(frames) == set(ORDER[1:])
        print("VERIFIED_ALL_SEVEN_DISPLAYED_METHODS_SAME_10K_CELLS_AND_RECORDED_METRICS", flush=True)

        # Same numeric category always has the same color within a method across sections.
        # Color identities are deliberately not aligned to reference layers or other methods.
        colors = {"Reference cortical layers": LAYER_COLORS}
        palette = list(plt.get_cmap("tab20").colors)
        for name, frame in frames.items():
            labels = sorted_labels(frame.predicted_cluster)
            assert len(labels) <= len(palette), (name, len(labels))
            colors[name] = dict(zip(labels, palette))
        sections = expected.groupby(["donor_id", "sample_id"]).size().reset_index(name="evaluation_centers")
        representatives = set(sections.sort_values(["donor_id", "evaluation_centers", "sample_id"], ascending=[True, False, True])
                              .drop_duplicates("donor_id").sample_id)
        metrics_rows, section_records = [], []
        merged = expected[["cell_barcode", "donor_id", "sample_id", "niche", "x_um", "y_um"]].copy()
        for name, frame in frames.items():
            merged[name] = frame.predicted_cluster.to_numpy()
        if not args.main_only:
            merged.to_csv(out / "aligned_plot_data_seed42.csv", index=False)
        bg_donors, bg_samples = obs.donor_id.astype(str).to_numpy(), obs.sample_id.astype(str).to_numpy()
        panels_dir = out / "individual_panels"
        panels_dir.mkdir(exist_ok=True)

        def draw_panel(ax, name, block, background, bounds, panel_letter=None, compact=False, legend_ax=None):
            truth = block.niche.to_numpy()
            values = truth if name == ORDER[0] else block[name].to_numpy()
            active = [v for v in colors[name] if v in set(values)]
            ax.scatter(background[:, 0] / 1000, background[:, 1] / 1000, s=0.65 if compact else 1.1,
                       c="#e5e8eb" if compact else "#dedede", edgecolors="none", rasterized=True, zorder=1)
            for label in active:
                mask = values == label
                ax.scatter(block.x_um.to_numpy()[mask] / 1000, block.y_um.to_numpy()[mask] / 1000,
                           s=7.2 if compact else 8.5, color=colors[name][label], linewidths=0, rasterized=True, zorder=2)
            ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel="X (mm)", ylabel="Y (mm)")
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=9 if compact else 8, length=2.5)
            if compact:
                ax.set(xlabel="", ylabel="", xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_color("#b8bec4")
            title = (f"{panel_letter}  " if panel_letter else "") + name
            if name == ORDER[0]:
                subtitle = f"Reference | {len(active)} annotated layers"
            else:
                ari = adjusted_rand_score(truth, values)
                subtitle = f"Slice ARI={ari:.3f} | K={len(active)} | r*={resolutions[name]:g}"
            if compact:
                title = (f"({panel_letter})  " if panel_letter else "") + ("Reference layers" if name == ORDER[0] else name)
                ax.set_title(title, fontsize=11, pad=7)
                if name != ORDER[0]:
                    ax.text(0.97, 0.035, f"ARI = {adjusted_rand_score(truth, values):.3f}",
                            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.5}, zorder=5)
            else:
                ax.set_title(title + "\n" + subtitle, fontsize=10.5, pad=9, linespacing=1.6)
            handles = [Line2D([], [], linestyle="", marker="o", markersize=4.5, color=colors[name][v], label=v) for v in active]
            if not compact:
                ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22 if compact else -0.17),
                          ncol=min(5 if compact else 6, len(active)), fontsize=6.5 if compact else 7.2,
                          frameon=False, handletextpad=0.25, columnspacing=0.55 if compact else 0.75, borderaxespad=0)
            elif legend_ax is not None:
                legend_ax.set_axis_off()
                legend_ax.legend(handles=handles, loc="upper center", ncol=min(5 if name == ORDER[0] else 4, len(active)),
                                 fontsize=8, frameon=False, handlelength=0.65, handletextpad=0.25, columnspacing=0.60,
                                 borderaxespad=0, labelspacing=0.25)
            if name == ORDER[0]:
                inset = ax.inset_axes([0.015, 0.755, 0.23, 0.23])
                inset.scatter(background[:, 0] / 1000, background[:, 1] / 1000, s=0.2,
                              c="#cbd0d5", edgecolors="none", rasterized=True)
                inset.add_patch(Rectangle((bounds[0], bounds[2]), bounds[1] - bounds[0],
                                          bounds[3] - bounds[2], fill=False, edgecolor="#263d5a", linewidth=0.8))
                inset.set_aspect("equal")
                inset.set(xticks=[], yticks=[])
                if not compact:
                    inset.set_title("Full section", fontsize=5.5, pad=1)

        def render_main(block, background, bounds):
            # Two rows of four panels, with independent legend rows below each plot.
            fig = plt.figure(figsize=(7.2, 4.5))
            grid = fig.add_gridspec(2, 4, left=0.02, right=0.99, bottom=0.035, top=0.92,
                                    hspace=0.12, wspace=0.10)
            positions = [(row, col) for row in range(2) for col in range(4)]
            legend_axes = []
            for n, (name, (row, col)) in enumerate(zip(ORDER, positions)):
                slot = grid[row, col].subgridspec(2, 1, height_ratios=[1, 0.42], hspace=0.12)
                ax = fig.add_subplot(slot[0])
                legend_ax = fig.add_subplot(slot[1])
                legend_axes.append(legend_ax)
                draw_panel(ax, name, block, background, bounds, chr(65 + n), compact=True, legend_ax=legend_ax)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for legend_ax in legend_axes:
                box = legend_ax.get_legend().get_window_extent(renderer)
                assert box.x0 >= 0 and box.x1 <= fig.bbox.x1 and box.y0 >= 0, "Legend extends beyond figure canvas"
            for suffix in ["pdf", "svg", "png"]:
                fig.savefig(out / f"fig_seaad_spatial_domain_comparison.{suffix}", dpi=600)
            plt.close(fig)

        if args.main_only:
            block = merged[merged.donor_id.eq("H20.33.035") & merged.sample_id.eq("1194111807")]
            assert len(block) == 1154
            background = coords[(bg_donors == "H20.33.035") & (bg_samples == "1194111807")]
            foreground = block[["x_um", "y_um"]].to_numpy() / 1000
            low, high = foreground.min(axis=0), foreground.max(axis=0)
            pad = np.maximum((high - low) * 0.08, 0.03)
            bounds = (low[0] - pad[0], high[0] + pad[0], low[1] - pad[1], high[1] + pad[1])
            render_main(block, background, bounds)
            metadata_path = out / "provenance.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["main_figure_render"] = {
                "layout": "2 rows x 4 columns", "figsize_inches": [7.2, 4.5],
                "panel_title_fontsize_pt": 11, "row_hspace": 0.12, "column_wspace": 0.10,
                "legend_height_ratio": 0.42,
                "font": "Times New Roman", "panel_labels": "(A)-(H)",
                "ari_position": "inside prediction panel, lower right", "coordinate_axes_hidden": True,
                "raster_dpi": 600, "scatter_rasterized": True, "axes_text_legends_vector": True,
                "global_title": False, "per_panel_legends": True, "script_sha256": sha(Path(__file__)),
                "scope": "Figure 2 appearance only; frozen assignments, coordinates, bounds, category colors and metrics unchanged",
            }
            for suffix in ["pdf", "svg", "png"]:
                p = out / f"fig_seaad_spatial_domain_comparison.{suffix}"
                metadata["generated_files"][p.name] = {"sha256": sha(p), "bytes": p.stat().st_size}
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("MAIN_FIGURE_COMPLETE 8 panels; per-panel legends; 600dpi raster scatter; vector text", flush=True)
            return

        with PdfPages(out / "sea_ad_all_sections_all_methods_seed42.pdf") as all_pdf, \
             PdfPages(out / "sea_ad_largest_section_per_donor_seed42.pdf") as representative_pdf:
            for _, section in sections.iterrows():
                donor, sample = section.donor_id, section.sample_id
                block = merged[merged.donor_id.eq(donor) & merged.sample_id.eq(sample)]
                background = coords[(bg_donors == donor) & (bg_samples == sample)]
                assert len(background) >= len(block)
                foreground = block[["x_um", "y_um"]].to_numpy() / 1000
                low, high = foreground.min(axis=0), foreground.max(axis=0)
                pad = np.maximum((high - low) * 0.08, 0.03)
                bounds = (low[0] - pad[0], high[0] + pad[0], low[1] - pad[1], high[1] + pad[1])
                fig, axes = plt.subplots(2, 4, figsize=(20, 11.5))
                fig.subplots_adjust(left=0.055, right=0.98, bottom=0.20, top=0.85, hspace=0.77, wspace=0.34)
                fig.suptitle(f"SEA-AD | donor {donor} | physical section {sample} | seed {seed}",
                             fontsize=17, fontweight="bold", y=0.97)
                fig.text(0.5, 0.915, f"Same {len(block):,} fixed evaluation centers in every panel; gray = tissue context; full-section location in panel A inset",
                         ha="center", fontsize=10, color="#45505c")
                for n, (name, ax) in enumerate(zip(ORDER, axes.flat)):
                    draw_panel(ax, name, block, background, bounds, chr(65 + n))
                    if name != ORDER[0]:
                        values = block[name].to_numpy()
                        metrics_rows.append({"donor_id": donor, "sample_id": sample, "method": name,
                            "seed": seed, "evaluation_centers": len(block), "background_cells": len(background),
                            "slice_ari": adjusted_rand_score(block.niche, values),
                            "slice_nmi": normalized_mutual_info_score(block.niche, values),
                            "active_clusters_in_slice": len(set(values)), "frozen_r_star": resolutions[name]})
                fig.text(0.5, 0.028, "Saved evaluation clusters; no refitting, reclustering, smoothing or label-based color matching.\n"
                         "Colors identify categories within each panel; the same color across different methods has no shared meaning.\n"
                         "Slice ARI is a descriptive subset metric; Table 1 reports donor-macro results across three seeds.",
                         ha="center", fontsize=10.5, linespacing=1.5, color="#45505c")
                stem = f"sea_ad_{donor.replace('.', '_')}_{sample}_all_methods_seed{seed}"
                for suffix in ["pdf", "svg", "png"]:
                    fig.savefig(out / f"{stem}.{suffix}", dpi=180)
                if donor == "H20.33.035" and sample == "1194111807":
                    render_main(block, background, bounds)
                all_pdf.savefig(fig)
                if sample in representatives:
                    representative_pdf.savefig(fig)
                plt.close(fig)
                if sample in representatives:
                    for n, name in enumerate(ORDER):
                        fig, ax = plt.subplots(figsize=(5.4, 6.4))
                        fig.subplots_adjust(left=0.16, right=0.97, bottom=0.25, top=0.83)
                        draw_panel(ax, name, block, background, bounds)
                        fig.suptitle(f"{donor} | section {sample} | seed {seed}", fontsize=10, y=0.96)
                        safe_name = name.lower().replace(" ", "_").replace("-", "_")
                        for suffix in ["pdf", "svg", "png"]:
                            fig.savefig(panels_dir / f"{donor.replace('.', '_')}_{sample}_{n:02d}_{safe_name}.{suffix}", dpi=180)
                        plt.close(fig)
                section_records.append({"donor_id": donor, "sample_id": sample, "evaluation_centers": len(block),
                                        "background_cells": len(background), "largest_section_for_donor": sample in representatives,
                                        "figure_stem": stem})
                print("SECTION_DONE", donor, sample, len(block), len(background), flush=True)
        pd.DataFrame(metrics_rows).to_csv(out / "slice_metrics_seed42.csv", index=False)
        pd.DataFrame(section_records).to_csv(out / "section_inventory.csv", index=False)
        stat = DATA.stat()
        provenance = {"dataset": "SEA-AD", "seed": seed, "method_order": ORDER, "sources": source_records,
                      "data_file": str(DATA.relative_to(ROOT)), "data_file_bytes": stat.st_size, "data_file_mtime_ns": stat.st_mtime_ns,
                      "source_coordinates": "obsm['spatial']; micron; displayed in mm by dividing by 1000",
                      "physical_section_key": ["donor_id", "sample_id"], "n_frozen_centers": len(ids),
                      "view_policy": "Per-section bounding box of all fixed evaluation centers plus 8 percent padding; no evaluation center omitted. Reference panel inset shows full physical section and shared view rectangle.",
                      "frozen_cell_order_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                      "section_policy": "All 11 test physical sections are provided. Per-donor overview and individual panels use the largest evaluation-center count, ties resolved by sample ID; no outcome-based selection.",
                      "display_scope": "Seven displayed methods; CellNiche and DualFreq-PCA omitted from SEA-AD spatial figures at author request. All original metrics/assets retained; DualFreq-PCA remains in Table 1. Main figure uses donor H20.33.035, section1194111807, seed42.",
                      "reference_colors": LAYER_COLORS, "method_category_colors": colors,
                      "semantics": "Saved clusters cropped to physical sections. No clustering or retraining. No label-color alignment or cross-method color equivalence. Gray background is all cells in that physical section, not predicted cells.",
                      "sections": section_records, "script_sha256": sha(Path(__file__)),
                      "generated_files": {str(p.relative_to(out)): {"sha256": sha(p), "bytes": p.stat().st_size}
                                          for p in out.rglob("*") if p.is_file() and p.name not in {"provenance.json", "README.zh.md"}}}
        (out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        note = """# SEA-AD spatial-domain identification on physical sections: candidate figures for all methods

All seven displayed methods use the formal assets corresponding to Table 1, with seed 42 fixed. At the author's request, CellNiche and DualFreq-PCA are omitted from the spatial figures; DualFreq-PCA remains in Table 1, and all original results and assets are retained. BANKSY is an abbreviated display name; its implementation is unchanged. FreqNiche uses the updated configuration, the three existing baselines use their original formal clustering outputs, and GraphST, SEDR and SpaGCN use verified assets from completed additional runs. All methods cover exactly the same 10,000 evaluation centers, with barcode, donor, physical section and reference layer checked individually. Five-donor ARI/NMI values recomputed from cluster labels match each method's original records.

- fig_seaad_spatial_domain_comparison.pdf/png/svg: selected main-text figure for donor H20.33.035, section1194111807, with 1,154 fixed evaluation centers; reference layers and seven methods in a 2-by-4 layout.
- sea_ad_all_sections_all_methods_seed42.pdf: all 11 physical sections, each page containing eight panels for the reference layers and seven methods.
- sea_ad_largest_section_per_donor_seed42.pdf: the section with the most evaluation centers for each Test donor, totaling five pages; selection does not use metrics or visual appearance.
- sea_ad_<donor>_<sample>_all_methods_seed42.png/pdf/svg: combined figures for direct comparison.
- individual_panels/: 40 standalone panels for the five selected sections, each in PNG/PDF/SVG format for subsequent method selection and layout.
- aligned_plot_data_seed42.csv: coordinates, reference labels and seven sets of predicted labels for the 10,000 evaluation centers.
- slice_metrics_seed42.csv: descriptive ARI/NMI, locally active cluster counts and frozen resolutions for 77 method-section combinations.
- section_inventory.csv and provenance.json: section inventory, source checksums and color mappings.

Saved labels are filtered by physical section only. No retraining, reclustering, smoothing, interpolation or ground-truth-based color alignment is performed. Colored points represent frozen evaluation centers only. Each section's main-panel view uses the bounding box of all evaluation centers plus 8 percent padding, omitting no evaluation centers; gray points show tissue background within that view. The reference-layer panel includes an inset showing the full physical section and the main-panel location. Coordinates retain their original orientation and equal aspect ratio, with microns converted to millimeters; all ten main panels for a given section share the same view.

Reference layers use a fixed palette. Predicted cluster IDs remain unchanged, and palettes are independent across methods; identical colors do not imply the same cortical layer. The three additional methods use donor-wise clustering. The other methods use saved labels obtained by clustering all Test centers and then cropping to sections, following their respective formal readout protocols.

ARI values shown in these figures are descriptive results for individual sections, not the three-seed cross-donor means in Table 1; they must not be used to reselect models or resolutions. All physical sections are retained, including donors with poor performance. The main text uses only the author-specified donor H20.33.035; the remaining figures form the complete candidate archive. To reproduce these figures, run experiments_analysis/plot_figures/plot_seaad_spatial_domains.py with ~/.venvs/FreqNiche-spatial-baselines/bin/python.
"""
        (out / "README.zh.md").write_text(note, encoding="utf-8")
        print("PLOT_COMPLETE", len(section_records), "sections; 7 methods; 40 standalone panels; main donor H20.33.035", out, flush=True)
    finally:
        data.file.close()


if __name__ == "__main__":
    main()
