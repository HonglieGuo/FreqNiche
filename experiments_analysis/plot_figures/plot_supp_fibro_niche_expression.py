"""Compose the four existing Fibro analyses without rerunning statistics.

Uses the exact original gene/pathway rankings and saved effect estimates.
Only the two spatial panels read raw counts, applying the original per-cell
normalization for display. Formal result assets are always read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "outputs/runs/analyses/human_skin/2026-09-23/20260923_human_skin_task_c/formal_results_v1"
ASSIGNMENTS = ROOT / "outputs/runs/analyses/human_skin/2026-09-23/20260923_human_skin_task_c/task_c_assignments_seed42.csv"
DATASET = ROOT / "datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad"
BLUE, ORANGE = "#2E6FAD", "#C56C3A"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style() -> str:
    for folder in (Path("/mnt/c/Windows/Fonts"), Path("C:/Windows/Fonts")):
        for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
            if (folder / name).exists():
                font_manager.fontManager.addfont(str(folder / name))
    resolved = font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "mathtext.fontset": "stix",
    })
    return resolved


def spatial_data(genes: list[str]):
    assignments = pd.read_csv(ASSIGNMENTS)
    flag = assignments["is_task_c_selected"].astype(str).str.lower().eq("true")
    selected = assignments.loc[flag & assignments["donor_id"].eq("D145")].copy()
    counts = selected.groupby("task_c_group").size().to_dict()
    assert counts == {"PERIVASC_C0": 114, "STROMA_C2": 287}, counts
    backed = ad.read_h5ad(DATASET, backed="r")
    try:
        background = backed.obs.loc[
            backed.obs["donor_id"].astype(str).eq("D145"), ["center_x", "center_y"]
        ].copy()
        positions = backed.obs_names.get_indexer(selected["cell_barcode"].astype(str))
        assert (positions >= 0).all()
        selected_ad = backed[np.sort(positions)].to_memory()
    finally:
        backed.file.close()
    assert selected_ad.n_vars == 562 and selected_ad.n_obs == 401
    sc.pp.normalize_total(selected_ad, target_sum=1e4)
    sc.pp.log1p(selected_ad)
    values = selected_ad[:, genes].X
    values = values.toarray() if sparse.issparse(values) else np.asarray(values)
    assert np.isfinite(values).all() and values.min() >= 0
    return background, selected_ad.obs[["center_x", "center_y"]].copy(), values


def title(fig, label, text, x, y):
    fig.text(x, y, f"({label})  {text}", ha="center", va="bottom", fontsize=9)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(pad=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Paper/image-gen/s5_fibro_niche_expression")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RESULTS / "task_c_results_manifest.json").read_text())
    expected = manifest["outputs"]
    source_files = [RESULTS / name for name in expected]
    before = {str(p.relative_to(ROOT)): digest(p) for p in source_files}
    for name, sha in expected.items():
        assert digest(RESULTS / name) == sha, name
    assert digest(ASSIGNMENTS) == manifest["inputs"]["assignments"]["sha256"]
    effects = pd.read_csv(RESULTS / "task_c_gene_effects.csv")
    ora = pd.read_csv(RESULTS / "task_c_ora_results.csv")
    # These selections exactly match the four original plotting functions.
    forest = effects.reindex(effects["mean_log_expr_diff"].abs().sort_values(ascending=False).index).head(24).sort_values("mean_log_expr_diff")
    heat = effects.reindex(effects["mean_pseudobulk_log2fc"].abs().sort_values(ascending=False).index).head(30)
    pathways = ora.loc[ora["Overlap_Size"] > 0].sort_values(["Q_value_BH", "P_value"]).head(24).copy()
    genes = [str(effects.sort_values("mean_pseudobulk_log2fc", ascending=False).iloc[0]["gene"]),
             str(effects.sort_values("mean_pseudobulk_log2fc", ascending=True).iloc[0]["gene"])]
    assert genes == ["POPDC3", "IL2RG"]
    background, cells, expression = spatial_data(genes)
    font = style()
    fig = plt.figure(figsize=(7.2, 8.4), facecolor="white")

    ax = fig.add_axes([0.105, 0.555, 0.325, 0.375])
    x = forest["mean_log_expr_diff"].to_numpy()
    y = np.arange(len(forest))
    ax.errorbar(x, y, xerr=np.vstack([
        x - forest["ci95_lower_log_expr_diff"].to_numpy(),
        forest["ci95_upper_log_expr_diff"].to_numpy() - x,
    ]), fmt="none", ecolor="#71859C", elinewidth=0.7, capsize=1.6, capthick=0.6)
    ax.scatter(x, y, c=np.where(x >= 0, BLUE, ORANGE), s=11, zorder=3)
    ax.axvline(0, color="#666666", linewidth=0.6, linestyle="--")
    ax.set_yticks(y, forest["gene"], fontstyle="italic")
    ax.set_ylim(-0.8, len(forest) - 0.2)
    ax.set_xlabel("Mean log-expression difference\nSTROMA-enriched − PERIVASC-enriched", labelpad=4)
    clean(ax)
    title(fig, "A", "Expression differences", 0.265, 0.947)

    ax = fig.add_axes([0.65, 0.555, 0.23, 0.375])
    matrix = heat[[f"pseudobulk_log2fc_{d}" for d in ["D082", "D145", "D151"]]].to_numpy()
    vmax = float(np.abs(matrix).max())
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest", rasterized=True)
    ax.set_yticks(np.arange(len(heat)), heat["gene"], fontstyle="italic")
    ax.set_xticks([0, 1, 2], ["D082", "D145", "D151"])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=0, pad=4)
    for spine in ax.spines.values(): spine.set_visible(False)
    cax = fig.add_axes([0.9, 0.555, 0.016, 0.375])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Pseudobulk $\\log_2$FC", fontsize=8)
    cb.outline.set_linewidth(0.5)
    title(fig, "B", "Across-donor consistency", 0.745, 0.947)

    # Shared color scale; every background and expression point is rasterized.
    norm = Normalize(vmin=0, vmax=float(np.ceil(expression.max())))
    bounds = [background["center_x"].min(), background["center_x"].max(),
              background["center_y"].min(), background["center_y"].max()]
    for k, (label, gene, left) in enumerate(zip(["C", "D"], genes, [0.035, 0.278])):
        ax = fig.add_axes([left, 0.135, 0.222, 0.285])
        ax.scatter(background["center_x"], background["center_y"], s=0.055,
                   c="#DDDDDD", linewidths=0, rasterized=True)
        points = ax.scatter(cells["center_x"], cells["center_y"], c=expression[:, k],
                            norm=norm, cmap="magma", s=1.9, linewidths=0, rasterized=True)
        ax.set_xlim(bounds[0] - 200, bounds[1] + 200)
        ax.set_ylim(bounds[2] - 200, bounds[3] + 200)
        ax.set_aspect("equal")
        ax.set_anchor("N")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_linewidth(0.5)
        fig.text(left + 0.111, 0.447, f"({label})  {gene}",
                 ha="center", va="bottom", fontsize=9, fontstyle="italic")
    cax = fig.add_axes([0.09, 0.245, 0.345, 0.011])
    cb = fig.colorbar(points, cax=cax, orientation="horizontal", ticks=[0, 2, 4, 6])
    cb.set_label("log1p normalized expression", labelpad=2)
    cb.outline.set_linewidth(0.5)
    fig.text(0.265, 0.197, "D145 Fibro", fontsize=8, ha="center")

    ax = fig.add_axes([0.775, 0.135, 0.18, 0.285])
    pathway_labels = pathways["Pathway"].str.replace(r" \(GO:\d+\)$", "", regex=True).tolist()
    display_corrections = {"Pperoxisome": "Peroxisome", "heme Metabolism": "Heme Metabolism"}
    pathway_labels = [display_corrections.get(name, name) for name in pathway_labels]
    unique_pathways = list(dict.fromkeys(pathway_labels))
    positions = [unique_pathways.index(name) for name in pathway_labels]
    q = -np.log10(pathways["Q_value_BH"].clip(lower=1e-12).to_numpy())
    sizes = 8 + 40 * q
    colors = np.where(pathways["Direction"].eq("Up_STROMA_enriched"), BLUE, ORANGE)
    ax.scatter(pathways["GeneRatio"], positions, s=sizes, c=colors,
               linewidths=0.3, edgecolors="white", alpha=0.95)
    ax.set_yticks(np.arange(len(unique_pathways)), unique_pathways, fontsize=6.9)
    ax.invert_yaxis()
    ax.set_xlabel("Overlap / pathway size", labelpad=4)
    ax.set_xlim(0.04, 0.48)
    ax.set_xticks([0.1, 0.3, 0.5])
    ax.grid(axis="x", color="#E7E9EC", linewidth=0.45)
    ax.set_axisbelow(True)
    clean(ax)
    title(fig, "E", "Pathway overlap", 0.755, 0.447)
    direction = [Line2D([], [], marker="o", linestyle="", color=c, markersize=4, label=label)
                 for c, label in [(BLUE, "STROMA-enriched"), (ORANGE, "PERIVASC-enriched")]]
    fig.legend(handles=direction, loc="center", bbox_to_anchor=(0.755, 0.485),
               ncol=2, frameon=False, fontsize=7, handletextpad=0.3, columnspacing=0.8)
    size_handles = [plt.scatter([], [], s=8 + 40 * v, c="#6E7784", linewidths=0) for v in [0, 1, 2]]
    fig.legend(size_handles, ["0", "1", "2"], title="$-\\log_{10}(q)$",
               loc="center", bbox_to_anchor=(0.755, 0.055), ncol=3,
               frameon=False, fontsize=7, title_fontsize=7.5, handletextpad=0.5,
               columnspacing=1)

    stem = args.output_dir / "fig_supp_fibro_niche_expression"
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(stem.with_suffix("." + suffix), dpi=600, bbox_inches="tight", pad_inches=0.055)
    plt.close(fig)
    after = {str(p.relative_to(ROOT)): digest(p) for p in source_files}
    assert before == after
    payload = {"status": "passed", "source_results_unchanged": True,
               "source_hashes": before, "font": str(font), "scatter_dpi": 600,
               "no_statistics_recomputed": True, "pathway_display_corrections": display_corrections,
               "forest_genes": forest["gene"].tolist(),
               "heatmap_genes": heat["gene"].tolist(), "ora_rows": pathways[["Database", "Direction", "Pathway", "Q_value_BH"]].to_dict("records"),
               "spatial_genes": genes, "spatial_selected_cells": len(cells),
               "spatial_background_cells": len(background), "expression_color_limits": [norm.vmin, norm.vmax]}
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
