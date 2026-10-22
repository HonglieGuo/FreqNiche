"""Calculations for within-cell-type niche-associated expression analysis.

Mean log-expression differences, donor-level pseudobulk changes and descriptive
pathway overlap follow the manuscript's same-cell-type expression analysis.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
from typing import Any
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
from scipy.stats import hypergeom, t

ROOT = Path(__file__).resolve().parents[2]
GMT_HASHES_PATH = ROOT / "data/gmt/gmt_hashes.txt"
HALLMARK_PATH = ROOT / "data/gmt/MSigDB_Hallmark_2020.gmt"
GO_BP_PATH = ROOT / "data/gmt/GO_Biological_Process_2023.gmt"
EXPECTED_GROUPS = ("PERIVASC_C0", "STROMA_C2")
EXPECTED_N_GENES = 562

def compute_sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing required asset: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read_expected_gmt_hashes() -> dict[str, str]:
    expected: dict[str, str] = {}
    with GMT_HASHES_PATH.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if ":" not in line:
                raise RuntimeError(f"Malformed GMT hash line: {line}")
            name, digest = line.split(":", 1)
            expected[name.strip()] = digest.strip()
    required = {
        HALLMARK_PATH.name: compute_sha256(HALLMARK_PATH),
        GO_BP_PATH.name: compute_sha256(GO_BP_PATH),
    }
    if expected != required:
        raise RuntimeError(f"GMT provenance mismatch: expected {required}, found {expected}")
    return expected

def check_raw_counts(matrix: Any) -> None:
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix).ravel()
    if values.size and (not np.isfinite(values).all() or np.any(values < 0)):
        raise RuntimeError("Selected expression contains negative or non-finite values")
    if values.size and not np.allclose(values, np.rint(values), rtol=0.0, atol=1e-6):
        raise RuntimeError("Selected expression is not integer raw count data")

def compute_effects(
    selected_adata: ad.AnnData,
    valid_donors: list[str],
) -> pd.DataFrame:
    valid = selected_adata[selected_adata.obs["donor_id"].isin(valid_donors)].copy()
    if valid.n_vars != EXPECTED_N_GENES:
        raise RuntimeError(f"Expected {EXPECTED_N_GENES} genes, found {valid.n_vars}")
    normalized = valid.copy()
    sc.pp.normalize_total(normalized, target_sum=1e4)
    sc.pp.log1p(normalized)

    raw_cpm: dict[str, dict[str, np.ndarray]] = {}
    normalized_means: dict[str, dict[str, np.ndarray]] = {}
    for donor in valid_donors:
        raw_cpm[donor], normalized_means[donor] = {}, {}
        for group in EXPECTED_GROUPS:
            mask = (
                (valid.obs["donor_id"].astype(str) == donor)
                & (valid.obs["expression_group"].astype(str) == group)
            ).to_numpy()
            raw_subset = valid.X[mask]
            count_sum = (
                np.asarray(raw_subset.sum(axis=0)).ravel()
                if sp.issparse(raw_subset) else np.asarray(raw_subset).sum(axis=0)
            )
            library_size = float(count_sum.sum())
            if library_size <= 0:
                raise RuntimeError(f"Zero pseudobulk library size for {donor}/{group}")
            raw_cpm[donor][group] = count_sum / library_size * 1e6

            norm_subset = normalized.X[mask]
            normalized_means[donor][group] = (
                np.asarray(norm_subset.mean(axis=0)).ravel()
                if sp.issparse(norm_subset) else np.asarray(norm_subset).mean(axis=0)
            )

    t_critical = float(t.ppf(0.975, len(valid_donors) - 1))
    records: list[dict[str, Any]] = []
    for gene_idx, gene in enumerate(valid.var_names.astype(str)):
        record: dict[str, Any] = {"gene": gene}
        lfc_values, led_values = [], []
        for donor in valid_donors:
            lfc = float(
                np.log2(raw_cpm[donor]["STROMA_C2"][gene_idx] + 1.0)
                - np.log2(raw_cpm[donor]["PERIVASC_C0"][gene_idx] + 1.0)
            )
            led = float(
                normalized_means[donor]["STROMA_C2"][gene_idx]
                - normalized_means[donor]["PERIVASC_C0"][gene_idx]
            )
            record[f"pseudobulk_log2fc_{donor}"] = lfc
            record[f"mean_log_expr_diff_{donor}"] = led
            lfc_values.append(lfc)
            led_values.append(led)

        for prefix, values in (
            ("pseudobulk_log2fc", np.asarray(lfc_values)),
            ("log_expr_diff", np.asarray(led_values)),
        ):
            mean = float(values.mean())
            sem = float(values.std(ddof=1) / np.sqrt(len(values)))
            direction_count = int(np.sum(values > 0)) if mean > 0 else int(np.sum(values < 0))
            record[f"mean_{prefix}"] = mean
            record[f"ci95_lower_{prefix}"] = mean - t_critical * sem
            record[f"ci95_upper_{prefix}"] = mean + t_critical * sem
            record[f"direction_count_{prefix}"] = direction_count
            record[f"direction_rate_{prefix}"] = direction_count / len(values)
        records.append(record)

    result = pd.DataFrame(records)
    numeric = result.select_dtypes(include=[np.number]).to_numpy()
    if result.shape[0] != EXPECTED_N_GENES or not np.isfinite(numeric).all():
        raise RuntimeError("Gene effect table has an invalid shape or non-finite values")
    return result.sort_values("mean_pseudobulk_log2fc", ascending=False).reset_index(drop=True)

def load_gmt(path: Path, background: set[str]) -> dict[str, set[str]]:
    gene_sets: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                overlap = set(parts[2:]) & background
                if len(overlap) >= 3:
                    gene_sets[parts[0]] = overlap
    return gene_sets

def bh_correction(p_values: np.ndarray) -> np.ndarray:
    if len(p_values) == 0:
        return np.array([], dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 1.0
    for rank_idx in range(len(order) - 1, -1, -1):
        original_idx = order[rank_idx]
        running = min(running, float(p_values[original_idx]) * len(order) / (rank_idx + 1))
        adjusted[original_idx] = min(running, 1.0)
    return adjusted

def compute_ora(
    effects: pd.DataFrame,
    gene_names: list[str],
    lfc_threshold: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required_rate = 0.8
    up = set(effects.loc[
        (effects["mean_pseudobulk_log2fc"] >= lfc_threshold)
        & (effects["direction_rate_pseudobulk_log2fc"] >= required_rate), "gene"
    ])
    down = set(effects.loc[
        (effects["mean_pseudobulk_log2fc"] <= -lfc_threshold)
        & (effects["direction_rate_pseudobulk_log2fc"] >= required_rate), "gene"
    ])
    background = set(gene_names)
    if len(background) != EXPECTED_N_GENES:
        raise RuntimeError(f"ORA background must contain {EXPECTED_N_GENES} genes")

    rows: list[dict[str, Any]] = []
    for database, path in (("Hallmark", HALLMARK_PATH), ("GO_BP", GO_BP_PATH)):
        for direction, selected in (("Up_STROMA_enriched", up), ("Down_STROMA_enriched", down)):
            for pathway, pathway_genes in load_gmt(path, background).items():
                overlap = selected & pathway_genes
                p_value = (
                    float(hypergeom.sf(len(overlap) - 1, len(background), len(pathway_genes), len(selected)))
                    if selected and overlap else 1.0
                )
                rows.append({
                    "Database": database,
                    "Direction": direction,
                    "Pathway": pathway,
                    "P_value": p_value,
                    "Overlap_Size": len(overlap),
                    "Selected_Gene_Set_Size": len(selected),
                    "Pathway_Size": len(pathway_genes),
                    "GeneRatio": len(overlap) / len(pathway_genes),
                    "Genes": ";".join(sorted(overlap)),
                })
    columns = [
        "Database", "Direction", "Pathway", "P_value", "Overlap_Size",
        "Selected_Gene_Set_Size", "Pathway_Size", "GeneRatio", "Genes", "Q_value_BH",
    ]
    if not rows:
        return pd.DataFrame(columns=columns), {"up": len(up), "down": len(down)}
    result = pd.DataFrame(rows)
    result["Q_value_BH"] = 1.0
    for _, group in result.groupby(["Database", "Direction"], observed=True):
        result.loc[group.index, "Q_value_BH"] = bh_correction(group["P_value"].to_numpy())
    result = result[columns].sort_values(["Direction", "Database", "Q_value_BH", "P_value"])
    return result.reset_index(drop=True), {"up": len(up), "down": len(down)}

def save_figure_atomic(fig: plt.Figure, path: Path) -> None:
    temp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.pdf")
    fig.savefig(temp_path, bbox_inches="tight")
    plt.close(fig)
    os.replace(temp_path, path)

def plot_forest(effects: pd.DataFrame, path: Path, top_n: int = 24) -> None:
    ranked = effects.reindex(effects["mean_log_expr_diff"].abs().sort_values(ascending=False).index)
    shown = ranked.head(top_n).sort_values("mean_log_expr_diff")
    fig, ax = plt.subplots(figsize=(8.2, 9.0))
    y = np.arange(len(shown))
    x = shown["mean_log_expr_diff"].to_numpy()
    lower = x - shown["ci95_lower_log_expr_diff"].to_numpy()
    upper = shown["ci95_upper_log_expr_diff"].to_numpy() - x
    ax.errorbar(x, y, xerr=np.vstack([lower, upper]), fmt="o", capsize=3, color="#3B6FB6")
    ax.set_yticks(y, shown["gene"])
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean log-expression difference (STROMA-enriched - PERIVASC-enriched)")
    ax.set_title("Fibro niche-associated expression effects (descriptive)")
    save_figure_atomic(fig, path)

def plot_consistency(
    effects: pd.DataFrame, path: Path, valid_donors: list[str], top_n: int = 30
) -> None:
    ranked = effects.reindex(effects["mean_pseudobulk_log2fc"].abs().sort_values(ascending=False).index)
    shown = ranked.head(top_n)
    columns = [f"pseudobulk_log2fc_{donor}" for donor in valid_donors]
    fig, ax = plt.subplots(figsize=(6.6, 9.4))
    sns.heatmap(
        shown[columns], cmap="vlag", center=0, yticklabels=shown["gene"],
        xticklabels=valid_donors, ax=ax,
    )
    ax.set_title("Donor-level pseudobulk $\\log_2$FC consistency")
    ax.set_xlabel("Donor")
    ax.set_ylabel("Gene")
    save_figure_atomic(fig, path)

def plot_spatial_representatives(
    normalized: ad.AnnData,
    background_obs: pd.DataFrame,
    effects: pd.DataFrame,
    path: Path,
    donor: str = "D145",
) -> None:
    positive = effects.sort_values("mean_pseudobulk_log2fc", ascending=False).iloc[0]["gene"]
    negative = effects.sort_values("mean_pseudobulk_log2fc", ascending=True).iloc[0]["gene"]
    genes = [positive, negative]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), constrained_layout=True)
    donor_background = background_obs.loc[background_obs["donor_id"].astype(str) == donor]
    donor_selected = normalized[normalized.obs["donor_id"].astype(str) == donor]
    for ax, gene in zip(axes, genes):
        ax.scatter(
            donor_background["center_x"], donor_background["center_y"],
            s=0.25, color="#D9D9D9", rasterized=True,
        )
        expression = donor_selected[:, gene].X
        expression = expression.toarray().ravel() if sp.issparse(expression) else np.asarray(expression).ravel()
        points = ax.scatter(
            donor_selected.obs["center_x"], donor_selected.obs["center_y"],
            c=expression, cmap="magma", s=5, rasterized=True,
        )
        ax.set_title(f"{gene} ({donor} selected Fibro)")
        ax.set_aspect("equal")
        ax.set_xlabel("Spatial X")
        ax.set_ylabel("Spatial Y")
        fig.colorbar(points, ax=ax, label="log1p normalized expression")
    save_figure_atomic(fig, path)

def plot_ora(ora: pd.DataFrame, path: Path, top_n: int = 24) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    informative = ora.loc[ora["Overlap_Size"] > 0].copy()
    if informative.empty:
        ax.text(0.5, 0.5, "No pathway overlaps passed the frozen gene filters", ha="center", va="center")
        ax.set_axis_off()
    else:
        shown = informative.sort_values(["Q_value_BH", "P_value"]).head(top_n)
        shown["minus_log10_q"] = -np.log10(shown["Q_value_BH"].clip(lower=1e-12))
        sns.scatterplot(
            data=shown, x="GeneRatio", y="Pathway", size="minus_log10_q",
            hue="Direction", sizes=(25, 220), ax=ax,
        )
        ax.set_title("Descriptive over-representation analysis")
        ax.set_xlabel("Overlap / pathway size")
        ax.set_ylabel("")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure_atomic(fig, path)

def atomic_publish_directory(staging: Path, final: Path) -> None:
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite formal result directory: {final}")
    os.replace(staging, final)
