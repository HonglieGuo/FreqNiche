"""Within-cell-type niche-associated expression analysis from a completed run.

Compute donor-level expression effects, pseudobulk changes and pathway overlap
for Fibro cells in the manuscript's STROMA- and PERIVASC-enriched niche groups.
Cluster roles must satisfy the recorded enrichment and minimum-donor conditions.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.freqniche_st.downstream_inputs import RunInputs, sha256
from experiments_analysis.niche_associated_expression import expression_analysis as analysis


def select_groups(metadata: pd.DataFrame, clusters: np.ndarray, *, perivasc: str, stroma: str,
                  min_cells: int) -> tuple[pd.DataFrame, list[str], dict]:
    if perivasc == stroma:
        raise ValueError("The two biological roles must use different clusters")
    if min_cells < 20:
        raise ValueError("Niche-associated expression analysis requires at least 20 Fibro cells per donor/group")
    metadata = metadata.copy()
    metadata["pred_freqniche"] = np.asarray(clusters).astype(str)
    summary = pd.crosstab(metadata["pred_freqniche"], metadata["niche"], normalize="index")
    print("Post-hoc cluster niche proportions:\n", summary.to_string())
    enrichment = {}
    for role, cluster, labels in (("perivasc", perivasc, ["PERIVASC-I", "PERIVASC-II"]),
                                  ("stroma", stroma, ["STROMA"])):
        members = metadata.loc[metadata["pred_freqniche"] == cluster]
        ratio = float(members["niche"].isin(labels).mean()) if len(members) else 0.0
        if ratio <= 0.5:
            raise ValueError(
                f"Cluster {cluster} does not meet the declared {role} role (fraction={ratio:.3f}). "
                "Inspect the proportions; explicitly supply --perivasc-cluster/--stroma-cluster. "
                "If the biological conditions are absent, Niche-associated expression analysis is not applicable to this run."
            )
        enrichment[role] = ratio
    selected = metadata.loc[metadata["cell_type"].eq("Fibro")
                            & metadata["pred_freqniche"].isin([perivasc, stroma])].copy()
    # These names are stable role keys used by the historical effect calculator;
    # actual new cluster IDs are recorded separately in the manifest.
    selected["expression_group"] = np.where(selected["pred_freqniche"].eq(stroma), "STROMA_C2", "PERIVASC_C0")
    counts = selected.groupby(["donor_id", "expression_group"], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(columns=analysis.EXPECTED_GROUPS, fill_value=0)
    donors = sorted(counts.index[(counts >= min_cells).all(axis=1)].astype(str).tolist())
    if len(donors) < 3:
        raise ValueError(f"Niche-associated expression analysis requires >=3 donors with >= {min_cells} cells in both groups; found {donors}")
    return selected, donors, {"roles": {"PERIVASC_C0": perivasc, "STROMA_C2": stroma},
                              "enrichment": enrichment, "counts": counts.to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--perivasc-cluster", default="0")
    parser.add_argument("--stroma-cluster", default="2")
    parser.add_argument("--min-cells", type=int, default=20)
    parser.add_argument("--lfc-threshold", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs/biology without writing outputs")
    args = parser.parse_args()
    if not np.isfinite(args.lfc_threshold) or args.lfc_threshold <= 0:
        parser.error("--lfc-threshold must be finite and positive")
    inputs = RunInputs(args.run_dir, "human_skin")
    if 42 not in inputs.seeds:
        raise ValueError("Niche-associated expression analysis uses the predeclared Seed 42 embedding")
    target = inputs.run_dir / "downstream/niche_associated_expression"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite Niche-associated expression analysis outputs: {target}")
    centers = inputs.centers["test"]
    barcodes = centers["cell_barcode"].tolist()
    backed = ad.read_h5ad(inputs.data_path, backed="r")
    try:
        obs = backed.obs.loc[barcodes].copy()
        obs.insert(0, "cell_barcode", barcodes)
        obs["embedding_row"] = np.arange(len(obs))
        embedding = np.load(inputs.embedding_path("test", 42), allow_pickle=False)
        clustering = ad.AnnData(X=embedding)
        sc.pp.neighbors(clustering, n_neighbors=15, use_rep="X", random_state=42)
        sc.tl.leiden(clustering, resolution=0.8, key_added="cluster", random_state=42)
        selected, donors, selection = select_groups(
            obs, clustering.obs["cluster"].astype(str).to_numpy(),
            perivasc=args.perivasc_cluster, stroma=args.stroma_cluster, min_cells=args.min_cells,
        )
        # Sort row positions for HDF5 fancy indexing, keeping metadata aligned.
        positions = backed.obs_names.get_indexer(selected["cell_barcode"])
        if (positions < 0).any():
            raise ValueError("Selected barcodes are absent from the source H5AD")
        order = np.argsort(positions)
        selected = selected.iloc[order]
        counts = backed[positions[order]].to_memory()
        counts.obs["expression_group"] = selected["expression_group"].to_numpy()
        background = backed.obs[["donor_id", "center_x", "center_y"]].copy()
    finally:
        backed.file.close()
    analysis.check_raw_counts(counts.X)
    if counts.n_vars != analysis.EXPECTED_N_GENES:
        raise ValueError(f"Expected the released {analysis.EXPECTED_N_GENES}-gene Human Skin panel")
    gmt_hashes = analysis.read_expected_gmt_hashes()
    if args.dry_run:
        print(f"Niche-associated expression analysis preflight passed: {counts.n_obs} cells, {inputs.width} dimensions, donors={donors}; no writes")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".niche_associated_expression.tmp.{os.getpid()}")
    staging.mkdir(exist_ok=False)
    try:
        (staging / "figures").mkdir()
        selected.to_csv(staging / "assignments.csv", index=False)
        pd.DataFrame({"cell_barcode": barcodes, "cluster": clustering.obs["cluster"].astype(str).to_numpy()}).to_csv(
            staging / "all_test_clusters.csv", index=False)
        effects = analysis.compute_effects(counts, donors)
        ora, gene_counts = analysis.compute_ora(effects, counts.var_names.astype(str).tolist(), args.lfc_threshold)
        effects.to_csv(staging / "gene_effects.csv", index=False)
        ora.to_csv(staging / "ora_results.csv", index=False)
        normalized = counts[counts.obs["donor_id"].isin(donors)].copy()
        sc.pp.normalize_total(normalized, target_sum=1e4)
        sc.pp.log1p(normalized)
        analysis.plot_forest(effects, staging / "figures/effects.pdf")
        analysis.plot_consistency(effects, staging / "figures/consistency.pdf", donors)
        analysis.plot_spatial_representatives(normalized, background, effects, staging / "figures/spatial.pdf",
                                             donor="D145" if "D145" in donors else donors[0])
        analysis.plot_ora(ora, staging / "figures/ora.pdf")
        manifest = {
            "analysis": "Niche-associated expression analysis new-run post-hoc descriptive analysis",
            "input_binding_sha256": sha256(inputs.run_dir / "downstream_inputs.json"),
            "runner_sha256": sha256(Path(__file__)),
            "calculator_sha256": sha256(Path(analysis.__file__)),
            "seed": 42, "embedding_width": inputs.width,
            "leiden": {"n_neighbors": 15, "resolution": 0.8, "random_state": 42},
            "valid_donors": donors, "N_eff": len(donors), "min_cells": args.min_cells,
            "lfc_threshold": args.lfc_threshold, "selected_gene_counts": gene_counts,
            "selection": selection, "gmt_hashes": gmt_hashes,
            "outputs": {p.relative_to(staging).as_posix(): sha256(p) for p in staging.rglob("*") if p.is_file()},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        analysis.atomic_publish_directory(staging, target)
        print(f"Niche-associated expression analysis complete: {target}")
    finally:
        # Only the staging directory created by this invocation is cleaned up.
        if staging.exists() and staging.resolve().parent == target.parent.resolve():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
