"""Complete Human Skin Raw-PCA and DualFreq-PCA with frozen W24 readouts.

PCA is fitted on all Train+Validation cells using exact covariance moments.
Processing physical samples separately avoids materializing a million-row dense
dual-frequency feature matrix; it does not subsample the PCA fitting population.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import anndata as ad
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import PCA
from src.benchmark import evaluation_w24 as readout
from src.benchmark.adapters.spatial_artifacts import sha256, signed_read, signed_write
from src.freqniche_st.graph import build_global_spatial_graph

DATA = ROOT / "datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad"
VAL = ROOT / "outputs/contracts/human_skin/validation_barcodes_8k.csv"
TEST = ROOT / "outputs/contracts/human_skin/test_centers_8k.csv"
FOLDS = ROOT / "baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final/bootstrap_manifest_val8k_10fold.json"
DEFAULT_OUTPUT = ROOT / "baselines/Baselines_outputs/runs/human_skin/2026-10-17/20261017_linear_controls"
METHODS, SEEDS = ["raw_pca", "px_resdiff"], [42, 123, 2024]


def pca_from_moments(total, gram, count, n_components=50):
    """Exact centered PCA, equivalent to covariance_eigh, without a dense N-by-D cache."""
    covariance = (gram - np.outer(total, total) / count) / (count - 1)
    covariance = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(covariance)
    values, vectors = np.maximum(values[::-1], 0), vectors[:, ::-1]
    components = vectors.T
    signs = np.sign(components[np.arange(len(components)), np.abs(components).argmax(axis=1)])
    components *= np.where(signs == 0, 1, signs)[:, None]
    model = PCA(n_components=n_components, svd_solver="covariance_eigh")
    model.n_samples_, model.n_features_in_, model.n_components_ = count, len(total), n_components
    model.mean_, model.components_ = total / count, components[:n_components].copy()
    model.explained_variance_ = values[:n_components].copy()
    model.explained_variance_ratio_ = values[:n_components] / values.sum()
    model.singular_values_ = np.sqrt(values[:n_components] * (count - 1))
    model.noise_variance_ = float(values[n_components:].mean())
    model._fit_svd_solver = "covariance_eigh"
    return model


def verify_pca_math():
    x = np.random.default_rng(9).normal(size=(90, 12))
    model = pca_from_moments(x.sum(axis=0), x.T @ x, len(x), 5)
    reference = PCA(n_components=5, svd_solver="covariance_eigh").fit(x)
    assert np.allclose(model.components_, reference.components_, atol=1e-11)
    assert np.allclose(model.transform(x), reference.transform(x), atol=1e-11)
    assert np.allclose(model.explained_variance_, reference.explained_variance_, atol=1e-11)


def prepare(out):
    path = out / "prepared.json"
    if path.exists():
        report = signed_read(path)
        assert report["script_sha256"] == sha256(Path(__file__))
        for rel, digest in report["files"].items():
            assert sha256(out / rel) == digest, rel
        print("VERIFIED_PREPARED_ASSETS", flush=True)
        return report
    verify_pca_math()
    for name, source in [("validation_centers.csv", VAL), ("test_centers.csv", TEST), ("validation_folds.json", FOLDS)]:
        shutil.copyfile(source, out / name)
    source = ad.read_h5ad(DATA, backed="r")
    try:
        val = pd.read_csv(VAL, dtype=str)
        test = pd.read_csv(TEST, dtype=str)
        indices = {"validation": source.obs_names.get_indexer(val.cell_barcode),
                   "test": source.obs_names.get_indexer(test.cell_barcode)}
        for split, index in indices.items():
            assert len(index) == 8000 and len(set(index)) == 8000 and (index >= 0).all()
            assert (source.obs["split"].iloc[index].astype(str) == split).all()
        # Only numeric expression, physical coordinates and split/sample IDs enter feature fitting.
        work = ad.AnnData(X=source.X[:].tocsr())
        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)
        x = work.X.tocsr()
        coords = np.asarray(source.obsm["spatial"], dtype=np.float32)
        samples = source.obs["sample_id"].astype(str).to_numpy()
        splits = source.obs["split"].astype(str).to_numpy()
        assert set(splits) == {"train", "validation", "test"}
        g = x.shape[1]
        xx, xl, ll = [np.zeros((g, g), dtype=np.float64) for _ in range(3)]
        sx, sl, count = np.zeros(g), np.zeros(g), 0
        dual = {split: np.empty((8000, 2 * g), dtype=np.float64) for split in indices}
        maps = {}
        for split, index in indices.items():
            mapping = np.full(len(samples), -1, dtype=np.int32)
            mapping[index] = np.arange(8000)
            maps[split] = mapping
        groups = pd.Series(np.arange(len(samples))).groupby(samples, sort=True).apply(np.asarray)
        print("PREPARING_FULL_CONTEXT", x.shape, "physical_samples", len(groups), flush=True)
        graph_records = []
        for n, (sample, index) in enumerate(groups.items()):
            sample_splits = set(splits[index])
            assert len(sample_splits) == 1
            split = next(iter(sample_splits))
            xs = x[index]
            p, a = build_global_spatial_graph(coords[index], np.repeat(sample, len(index)), radius=50.0, sigma=50.0 / 3.0)
            assert np.allclose(np.asarray(p.sum(axis=1)).ravel(), 1, atol=1e-5)
            low = (p @ xs).toarray().astype(np.float32)
            for phase, mapping in maps.items():
                selected = mapping[index] >= 0
                if selected.any():
                    rows = mapping[index[selected]]
                    own = xs[selected].toarray().astype(np.float64)
                    local_low = low[selected].astype(np.float64)
                    dual[phase][rows] = np.concatenate([local_low, own - local_low], axis=1)
            if split != "test":
                for start in range(0, len(index), 4096):
                    xb = xs[start:start + 4096].astype(np.float64)
                    lb = low[start:start + 4096].astype(np.float64)
                    xx += (xb.T @ xb).toarray()
                    xl += np.asarray(xb.T @ lb)
                    ll += lb.T @ lb
                    sx += np.asarray(xb.sum(axis=0)).ravel()
                    sl += lb.sum(axis=0)
                    count += len(lb)
            graph_records.append({"sample_id": sample, "split": split, "cells": len(index), "edges": int(a.nnz)})
            print("SAMPLE_FEATURES", n + 1, len(groups), sample, split, len(index), flush=True)
            del p, a, low, xs
        assert count == int((splits != "test").sum())
        pd.DataFrame(graph_records).to_csv(out / "sample_graph_summary.csv", index=False)
        pd.DataFrame({"gene": source.var_names.astype(str)}).to_csv(out / "genes.csv", index=False)
        trainval = source.obs.loc[splits != "test", ["donor_id", "sample_id"]].copy()
        trainval.insert(0, "cell_barcode", source.obs_names[splits != "test"].astype(str))
        trainval.to_csv(out / "pca_fit_cells.csv", index=False)
        models = {"raw_pca": pca_from_moments(sx, xx, count),
                  "px_resdiff": pca_from_moments(np.concatenate([sl, sx - sl]),
                      np.block([[ll, xl.T - ll], [xl - ll, xx - xl - xl.T + ll]]), count)}
        for method, model in models.items():
            directory = out / method
            directory.mkdir(exist_ok=True)
            joblib.dump(model, directory / "pca.joblib")
            np.savez_compressed(directory / "pca_basis.npz", mean=model.mean_, components=model.components_,
                                explained_variance=model.explained_variance_, n_fit_cells=count)
            for split, index in indices.items():
                features = x[index].toarray() if method == "raw_pca" else dual[split]
                z = model.transform(features).astype(np.float32)
                assert z.shape == (8000, 50) and np.isfinite(z).all()
                np.save(directory / f"{split}_features.npy", features, allow_pickle=False)
                np.save(directory / f"{split}_embedding.npy", z, allow_pickle=False)
                restored = joblib.load(directory / "pca.joblib").transform(np.load(directory / f"{split}_features.npy"))
                assert np.allclose(z, restored, atol=1e-5, rtol=1e-5)
        report = {"status": "prepared", "script_sha256": sha256(Path(__file__)),
                  "data_sha256": sha256(DATA), "fit_scope": "all Train+Validation cells; no Test cells",
                  "fit_cells": count, "pca_components": 50, "pca_solver": "exact covariance_eigh via per-sample moments",
                  "normalization": "normalize_total(1e4), log1p; no scaling",
                  "dual_frequency": "[P X, X-P X], Gaussian radius50um sigma50/3um, self loops1, sample isolated",
                  "validation_labels_used": False, "seeds": SEEDS,
                  "readout_sha256": sha256(Path(readout.__file__)),
                  "files": {str(p.relative_to(out)): sha256(p) for p in out.rglob("*")
                            if p.is_file() and "logs" not in p.parts and p.name != "run_state.json"}}
        signed_write(path, report)
        print("PREPARE_COMPLETE fit_cells=", count, flush=True)
        return report
    finally:
        source.file.close()
        gc.collect()


def phase(out, method, seed, stage):
    report = signed_read(out / "prepared.json")
    assert report["script_sha256"] == sha256(Path(__file__))
    assert report["readout_sha256"] == sha256(Path(readout.__file__))
    directory = out / method / f"seed_{seed}"
    directory.mkdir(exist_ok=True)
    destination = directory / ("selection.json" if stage == "validation" else "test_result.json")
    if destination.exists():
        raise FileExistsError(f"Phase already completed: {destination}")
    z_path = out / method / f'{"validation" if stage == "validation" else "test"}_embedding.npy'
    assert sha256(z_path) == report["files"][str(z_path.relative_to(out))]
    z = np.load(z_path)
    if stage == "validation":
        r, stability = readout.select_unsupervised_resolution_r_star(
            z, manifest_path=out / "validation_folds.json", val_barcodes_path=out / "validation_centers.csv", seed=seed)
        signed_write(destination, {"method": method, "seed": seed, "r_star": r, "stability": stability,
                                   "prepared_sha256": sha256(out / "prepared.json")})
        print("FROZEN_SELECTION", method, seed, r, flush=True)
        return
    selection = signed_read(directory / "selection.json")
    assert selection["prepared_sha256"] == sha256(out / "prepared.json")
    meta = pd.read_csv(out / "test_centers.csv", dtype=str)
    captured = {}
    original = sc.tl.leiden
    def record_labels(data, *args, **kwargs):
        result = original(data, *args, **kwargs)
        captured["labels"] = data.obs[kwargs["key_added"]].astype(str).to_numpy()
        return result
    with patch.object(sc.tl, "leiden", record_labels):
        metrics = readout.evaluate_common_leiden_readout(z, meta, selection["r_star"], seed=seed, compute_asw=False)
    metrics.pop("asw_silhouette", None)
    assignments = meta[["cell_barcode", "donor_id", "sample_id", "niche"]].copy()
    assignments["predicted_cluster"] = captured["labels"]
    assignments["r_star"] = selection["r_star"]
    assignments.to_csv(directory / "assignments.csv", index=False)
    signed_write(destination, {**metrics, "method": method, "seed": seed,
                               "selection_sha256": sha256(directory / "selection.json"),
                               "assignments_sha256": sha256(directory / "assignments.csv")})
    print("TEST_COMPLETE", method, seed, metrics["test_donor_macro_ari"], metrics["test_donor_macro_nmi"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=["run", "validation", "test"], default="run")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.phase != "run":
        phase(out, args.method, args.seed, args.phase)
        return
    state_path = out / "run_state.json"
    state = {"status": "running", "started_at": datetime.now().astimezone().isoformat(), "completed_phases": []}
    signed_write(state_path, state)
    try:
        prepare(out)
        environment = dict(os.environ)
        environment.update({name: "1" for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]})
        logs = out / "logs"
        logs.mkdir(exist_ok=True)
        def launch(job):
            method, seed, stage = job
            log = logs / f"{method}_{seed}_{stage}.log"
            command = [sys.executable, str(Path(__file__)), "--output-dir", str(out), "--phase", stage,
                       "--method", method, "--seed", str(seed)]
            with log.open("w", encoding="utf-8") as handle:
                subprocess.run(command, env=environment, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
            return f"{method}/{seed}/{stage}"
        # Every selection is frozen before any Test metric is evaluated.
        for stage in ["validation", "test"]:
            jobs = [(m, s, stage) for m in METHODS for s in SEEDS]
            with ThreadPoolExecutor(max_workers=3) as pool:
                for completed in pool.map(launch, jobs):
                    state["completed_phases"].append(completed)
                    signed_write(state_path, state)
                    print("PHASE_COMPLETE", completed, flush=True)
        summary = {}
        for method in METHODS:
            results = [signed_read(out / method / f"seed_{s}/test_result.json") for s in SEEDS]
            summary[method] = {"ari_mean": statistics.mean(r["test_donor_macro_ari"] for r in results),
                               "ari_sd": statistics.stdev(r["test_donor_macro_ari"] for r in results),
                               "nmi_mean": statistics.mean(r["test_donor_macro_nmi"] for r in results),
                               "nmi_sd": statistics.stdev(r["test_donor_macro_nmi"] for r in results), "seeds": SEEDS}
        signed_write(out / "summary.json", summary)
        state.update(status="complete", finished_at=datetime.now().astimezone().isoformat())
        signed_write(state_path, state)
        print("LINEAR_CONTROLS_COMPLETE", json.dumps(summary), flush=True)
    except Exception as error:
        state.update(status="failed", error=repr(error))
        signed_write(state_path, state)
        raise


if __name__ == "__main__":
    main()
