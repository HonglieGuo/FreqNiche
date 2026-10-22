"""Read-only integrity, PCA replay and metric verification for the new linear controls."""
import argparse
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from src.benchmark.adapters.spatial_artifacts import sha256, signed_read, signed_write
from scripts.benchmark.run_human_skin_linear_controls import DEFAULT_OUTPUT, DATA, VAL, TEST, FOLDS, METHODS, SEEDS, verify_pca_math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    state = signed_read(out / "run_state.json")
    assert state["status"] == "complete" and len(state["completed_phases"]) == 12
    prepared = signed_read(out / "prepared.json")
    assert sha256(DATA) == prepared["data_sha256"]
    assert sha256(ROOT / "scripts/benchmark/run_human_skin_linear_controls.py") == prepared["script_sha256"]
    for rel, digest in prepared["files"].items():
        assert sha256(out / rel) == digest, rel
    for source, copy in [(VAL, "validation_centers.csv"), (TEST, "test_centers.csv"), (FOLDS, "validation_folds.json")]:
        assert sha256(source) == sha256(out / copy)
    verify_pca_math()
    meta = pd.read_csv(out / "test_centers.csv", dtype=str)
    assert len(meta) == 8000 and meta.cell_barcode.is_unique
    assert meta.groupby("donor_id").size().to_dict() == {d: 2000 for d in ["D028", "D082", "D145", "D151"]}
    fitted = pd.read_csv(out / "pca_fit_cells.csv", dtype=str)
    assert len(fitted) == 887228 and fitted.cell_barcode.is_unique
    assert not set(fitted.donor_id).intersection(meta.donor_id)
    summary = signed_read(out / "summary.json")
    checked = []
    for method in METHODS:
        model = joblib.load(out / method / "pca.joblib")
        assert model.n_samples_ == 887228 and model.n_components_ == 50
        for split in ["validation", "test"]:
            features = np.load(out / method / f"{split}_features.npy", mmap_mode="r")
            z = np.load(out / method / f"{split}_embedding.npy")
            assert z.shape == (8000, 50) and np.isfinite(z).all()
            assert np.allclose(model.transform(features), z, atol=1e-5, rtol=1e-5)
        aris, nmis = [], []
        for seed in SEEDS:
            directory = out / method / f"seed_{seed}"
            selection = signed_read(directory / "selection.json")
            assert selection["prepared_sha256"] == sha256(out / "prepared.json")
            result = signed_read(directory / "test_result.json")
            assert result["selection_sha256"] == sha256(directory / "selection.json")
            assert result["assignments_sha256"] == sha256(directory / "assignments.csv")
            assignment = pd.read_csv(directory / "assignments.csv", dtype=str)
            assert np.array_equal(assignment.cell_barcode, meta.cell_barcode)
            assert np.array_equal(assignment.niche, meta.niche)
            assert (assignment.r_star.astype(float) == selection["r_star"]).all()
            values = []
            for donor, block in assignment.groupby("donor_id"):
                ari = adjusted_rand_score(block.niche, block.predicted_cluster)
                nmi = normalized_mutual_info_score(block.niche, block.predicted_cluster)
                stored = result["per_donor_metrics"][donor]
                assert abs(ari - stored["ari"]) < 1e-12 and abs(nmi - stored["nmi"]) < 1e-12
                values.append((ari, nmi))
            ari, nmi = np.mean(values, axis=0)
            assert abs(ari - result["test_donor_macro_ari"]) < 1e-12
            assert abs(nmi - result["test_donor_macro_nmi"]) < 1e-12
            aris.append(float(ari))
            nmis.append(float(nmi))
            checked.append({"method": method, "seed": seed, "r_star": selection["r_star"],
                            "ari": float(ari), "nmi": float(nmi)})
        for field, value in [("ari_mean", statistics.mean(aris)), ("ari_sd", statistics.stdev(aris)),
                             ("nmi_mean", statistics.mean(nmis)), ("nmi_sd", statistics.stdev(nmis))]:
            assert abs(value - summary[method][field]) < 1e-12
    signed_write(out / "asset_audit.json", {"status": "passed", "protected_for_paper": True,
        "pca_replay": True, "raw_metric_recalculation": True, "no_test_cells_in_pca_fit": True,
        "prepared_sha256": sha256(out / "prepared.json"), "summary_sha256": sha256(out / "summary.json"),
        "results": checked})
    print("LINEAR_CONTROLS_AUDIT_PASS: 2 PCA models, 4 embeddings, 6 frozen selections and 6 Test assignment/metric bundles")


if __name__ == "__main__":
    main()
