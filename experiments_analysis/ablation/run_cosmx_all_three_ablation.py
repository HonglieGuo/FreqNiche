"""Add a combined three-component ablation without changing the frozen six-way run."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments_analysis.ablation import run_cosmx_component_ablation as base

VARIANT = "all_three"
DISABLED = ["low_frequency_condition", "high_frequency_condition",
            "visible_and_target_relative_coordinates", "diffusion_training"]
BUDGET = dict(max_epochs=50, patience=7, batch_size=64, min_delta=1e-4)
BASE_MODEL_FOR = "deterministic"


@contextmanager
def combined_intervention():
    """Compose existing interventions in this process only; no source modification."""
    original = base.intervene

    def intervene(batch, variant, device):
        if variant != BASE_MODEL_FOR:
            raise ValueError("Combined intervention requires deterministic reconstruction")
        used = original(batch, "raw_only", device)
        return original(used, "no_coords", device)

    base.intervene = intervene
    try:
        yield
    finally:
        base.intervene = original


def verify_source(source):
    status = base.load(source / "status.json")
    audit = base.load(source / "asset_audit.json")
    if status["status"] != "complete" or audit["status"] != "passed" or audit["packages"] != 18:
        raise ValueError("Source run must be complete with 18 audited packages")
    manifest = base.load(source / "run_manifest.json")
    base.verify_signature(manifest)
    base.check_files(ROOT, manifest["code_sha256"])
    freeze = base.load(source / "freeze.json")
    base.verify_signature(freeze)
    base.check_files(source, freeze["entries"])
    if freeze["run_manifest_sha256"] != base.sha(source / "run_manifest.json"):
        raise ValueError("Source freeze binding changed")
    if manifest["budget"] != BUDGET or manifest["protocol"] != base.PROTOCOL:
        raise ValueError("Unexpected source protocol or training budget")
    if base.sha(source / "inputs/whitener.pt") != manifest["frozen_inputs"][str((base.COSMX_RUN / "whitener.pt").relative_to(ROOT))]:
        raise ValueError("Source whitener changed")
    files = [source / name for name in ["run_manifest.json", "freeze.json", "asset_audit.json",
             "inputs/whitener.pt", "inputs/val_bootstrap_folds.json"]]
    files += [source / "inputs" / split / "manifest.json" for split in ["train", "validation", "test"]]
    return {str(p.relative_to(ROOT)): base.sha(p) for p in files}


def load_context(source, split):
    # Missing caches are an error: prepare_context may only read existing source assets.
    path = source / "inputs" / split / "manifest.json"
    manifest = base.load(path)
    base.check_files(path.parent, manifest["files"])
    return base.prepare_context(source, split, {"h5ad_sha256": manifest["dataset_sha256"]})


def directory(out, seed):
    return out / VARIANT / f"seed_{seed}"


def require_freeze(out):
    freeze = base.load(out / "freeze.json")
    base.verify_signature(freeze)
    base.check_files(out, freeze["entries"])
    if freeze["run_manifest_sha256"] != base.sha(out / "run_manifest.json"):
        raise ValueError("Run changed after validation freeze")
    for seed in base.SEEDS:
        base.completed_selection(directory(out, seed))


def train_validation(args, source, out, device):
    train = load_context(source, "train")
    validation = load_context(source, "validation")
    whitener = base.read_whitener(source / "inputs/whitener.pt")
    folds = base.load(source / "inputs/val_bootstrap_folds.json")
    for seed in base.SEEDS:
        dest = directory(out, seed)
        dest.mkdir(parents=True, exist_ok=True)
        base.write(out / "status.json", {"status": "running", "phase": "train_validation",
                                        "variant": VARIANT, "seed": seed})
        if (dest / "selection.json").exists():
            base.completed_selection(dest)
            continue
        if not (dest / "training.json").exists():
            base.train_seed(args, dest, BASE_MODEL_FOR, seed, train, validation, whitener, device)
            record = base.load(dest / "training.json")
            record.update(variant=VARIANT, model_family=BASE_MODEL_FOR, disabled=DISABLED,
                          supervision="unchanged train-whitened high-frequency residual")
            base.write(dest / "training.json", record)
        training = base.load(dest / "training.json")
        if training.get("disabled") != DISABLED:
            raise ValueError("Training intervention provenance missing or changed")
        base.check_files(dest, {"model_best.pt": training["checkpoint_sha256"],
                               "history.json": training["history_sha256"]})
        model = base.load_model(dest, BASE_MODEL_FOR, device)
        z = base.extract(model, validation, BASE_MODEL_FOR, device)
        del model
        base.save_array(dest / "validation_embeddings.npy", z)
        shutil.copyfile(source / "inputs/validation/centers.csv", dest / "validation_centers.csv")
        resolution, report = base.select_resolution(z, folds, seed)
        names = ["model_best.pt", "history.json", "training.json",
                 "validation_embeddings.npy", "validation_centers.csv"]
        base.write(dest / "selection.json", base.sign({"variant": VARIANT, "seed": seed,
            "disabled": DISABLED, "r_star": resolution, "report": report,
            "assets": {name: base.sha(dest / name) for name in names}}))
        print("VALIDATION_COMPLETE", VARIANT, seed, resolution, flush=True)
    entries = {str((directory(out, seed) / "selection.json").relative_to(out)):
               base.sha(directory(out, seed) / "selection.json") for seed in base.SEEDS}
    base.write(out / "freeze.json", base.sign({"entries": entries,
              "run_manifest_sha256": base.sha(out / "run_manifest.json")}))


def test_phase(args, source, out, device):
    if not args.authorized:
        raise PermissionError("Test evaluation requires --authorized")
    require_freeze(out)  # All three validation selections precede reading Test arrays/labels.
    context = load_context(source, "test")
    centers = base.pd.read_csv(source / "inputs/test/centers.csv",
                              dtype={"cell_barcode": str, "reference_niche": str})
    from sklearn.metrics import adjusted_mutual_info_score
    reference = centers.reference_niche.to_numpy()
    for seed in base.SEEDS:
        dest = directory(out, seed)
        base.write(out / "status.json", {"status": "running", "phase": "test", "variant": VARIANT, "seed": seed})
        if (dest / "test_result.json").exists():
            base.check_files(dest, base.load(dest / "test_result.json")["assets"])
            continue
        selection = base.completed_selection(dest)
        model = base.load_model(dest, BASE_MODEL_FOR, device)
        z = base.extract(model, context, BASE_MODEL_FOR, device)
        del model
        base.save_array(dest / "test_embeddings.npy", z)
        predictions = base.cluster(z, selection["r_star"])
        assigned = centers.copy()
        assigned["predicted_cluster"] = predictions
        assigned.to_csv(dest / "test_assignments.csv", index=False)
        idx = base.np.random.RandomState(42).choice(len(z), min(5000, len(z)), replace=False)
        result = {"variant": VARIANT, "seed": seed, "r_star": selection["r_star"],
            "ari": float(base.adjusted_rand_score(reference, predictions)),
            "nmi": float(base.normalized_mutual_info_score(reference, predictions)),
            "ami": float(adjusted_mutual_info_score(reference, predictions, average_method="arithmetic")),
            "reference_niche_asw": float(base.silhouette_score(z[idx], reference[idx])),
            "n_clusters": int(len(base.np.unique(predictions))),
            "selection_sha256": base.sha(dest / "selection.json"),
            "assets": {name: base.sha(dest / name) for name in ["test_embeddings.npy", "test_assignments.csv"]}}
        base.write(dest / "test_result.json", result)
        print("TEST_COMPLETE", seed, result, flush=True)


def audit_run(args, source, out, device):
    from sklearn.metrics import adjusted_mutual_info_score
    manifest = base.load(out / "run_manifest.json")
    base.verify_signature(manifest)
    base.check_files(ROOT, manifest["source_files"])
    require_freeze(out)
    if (out / "asset_audit.json").exists():
        for package in base.load(out / "asset_audit.json")["assets"]:
            base.check_files(directory(out, package["seed"]), package["files"])
    reference = base.pd.read_csv(source / "inputs/test/centers.csv",
                                dtype={"cell_barcode": str, "reference_niche": str})
    packages, results = [], []
    contexts = {s: load_context(source, s) for s in ["validation", "test"]} if args.replay else {}
    for seed in base.SEEDS:
        dest = directory(out, seed)
        selected = base.completed_selection(dest)
        result = base.load(dest / "test_result.json")
        if result["selection_sha256"] != base.sha(dest / "selection.json") or result["r_star"] != selected["r_star"]:
            raise ValueError("Selection/result mismatch")
        base.check_files(dest, result["assets"])
        assignments = base.pd.read_csv(dest / "test_assignments.csv", dtype={"cell_barcode": str, "reference_niche": str})
        if not assignments.cell_barcode.equals(reference.cell_barcode) or not assignments.reference_niche.equals(reference.reference_niche):
            raise ValueError("Test order or reference changed")
        for key, scorer in [("ari", base.adjusted_rand_score), ("nmi", base.normalized_mutual_info_score),
                            ("ami", adjusted_mutual_info_score)]:
            if not base.np.isclose(scorer(reference.reference_niche, assignments.predicted_cluster), result[key], atol=1e-12, rtol=0):
                raise ValueError("Metric replay mismatch: " + key)
        strict = base.load_model(dest, BASE_MODEL_FOR, base.torch.device("cpu"))
        del strict
        if not (dest / "validation_centers.csv").read_bytes() == (source / "inputs/validation/centers.csv").read_bytes():
            raise ValueError("Validation center order changed")
        for split in ["validation", "test"]:
            saved = base.np.load(dest / f"{split}_embeddings.npy")
            if saved.shape != (10000, 128) or not base.np.isfinite(saved).all():
                raise ValueError("Invalid saved embedding")
            if args.replay:
                model = base.load_model(dest, BASE_MODEL_FOR, device)
                actual = base.extract(model, contexts[split], BASE_MODEL_FOR, device)
                del model
                if not base.np.allclose(actual, saved, atol=1e-5, rtol=1e-5):
                    raise ValueError("Model-to-embedding replay mismatch")
        z = base.np.load(dest / "test_embeddings.npy")
        idx = base.np.random.RandomState(42).choice(len(z), 5000, replace=False)
        asw = float(base.silhouette_score(z[idx], reference.reference_niche.iloc[idx]))
        if not base.np.isclose(asw, result["reference_niche_asw"], atol=1e-8, rtol=0):
            raise ValueError("ASW replay mismatch")
        names = ["model_best.pt", "checkpoint_last.pt", "training.json", "history.json", "selection.json",
                 "validation_embeddings.npy", "validation_centers.csv", "test_embeddings.npy", "test_assignments.csv", "test_result.json"]
        packages.append({"seed": seed, "files": {name: base.sha(dest / name) for name in names}})
        results.append(result)
    summary = {"variant": VARIANT, "display_name": "w/o All Three"}
    for key in ["ari", "nmi", "ami", "reference_niche_asw", "n_clusters"]:
        values = [r[key] for r in results]
        summary[key + "_mean"] = float(base.np.mean(values))
        summary[key + "_std"] = float(base.np.std(values, ddof=1))
    base.pd.DataFrame([{k: r[k] for k in ["variant", "seed", "r_star", "ari", "nmi", "ami", "reference_niche_asw", "n_clusters"]} for r in results]).to_csv(out / "result.csv", index=False)
    base.pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False)
    base.write(out / "summary.json", {"test_donor": "Lung9", "n_test_centers": 10000,
        "disabled": DISABLED, "supervision": "unchanged whitened high-frequency residual",
        "results": [summary], "note": "Single donor; sample SD across seeds is not donor uncertainty."})
    base.write(out / "asset_audit.json", {"status": "passed", "packages": 3,
        "strict_checkpoint_reload": True, "metrics_recomputed": ["ARI", "NMI", "AMI", "ASW"],
        "model_to_embedding_replay": bool(args.replay), "assets": packages})
    base.write(out / "status.json", {"status": "complete", "packages": 3, "audit": "passed"})
    print("ASSET_AUDIT_PASSED", 3, "forward_replay", args.replay, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=["all", "audit", "preflight"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--authorized", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    args.__dict__.update(BUDGET)
    source, out = args.source_run.resolve(), args.output_dir.resolve()
    allowed = (ROOT / "outputs/runs/ablations/cosmx_nsclc").resolve()
    if not out.is_relative_to(allowed) or out == allowed or out == source or out.is_relative_to(source):
        raise ValueError("Use a separate dated ablation output directory")
    if args.phase == "all" and not args.authorized:
        raise PermissionError("Full evaluation requires --authorized")
    base.runtime()
    files = verify_source(source)
    files.update({str(p.relative_to(ROOT)): base.sha(p) for p in [Path(__file__).resolve(), *base.source_paths()]})
    if args.phase == "preflight":
        for split in ["train", "validation"]:
            context = load_context(source, split)
            del context
        base.read_whitener(source / "inputs/whitener.pt")
        print("PREFLIGHT_PASSED: source/config/input hashes verified; no training/output writes")
        return
    if args.phase == "audit" and not (out / "run_manifest.json").exists():
        raise FileNotFoundError("No combined ablation run to audit")
    if out.exists() and any(out.iterdir()) and not args.resume and args.phase != "audit":
        raise FileExistsError("Output exists; use --resume")
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / ".run.lock").open("a") as lock, (out / "run.log").open("a", buffering=1) as log:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.stdout, sys.stderr = base.Tee(sys.stdout, log), base.Tee(sys.stderr, log)
        try:
            device = base.torch.device(args.device)
            if device.type == "cuda" and not base.torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            contract = dict(source_run=str(source.relative_to(ROOT)), source_files=files,
                            variant=VARIANT, disabled=DISABLED, budget=BUDGET,
                            protocol=copy.deepcopy(base.PROTOCOL), test_after_all_three_validation=True)
            if (out / "run_manifest.json").exists():
                old = base.load(out / "run_manifest.json")
                base.verify_signature(old)
                if old != base.sign(contract):
                    raise ValueError("Inputs, code, or configuration changed")
            else:
                base.write(out / "run_manifest.json", base.sign(contract))
                base.write(out / "environment.json", {"python": sys.version, "torch": base.torch.__version__,
                    "numpy": base.np.__version__, "device": str(device)})
            with combined_intervention():
                if args.phase == "all":
                    base.write(out / "status.json", {"status": "running", "phase": "preparing_inputs", "variant": VARIANT})
                    train_validation(args, source, out, device)
                    test_phase(args, source, out, device)
                audit_run(args, source, out, device)
        except Exception as error:
            previous = base.load(out / "status.json") if (out / "status.json").exists() else {}
            base.write(out / "status.json", {**previous, "status": "failed", "error": f"{type(error).__name__}: {error}"})
            raise


if __name__ == "__main__":
    main()
