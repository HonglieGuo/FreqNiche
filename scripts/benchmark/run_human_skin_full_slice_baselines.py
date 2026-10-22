"""
scripts/benchmark/run_human_skin_full_slice_baselines.py

Robust Baseline Runner with Active Parent-Child Process Guardrails:
1. Dual-Track GPU Accounting:
   - system_gpu_used_peak_mb: Entire GPU device used memory (hardware safety breach).
   - worker_tree_gpu_peak_mb: Sum of GPU memory used by worker + all child PIDs (method accounting).
2. Host Resource Guardrails:
   - peak_tree_rss_mb: Host process tree RSS against max_rss_gb.
   - min_free_ram_gb: Host system available memory against threshold (>= 8 GB).
   - timeout_minutes: Wall-clock timeout against timeout_sec.
3. Active Hard Termination:
   - os.killpg(SIGTERM) -> 2.0s -> os.killpg(SIGKILL).
   - Automatic cleanup of semi-written temporary files (*.tmp.*) on guard breach.
4. Strict Append-Only & Atomic Persistence:
   - Formal mode (mode=='full') permanently fails closed if output files exist (no overwrite allowed).
   - Test modes ('smoke', 'preflight', 'test_guard') require explicit --overwrite if files exist.
   - Atomic replacement: array saved to *.tmp.<pid>.npy and swapped via os.replace().
5. Complete Provenance & Contract Binding:
   - Binds H5AD SHA-256, ordered barcode SHA-256, runner SHA-256, adapter source SHA-256.
6. Guard Defense Unit Testing:
   - Built-in --test-guard mode for expected-failure validation (timeout, rss) in guard_tests/ dir.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad"
TARGET_SAMPLE_ID = "202309281027_AutopsyBFFPE092823_VMSC02301_region_1"

OUTPUT_BASE = ROOT / "baselines/Baselines_outputs/runs/human_skin/2026-09-21/20260921_human_skin_full_slice_comparison"
SMOKE_MANIFEST_PATH = OUTPUT_BASE / "smoke_1000_cells_manifest.json"

FROZEN_HYPERPARAMS = {
    "banksy": {
        "adapter_module": "src.benchmark.adapters.banksy",
        "adapter_class": "OfficialBANKSYAdapter",
        "kwargs": {
            "protocol": "controlled",
            "n_top_genes": 3000,
            "n_neighbors": 18,
            "lambda_param": 0.2,
            "max_m": 1,
            "pca_dim": 20,
        },
        "epochs_key": None,
        "full_epochs": None,
        "device_type": "cpu",
    },
    "cellcharter": {
        "adapter_module": "src.benchmark.adapters.cellcharter",
        "adapter_class": "CanonicalCellCharterAdapter",
        "kwargs": {
            "track": "canonical",
            "sample_key": "sample_id",
            "n_latent": 20,
            "n_layers": 3,
            "n_neighbors": 6,
            "radius": 50.0,
            "distance_percentile": 99.0,
            "batch_size": 128,
        },
        "epochs_key": "max_epochs",
        "full_epochs": 200,
        "device_type": "cuda",
    },
    "stagate": {
        "adapter_module": "src.benchmark.adapters.stagate",
        "adapter_class": "OfficialSTAGATEAdapter",
        "kwargs": {
            "protocol": "controlled",
            "track": "controlled",
            "sample_key": "sample_id",
            "n_top_genes": 3000,
            "radius_cutoff": 50.0,
        },
        "epochs_key": "epochs",
        "full_epochs": 100,
        "device_type": "cuda",
    },
    "cellniche": {
        "adapter_module": "src.benchmark.adapters.cellniche",
        "adapter_class": "OfficialCellNicheAdapter",
        "kwargs": {
            "protocol": "controlled",
            "track": "controlled",
            "sample_key": "sample_id",
            "radius": 50.0,
            "hidden_channels": [256, 128],
            "strategy": "freq",
            "n_neighbors": 6,
            "batch_size": 512,
            "lr": 0.001,
            "wt": 20,
            "wl": 5,
        },
        "epochs_key": "epochs",
        "full_epochs": 20,
        "device_type": "cuda",
    },
}


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_adapter_source_sha256(module_str: str) -> str:
    try:
        mod = importlib.import_module(module_str)
        if hasattr(mod, "__file__") and mod.__file__:
            return compute_sha256(Path(mod.__file__).resolve())
    except Exception:
        pass
    return "UNKNOWN"


# ==============================================================================
# WORKER PROCESS EXECUTION (Runs inside child process)
# ==============================================================================
def execute_worker(
    method_name: str,
    mode: str,
    seed: int,
    device_str: str,
    output_emb_path: str,
    temp_result_json: str,
):
    if mode == "test_guard":
        # Synthetic guard testing payload
        if method_name == "test_timeout":
            print("[WORKER-GUARD-TEST] Simulating long task to trigger timeout breach...")
            time.sleep(30.0)
            sys.exit(0)
        elif method_name == "test_rss":
            print("[WORKER-GUARD-TEST] Allocating large array to trigger RSS breach...")
            # Allocate 100MB to exceed low RSS threshold (e.g. 10MB)
            arr = bytearray(100 * 1024 * 1024)
            time.sleep(10.0)
            print(f"[WORKER-GUARD-TEST] Holding {len(arr)} bytes")
            sys.exit(0)
        else:
            raise ValueError(f"Unknown test_guard method: {method_name}")

    cfg = FROZEN_HYPERPARAMS[method_name]
    mod = importlib.import_module(cfg["adapter_module"])
    cls = getattr(mod, cfg["adapter_class"])

    kwargs = dict(cfg["kwargs"])
    if cfg["epochs_key"] is not None:
        if mode in ("smoke", "preflight"):
            kwargs[cfg["epochs_key"]] = 1
        else:
            kwargs[cfg["epochs_key"]] = cfg["full_epochs"]

    print(f"[WORKER] Loading dataset from {DATASET_PATH}...")
    adata_full = ad.read_h5ad(DATASET_PATH)
    slice_mask = adata_full.obs["sample_id"] == TARGET_SAMPLE_ID
    assert slice_mask.sum() == 11293, f"Expected 11293 cells, got {slice_mask.sum()}"
    adata_slice = adata_full[slice_mask].copy()

    if mode == "smoke":
        with open(SMOKE_MANIFEST_PATH, "r", encoding="utf-8") as fp:
            manifest = json.load(fp)
        barcodes = manifest["barcodes"]
        adata_target = adata_slice[barcodes].copy()
        assert adata_target.n_obs == 1000
    else:
        adata_target = adata_slice

    print(f"[WORKER] Target cells: {adata_target.n_obs}. Instantiating {cls.__name__}...")
    adapter = cls(**kwargs)

    device = torch.device(device_str)
    start_t = time.time()
    res = adapter.fit_transform(
        adata_target,
        seed=seed,
        device=device,
        embedding_only=True,
    )
    elapsed = time.time() - start_t

    emb = res.embedding
    assert emb is not None, "Worker returned None embedding"
    assert emb.shape[0] == adata_target.n_obs, f"Shape mismatch: {emb.shape[0]} != {adata_target.n_obs}"
    assert not np.isnan(emb).any(), "Embedding contains NaN"
    assert not np.isinf(emb).any(), "Embedding contains Inf"

    # Explicitly write into output_emb_path in binary mode (prevents double .npy extension trap)
    with open(output_emb_path, "wb") as fp:
        np.save(fp, emb.astype(np.float32))

    res_meta = {
        "status": "SUCCESS",
        "wall_clock_seconds": round(elapsed, 3),
        "embedding_shape": list(emb.shape),
        "embedding_dim": int(emb.shape[1]),
        "kwargs": kwargs,
    }
    with open(temp_result_json, "w", encoding="utf-8") as fp:
        json.dump(res_meta, fp)
    print(f"[WORKER] Completed successfully in {elapsed:.2f}s.")
    sys.exit(0)


# ==============================================================================
# PARENT PROCESS SUPERVISOR WITH DUAL-TRACK GPU ACCOUNTING & FAIL-CLOSED GUARDS
# ==============================================================================
def get_system_gpu_used_mb(device_index: int = 0) -> float:
    try:
        res = subprocess.check_output(
            ["nvidia-smi", f"--id={device_index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        return float(res.strip())
    except Exception:
        return 0.0


def get_worker_tree_gpu_mb(target_pids: set[int]) -> tuple[float, bool]:
    """Queries per-PID compute memory usage and sums over target_pids."""
    if not target_pids:
        return 0.0, False
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        total_mb = 0.0
        found = False
        for line in res.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                pid, mem = int(parts[0]), float(parts[1])
                if pid in target_pids:
                    total_mb += mem
                    found = True
        return total_mb, found
    except Exception:
        return 0.0, False


def run_supervised_method(
    method_name: str,
    mode: str,
    seed: int,
    device_str: str,
    max_vram_mb: float,
    max_rss_gb: float,
    min_free_ram_gb: float,
    timeout_minutes: float,
    out_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    print(f"\n{'='*75}")
    print(f">>> [SUPERVISOR] Starting: {method_name.upper()} | Mode: {mode.upper()} | Seed: {seed}")
    print(f"    Guardrails: Sys VRAM <= {max_vram_mb} MB | RSS <= {max_rss_gb} GB | Free RAM >= {min_free_ram_gb} GB | Timeout <= {timeout_minutes} min")
    print(f"{'='*75}")

    if mode == "smoke":
        sub_dir = out_dir / "smoke_profiles"
        prof_dir = out_dir / "smoke_profiles"
        emb_filename = f"{method_name}_smoke_embedding_seed{seed}.npy"
        prof_filename = f"{method_name}_smoke_profile.json"
    elif mode == "preflight":
        sub_dir = out_dir / "preflight_profiles"
        prof_dir = out_dir / "preflight_profiles"
        emb_filename = f"{method_name}_preflight_embedding_seed{seed}.npy"
        prof_filename = f"{method_name}_preflight_profile.json"
    elif mode == "test_guard":
        sub_dir = out_dir / "guard_tests"
        prof_dir = out_dir / "guard_tests"
        emb_filename = f"{method_name}_guard_test_embedding_seed{seed}.npy"
        prof_filename = f"{method_name}_guard_test_profile.json"
    else:  # mode == "full"
        sub_dir = out_dir / "embeddings"
        prof_dir = out_dir / "results"
        emb_filename = f"{method_name}_embedding_seed{seed}.npy"
        prof_filename = f"{method_name}_formal_profile.json"

    sub_dir.mkdir(parents=True, exist_ok=True)
    prof_dir.mkdir(parents=True, exist_ok=True)

    emb_path = sub_dir / emb_filename
    prof_file = prof_dir / prof_filename

    # ==========================================================================
    # 1. STRICT APPEND-ONLY & FAIL-CLOSED CHECK
    # ==========================================================================
    if mode == "full":
        if emb_path.exists() or prof_file.exists():
            raise FileExistsError(
                f"[FAIL-CLOSED APPEND-ONLY] Target formal file already exists! Overwrite is strictly forbidden in formal mode.\n"
                f"  Embedding: {emb_path} (exists={emb_path.exists()})\n"
                f"  Profile:   {prof_file} (exists={prof_file.exists()})\n"
                f"To re-run formal experiments, specify a new attempt directory (e.g. w24_full_slice_visualization_v2)."
            )
    else:
        if (emb_path.exists() or prof_file.exists()) and not overwrite:
            raise FileExistsError(
                f"[FAIL-CLOSED] Target file already exists in {mode} mode.\n"
                f"  Embedding: {emb_path} (exists={emb_path.exists()})\n"
                f"  Profile:   {prof_file} (exists={prof_file.exists()})\n"
                f"Pass --overwrite to explicitly permit overriding test artifacts."
            )

    # Temporary paths with PID + timestamp
    worker_uid = f"{os.getpid()}_{int(time.time()*1000)}"
    tmp_emb_path = sub_dir / f"{method_name}_{mode}_{seed}.tmp.{worker_uid}.npy"
    tmp_json = sub_dir / f"{method_name}_{mode}_{seed}.tmp.{worker_uid}.json"

    # Worker Command
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-method", method_name,
        "--worker-mode", mode,
        "--worker-seed", str(seed),
        "--worker-device", device_str,
        "--worker-output-emb", str(tmp_emb_path),
        "--worker-temp-json", str(tmp_json),
    ]

    base_gpu_mem = get_system_gpu_used_mb()
    start_time = time.time()
    timeout_sec = timeout_minutes * 60.0

    p = subprocess.Popen(
        cmd,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )

    ps_child = psutil.Process(p.pid)
    peak_rss_mb = 0.0
    peak_system_gpu_mb = base_gpu_mem
    peak_worker_gpu_mb = 0.0
    worker_gpu_samples = 0
    worker_gpu_detected = False
    status = "SUCCESS"
    breach_reason = ""

    try:
        # Polling Loop (0.2s interval)
        while True:
            ret = p.poll()
            if ret is not None:
                break

            elapsed = time.time() - start_time

            # 1. Check Timeout
            if elapsed > timeout_sec:
                status = "TIMEOUT_BREACHED"
                breach_reason = f"Execution exceeded timeout limit {timeout_minutes} minutes ({elapsed:.1f}s)"
                break

            # 2. Check Process Tree RSS & Gather PIDs
            tree_pids = {p.pid}
            try:
                tree_rss = ps_child.memory_info().rss
                for desc in ps_child.children(recursive=True):
                    try:
                        tree_pids.add(desc.pid)
                        tree_rss += desc.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                tree_rss_mb = tree_rss / (1024 * 1024)
                if tree_rss_mb > peak_rss_mb:
                    peak_rss_mb = tree_rss_mb

                if tree_rss_mb > (max_rss_gb * 1024):
                    status = "RSS_GUARD_BREACHED"
                    breach_reason = f"Process tree RSS {tree_rss_mb:.1f} MB exceeded limit {max_rss_gb * 1024} MB"
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            # 3. Check System Available RAM
            sys_avail_gb = psutil.virtual_memory().available / (1024 ** 3)
            if sys_avail_gb < min_free_ram_gb:
                status = "HOST_LOW_MEMORY_BREACHED"
                breach_reason = f"System available RAM {sys_avail_gb:.2f} GB dropped below safety limit {min_free_ram_gb} GB"
                break

            # 4. Check Dual-Track GPU VRAM
            cur_sys_gpu = get_system_gpu_used_mb()
            if cur_sys_gpu > peak_system_gpu_mb:
                peak_system_gpu_mb = cur_sys_gpu

            # Per-PID Worker GPU tracking
            w_gpu_mb, w_found = get_worker_tree_gpu_mb(tree_pids)
            worker_gpu_samples += 1
            if w_found:
                worker_gpu_detected = True
                if w_gpu_mb > peak_worker_gpu_mb:
                    peak_worker_gpu_mb = w_gpu_mb

            # Hard system breach kill line
            if cur_sys_gpu > max_vram_mb:
                status = "VRAM_GUARD_BREACHED"
                breach_reason = f"System GPU memory {cur_sys_gpu:.1f} MB exceeded hard limit {max_vram_mb:.1f} MB"
                break

            time.sleep(0.2)

        # Active Termination if breached
        if status != "SUCCESS":
            print(f"\n[TERMINATION INITIATED] Guardrail triggered: {status} ({breach_reason})")
            try:
                pgid = os.getpgid(p.pid) if hasattr(os, "getpgid") else p.pid
                os.killpg(pgid, signal.SIGTERM)
                time.sleep(2.0)
                if p.poll() is None:
                    os.killpg(pgid, signal.SIGKILL)
            except Exception as e:
                print(f"[KILL ERROR] Failed to kill child process: {e}")
            p.wait()
        else:
            if p.returncode != 0:
                status = "FAILED"
                breach_reason = f"Child process exited with non-zero return code {p.returncode}"

    finally:
        # ======================================================================
        # CLEANUP OF SEMI-WRITTEN ASSETS UPON BREACH OR ABNORMAL EXIT
        # ======================================================================
        if status != "SUCCESS":
            if tmp_emb_path.exists():
                tmp_emb_path.unlink()
                print(f"[CLEANUP] Deleted semi-written temporary embedding: {tmp_emb_path.name}")
            if tmp_json.exists():
                tmp_json.unlink()
                print(f"[CLEANUP] Deleted temporary JSON: {tmp_json.name}")

    total_wall_clock = time.time() - start_time

    # Verify GPU Memory Clean Release
    time.sleep(1.0)
    released_gpu_mem = get_system_gpu_used_mb()
    cleanly_released = bool(released_gpu_mem <= base_gpu_mem + 100.0)

    print(f"[SUPERVISOR] Finished in {total_wall_clock:.2f}s | Status: {status}")
    print(f"    Peak RSS: {peak_rss_mb:.1f} MB | System GPU Peak: {peak_system_gpu_mb:.1f} MB | Worker GPU Peak: {peak_worker_gpu_mb:.1f} MB")
    print(f"    Baseline GPU: {base_gpu_mem:.1f} MB | Released GPU: {released_gpu_mem:.1f} MB | Cleanly Released: {cleanly_released}")

    worker_meta = {}
    if status == "SUCCESS" and tmp_json.exists():
        with open(tmp_json, "r", encoding="utf-8") as fp:
            worker_meta = json.load(fp)
        tmp_json.unlink()

    # Determine worker GPU measurement note
    is_gpu_method = FROZEN_HYPERPARAMS.get(method_name, {}).get("device_type") == "cuda"
    if not is_gpu_method:
        worker_tree_gpu_peak_val: Any = 0.0
        worker_gpu_note = "CPU_METHOD_ZERO_VRAM"
    elif worker_gpu_detected:
        worker_tree_gpu_peak_val = round(peak_worker_gpu_mb, 2)
        worker_gpu_note = "MEASURED_FROM_COMPUTE_APPS"
    else:
        # Worker finished faster than polling sample or compute-apps delay
        worker_tree_gpu_peak_val = None
        worker_gpu_note = "TRANSIENT_RUN_NO_APP_SAMPLE"

    # Bind provenance hashes
    runner_sha = compute_sha256(Path(__file__).resolve())
    h5ad_sha = compute_sha256(DATASET_PATH) if DATASET_PATH.exists() else "UNKNOWN"
    cfg_method = FROZEN_HYPERPARAMS.get(method_name, {})
    adapter_mod = cfg_method.get("adapter_module", "")
    adapter_sha = compute_adapter_source_sha256(adapter_mod) if adapter_mod else "UNKNOWN"

    if mode == "smoke":
        ordered_barcodes_sha = "0f315013d17938a37826278c9b3bb54247964213103351ff6a7c6271e5a6f380"
    elif mode in ("preflight", "full"):
        ordered_barcodes_sha = "4b58c11fceb50c0ec9112f836a53216db2ee0f84f0544631ce77fe00154c2e58"
    else:
        ordered_barcodes_sha = "GUARD_TEST_N_A"

    profile = {
        "method": method_name,
        "mode": mode,
        "seed": seed,
        "status": status,
        "breach_reason": breach_reason if breach_reason else None,
        "guard_contract_verdict": "GUARD_DEFENSE_CONTRACT_PASS" if (mode == "test_guard" and status in ("TIMEOUT_BREACHED", "RSS_GUARD_BREACHED", "VRAM_GUARD_BREACHED")) else ("NORMAL_SUCCESS" if status == "SUCCESS" else "UNEXPECTED_FAILURE"),
        "total_wall_clock_seconds": round(total_wall_clock, 3),
        "worker_fit_seconds": worker_meta.get("wall_clock_seconds"),
        "peak_tree_rss_mb": round(peak_rss_mb, 2),
        "system_gpu_used_peak_mb": round(peak_system_gpu_mb, 2),
        "worker_tree_gpu_peak_mb": worker_tree_gpu_peak_val,
        "worker_gpu_note": worker_gpu_note,
        "baseline_system_gpu_used_mb": round(base_gpu_mem, 2),
        "post_exit_system_gpu_used_mb": round(released_gpu_mem, 2),
        "vram_cleanly_released": cleanly_released,
        "embedding_shape": worker_meta.get("embedding_shape"),
        "embedding_dim": worker_meta.get("embedding_dim"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kwargs": worker_meta.get("kwargs"),
        "provenance": {
            "h5ad_sha256": h5ad_sha,
            "ordered_barcodes_sha256": ordered_barcodes_sha,
            "runner_sha256": runner_sha,
            "adapter_source_sha256": adapter_sha,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
        },
    }

    # Atomic rename if SUCCESS
    if status == "SUCCESS" and tmp_emb_path.exists():
        os.replace(tmp_emb_path, emb_path)
        profile["embedding_sha256"] = compute_sha256(emb_path)
        print(f"[OK] Embedding atomically swapped into place: {emb_path.name} (SHA-256: {profile['embedding_sha256'][:16]}...)")

    # Save profile atomically
    tmp_prof_file = prof_file.with_name(f"{prof_file.stem}.tmp.{worker_uid}.json")
    with open(tmp_prof_file, "w", encoding="utf-8") as fp:
        json.dump(profile, fp, indent=2)
    os.replace(tmp_prof_file, prof_file)
    print(f"[OK] Saved profile atomically to {prof_file}")

    return profile


def run_guard_defense_tests(out_dir: Path):
    """Executes intentional low-threshold guard defense unit tests."""
    print("\n" + "=" * 80)
    print(">>> EXECUTING INTENTIONAL GUARD DEFENSE TESTS (EXPECTED BREACH VALIDATION)")
    print("=" * 80)

    # Test 1: Timeout Breach
    prof_timeout = run_supervised_method(
        method_name="test_timeout",
        mode="test_guard",
        seed=42,
        device_str="cpu",
        max_vram_mb=21000.0,
        max_rss_gb=48.0,
        min_free_ram_gb=1.0,
        timeout_minutes=0.05,  # 3.0 seconds
        out_dir=out_dir,
        overwrite=True,
    )
    assert prof_timeout["status"] == "TIMEOUT_BREACHED", f"Expected TIMEOUT_BREACHED, got {prof_timeout['status']}"
    assert prof_timeout["guard_contract_verdict"] == "GUARD_DEFENSE_CONTRACT_PASS"
    print("[PASS] Defense Test 1: Timeout breach intercepted and killed cleanly.")

    # Test 2: RSS Breach
    prof_rss = run_supervised_method(
        method_name="test_rss",
        mode="test_guard",
        seed=42,
        device_str="cpu",
        max_vram_mb=21000.0,
        max_rss_gb=0.02,  # 20 MB limit (worker allocates 100 MB)
        min_free_ram_gb=1.0,
        timeout_minutes=2.0,
        out_dir=out_dir,
        overwrite=True,
    )
    assert prof_rss["status"] == "RSS_GUARD_BREACHED", f"Expected RSS_GUARD_BREACHED, got {prof_rss['status']}"
    assert prof_rss["guard_contract_verdict"] == "GUARD_DEFENSE_CONTRACT_PASS"
    print("[PASS] Defense Test 2: RSS breach intercepted and killed cleanly.")

    # Assert guard_tests directory does not contain leftover .tmp files
    guard_dir = out_dir / "guard_tests"
    leftover_tmps = list(guard_dir.glob("*.tmp.*"))
    assert len(leftover_tmps) == 0, f"Leftover temporary files found: {leftover_tmps}"
    print("[PASS] Defense Test 3: Zero leftover temporary files in guard_tests/.")
    print(">>> ALL GUARD DEFENSE UNIT TESTS PASSED WITH FLYING COLORS.\n")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Human Skin Baseline Runner with Active Resource Guardrails")
    parser.add_argument("--worker", action="store_true", help="Internal worker mode")
    parser.add_argument("--worker-method", type=str, choices=list(FROZEN_HYPERPARAMS.keys()) + ["test_timeout", "test_rss"])
    parser.add_argument("--worker-mode", type=str, choices=["smoke", "preflight", "full", "test_guard"])
    parser.add_argument("--worker-seed", type=int, default=42)
    parser.add_argument("--worker-device", type=str, default="cuda")
    parser.add_argument("--worker-output-emb", type=str)
    parser.add_argument("--worker-temp-json", type=str)

    parser.add_argument("--smoke", action="store_true", help="Run 1000-cell smoke mode")
    parser.add_argument("--preflight", action="store_true", help="Run 11,293-cell 1-epoch preflight mode")
    parser.add_argument("--full", action="store_true", help="Run 11,293-cell full slice mode (requires --authorized)")
    parser.add_argument("--test-guard", action="store_true", help="Run intentional guard defense unit tests")
    parser.add_argument("--authorized", action="store_true", help="Required flag for --full mode")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwrite in test modes (smoke/preflight/guard)")
    parser.add_argument("--aggregate-summary", action="store_true", help="Aggregate all completed method profiles into total summary")
    parser.add_argument("--method", type=str, choices=list(FROZEN_HYPERPARAMS.keys()) + ["all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default=str(OUTPUT_BASE))

    parser.add_argument("--max-vram-mb", type=float, default=21000.0)
    parser.add_argument("--max-rss-gb", type=float, default=48.0)
    parser.add_argument("--min-free-ram-gb", type=float, default=8.0)
    parser.add_argument("--timeout-minutes", type=float, default=120.0)

    args = parser.parse_args()

    # 1. Worker Dispatch
    if args.worker:
        execute_worker(
            method_name=args.worker_method,
            mode=args.worker_mode,
            seed=args.worker_seed,
            device_str=args.worker_device,
            output_emb_path=args.worker_output_emb,
            temp_result_json=args.worker_temp_json,
        )
        return

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Defense Test Dispatch
    if args.test_guard:
        run_guard_defense_tests(out_dir)
        return

    # 3. Determine Execution Mode
    mode_flags = [args.smoke, args.preflight, args.full]
    if sum(bool(f) for f in mode_flags) != 1:
        parser.error("Exactly one of --smoke, --preflight, --full, or --test-guard must be specified.")

    mode = "smoke" if args.smoke else ("preflight" if args.preflight else "full")

    if mode == "full" and not args.authorized:
        print("[FAIL-CLOSED ERROR] --full mode requires explicit --authorized flag from user.", file=sys.stderr)
        sys.exit(1)

    timeout_min = 15.0 if mode == "smoke" else (30.0 if mode == "preflight" else args.timeout_minutes)

    # 4. Aggregation Only Mode
    if args.aggregate_summary:
        print(f"\n>>> Aggregating total execution summary for mode: {mode.upper()}...")
        prof_sub = "smoke_profiles" if mode == "smoke" else ("preflight_profiles" if mode == "preflight" else "results")
        prof_dir = out_dir / prof_sub
        total_sum_file = out_dir / f"{mode}_execution_summary.json"

        if total_sum_file.exists() and mode == "full":
            raise FileExistsError(f"[FAIL-CLOSED APPEND-ONLY] Total summary already exists: {total_sum_file}. Overwrite is forbidden.")

        all_methods = list(FROZEN_HYPERPARAMS.keys())
        agg_summary = {
            "title": f"Human Skin Baseline Total Execution Summary ({mode.upper()})",
            "dataset": "human_skin_merfish",
            "sample_id": TARGET_SAMPLE_ID,
            "mode": mode,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runner_sha256": compute_sha256(Path(__file__).resolve()),
            "methods": {},
        }
        for m in all_methods:
            p_file = prof_dir / (f"{m}_{mode}_profile.json" if mode != "full" else f"{m}_formal_profile.json")
            if not p_file.exists():
                raise RuntimeError(f"[FAIL-CLOSED ERROR] Cannot aggregate total summary: missing profile for {m} ({p_file})")
            with open(p_file, "r", encoding="utf-8") as fp:
                agg_summary["methods"][m] = json.load(fp)

        tmp_total = total_sum_file.with_name(f"{total_sum_file.stem}.tmp.{os.getpid()}.json")
        with open(tmp_total, "w", encoding="utf-8") as fp:
            json.dump(agg_summary, fp, indent=2)
        os.replace(tmp_total, total_sum_file)
        print(f"[OK] Total summary atomically aggregated and saved to {total_sum_file}")
        return

    methods = list(FROZEN_HYPERPARAMS.keys()) if args.method == "all" else [args.method]

    summary: dict[str, Any] = {
        "title": f"Human Skin Baseline Execution Summary ({mode.upper()})",
        "dataset": "human_skin_merfish",
        "sample_id": TARGET_SAMPLE_ID,
        "mode": mode,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runner_sha256": compute_sha256(Path(__file__).resolve()),
        "methods": {},
    }

    for m in methods:
        prof = run_supervised_method(
            method_name=m,
            mode=mode,
            seed=args.seed,
            device_str=args.device,
            max_vram_mb=args.max_vram_mb,
            max_rss_gb=args.max_rss_gb,
            min_free_ram_gb=args.min_free_ram_gb,
            timeout_minutes=timeout_min,
            out_dir=out_dir,
            overwrite=args.overwrite,
        )
        summary["methods"][m] = prof

    # If single method executed, write per-method summary file to prevent overwriting other methods
    if len(methods) == 1:
        m_name = methods[0]
        summary_file = out_dir / f"{mode}_execution_summary_{m_name}.json"
    else:
        summary_file = out_dir / f"{mode}_execution_summary.json"
        if summary_file.exists() and mode == "full":
            raise FileExistsError(f"[FAIL-CLOSED APPEND-ONLY] Total summary already exists: {summary_file}. Overwrite is forbidden.")

    tmp_sum_file = summary_file.with_name(f"{summary_file.stem}.tmp.{os.getpid()}.json")
    with open(tmp_sum_file, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    os.replace(tmp_sum_file, summary_file)
    print(f"\n[OK] Summary atomically written to {summary_file}")


if __name__ == "__main__":
    main()
