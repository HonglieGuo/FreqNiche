#!/usr/bin/env bash
set -euo pipefail

# Canonical baseline launcher.
# Comment out datasets or methods to exclude them; uncomment to restore them.
DATASETS=(
  # "human_skin"
  # "sea_ad"
  "cosmx_nsclc"
  # "xenium_prime5k"
)

METHODS=(
  # "banksy"
  # "cellcharter"
  # "cellniche"
  # "stagate"
  "sedr"     # Official SEDR core; bounded donor-wise transductive track.
  "graphst"  # Official GraphST core; bounded donor-wise transductive track.
  "spagcn"   # Official SpaGCN core; needs isolated Python with louvain.
)

SEEDS=(
  42
  123
  2024
)

PYTHON="${PYTHON:-$HOME/anaconda3/envs/FreqNiche/bin/python}"
DEVICE="${DEVICE:-cuda}"
TAG="${TAG:-final}"
DRY_RUN="${DRY_RUN:-false}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
OUTPUT_ROOT="${OUTPUT_ROOT:-baselines/Baselines_outputs/runs}"

# Mirror the live terminal (stdout and stderr) to one root-level baseline log.
RUN_LOG="${ROOT}/baselines/run_baselines_$(date +%Y%m%d-%H%M%S)_$$.txt"
on_exit() {
  status=$?
  trap - EXIT
  if (( status == 0 )); then
    if [[ "${PREFLIGHT_ONLY}" == "true" || "${DRY_RUN}" == "true" ]]; then
      printf '[FINISHED] Baseline checks/planning completed; no models were trained.\n'
    else
      printf '[FINISHED] Selected baseline runs completed.\n'
    fi
  else
    printf '[FAILED] Exit code: %s. Review the error above.\n' "${status}"
  fi
  printf '[LOG] %s\n' "${RUN_LOG}"
  exit "${status}"
}
trap on_exit EXIT
exec > >(tee -a "${RUN_LOG}") 2>&1
printf '[STARTED] %s | host=%s | pid=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$(hostname)" "$$"
printf '[LOG] %s\n' "${RUN_LOG}"

# Fail before any data preparation if SpaGCN's official Louvain path is selected
# without its isolated dependency overlay. Planning/dry-run needs no packages.
if [[ "${DRY_RUN}" != "true" || "${PREFLIGHT_ONLY}" == "true" ]]; then
  for method in "${METHODS[@]}"; do
    if [[ "${method}" == "spagcn" ]]; then
      spatial_python="${FREQNICHE_SPATIAL_BASELINE_PYTHON:-$HOME/.venvs/FreqNiche-spatial-baselines/bin/python}"
      if [[ ! -x "${spatial_python}" ]] || ! "${spatial_python}" -c 'import louvain, igraph, scanpy, torch' >/dev/null 2>&1; then
        echo "[PRE-FLIGHT FAILED] SpaGCN requires an isolated interpreter with louvain."
        echo "Create it as documented in baselines/README.md, then rerun."
        exit 2
      fi
      export FREQNICHE_SPATIAL_BASELINE_PYTHON="${spatial_python}"
      echo "[PRE-FLIGHT PASS] SpaGCN Python: ${spatial_python}"
      break
    fi
  done
fi
if [[ "${PREFLIGHT_ONLY}" == "true" ]]; then
  echo "[PRE-FLIGHT COMPLETE] No dataset preparation or model run was started."
  exit 0
fi

for dataset in "${DATASETS[@]}"; do
  available=()
  for method in "${METHODS[@]}"; do
    case "${dataset}:${method}" in
      human_skin:*) available+=("${method}") ;;
      xenium_prime5k:cellniche|xenium_prime5k:sedr|xenium_prime5k:graphst|xenium_prime5k:spagcn) ;;
      *) available+=("${method}") ;;
    esac
  done

  echo "=================================================================="
  echo "Baselines dataset: ${dataset} | methods: ${available[*]}"
  echo "=================================================================="
  args=(
    "${PYTHON}" baselines/run_baselines.py
    --dataset "${dataset}"
    --methods "${available[@]}"
    --seeds "${SEEDS[@]}"
    --device "${DEVICE}"
    --tag "${TAG}"
    --output-root "${OUTPUT_ROOT}"
  )
  if [[ "${DRY_RUN}" == "true" ]]; then
    args+=(--dry-run)
  fi
  "${args[@]}"
done
