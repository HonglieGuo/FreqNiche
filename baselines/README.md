# FreqNiche baseline source registry

This directory contains only the third-party source snapshots used by the
current FreqNiche manuscript: BANKSY, CellCharter and STAGATE-pyG. Additional
bounded spatial-domain controls retain the official SEDR, GraphST and SpaGCN
training cores. CellNiche is installed as a version-pinned Python package.
Do not edit a
vendored source without updating `registry.yaml` and the relevant run manifest.

## Layout policy

- The public code repository excludes generated results, metric tables, logs,
  embeddings and saved models, including `result_baselines.csv`,
  `metrics_manifest.json` and `Baselines_outputs/`. Running the scripts creates
  local outputs; existing result assets must be supplied separately for analyses
  that read previous runs.
- One top-level directory corresponds to one upstream method snapshot.
- Project-owned adapters live in `src/benchmark/adapters/`, not inside vendor
  directories.
- New generated results belong under `baselines/Baselines_outputs/runs/`, and summary rows
  are appended to `baselines/result_baselines.csv`.
- `registry.yaml` is the authoritative inventory for task relevance and
  integration status.
- Presence here does not make a method a valid comparator for every task.

## Canonical launcher

Edit the `DATASETS`, `METHODS` and `SEEDS` arrays in `run_baselines.sh`, then
run it from the repository root. Unsupported dataset/method combinations are
filtered explicitly. Use `DRY_RUN=true` to inspect commands without executing.
Use `PREFLIGHT_ONLY=true bash baselines/run_baselines.sh` to check the selected
SpaGCN interpreter before any data preparation. Every invocation mirrors
stdout/stderr live to `baselines/run_baselines_<timestamp>_<pid>.txt`; step logs
remain under the dated run directory.
The launcher uses the general baseline and CellCharter interpreters recorded in
`registry.yaml`; override them with `FREQNICHE_BASELINE_PYTHON` and
`FREQNICHE_CELLCHARTER_PYTHON` when running on another host.

SEDR, GraphST and SpaGCN are opt-in lines in `run_baselines.sh`. They use the
processed Human Skin, SEA-AD or CosMx H5AD and the same frozen Validation/Test
centers, but fit separately on each donor's selected cells. Results are marked
`transductive_center_only_donor_wise` and must not be mixed into the existing
full-context main ranking. Spatial graphs are sample-isolated, no H&E is used,
and the common Leiden resolution is selected on Validation without labels.
The adapter refuses >10,000 cells or a dense matrix >0.4 GiB. Xenium's
67,705-cell full Test is intentionally unsupported by these native dense
implementations. Their official core can be smoke-tested with
`scripts/benchmark/run_additional_spatial_baselines.py --dataset human_skin
--method sedr --phase smoke --device cpu` (substitute the method as needed).
Formal SpaGCN needs `louvain==0.8.2` with `igraph==0.11.9` in an isolated
environment; set `FREQNICHE_SPATIAL_BASELINE_PYTHON` to its interpreter. A
tested overlay can be created without changing the main FreqNiche environment:

```bash
$HOME/anaconda3/envs/FreqNiche/bin/python -m venv --system-site-packages "$HOME/.venvs/FreqNiche-spatial-baselines"
"$HOME/.venvs/FreqNiche-spatial-baselines/bin/pip" install louvain==0.8.2 igraph==0.11.9
FREQNICHE_BASELINE_PYTHON=$HOME/anaconda3/envs/FreqNiche/bin/python \
FREQNICHE_SPATIAL_BASELINE_PYTHON="$HOME/.venvs/FreqNiche-spatial-baselines/bin/python" \
DRY_RUN=true bash baselines/run_baselines.sh
```

After uncommenting the desired methods, remove `DRY_RUN=true` to execute.
Do not interpret the 2-epoch smoke or the donor-wise center-only results as
the existing full-context main benchmark.

## Reporting policy

Core comparators are selected by task relevance, public availability and input
compatibility before results are read. Successfully executed direct competitors
must be reported in the main text or supplementary material. Technical
exclusions require a reproducible reason; performance is not an exclusion
criterion.

## Environment notes

The root `requirements.txt` records the main WSL FreqNiche environment only.
Separate baseline environment snapshots are no longer distributed; interpreter
locations are recorded in `registry.yaml`. The local CellCharter environment
remains isolated from the general baseline environment.
