# Manuscript experiments and downstream analyses

This directory follows the main-text and supplementary experiments referenced by `Paper/FreqNiche-zh/main.tex`. Tasks use descriptive scientific names rather than A/B/C letter identifiers.

## Task names and launch commands

| Manuscript analysis | `--tasks` option | Code directory | Dataset |
| --- | --- | --- | --- |
| Neighborhood cell-composition prediction | `neighborhood_composition_prediction` | `neighborhood_composition_prediction/` | Human Skin |
| Cortical-depth prediction (continuous anatomical gradient) | `cortical_depth_prediction` | `cortical_depth_prediction/` | SEA-AD |
| Within-cell-type niche-associated expression analysis | `niche_associated_expression` | `niche_associated_expression/` | Human Skin Fibro |

Cross-donor spatial-domain identification (SEA-AD) and cellular-niche identification (Human Skin/CosMx) are already included in model Test evaluation through `scripts/benchmark/`; no additional prediction task needs to be selected. Unlabeled Xenium evaluation is also part of the model evaluation workflow.

Install the environment, download the inputs following the [data guide](../datasets/README.md), and run from the repository root:

```bash
# Training, Test inference, neighborhood composition and niche-associated expression
python reproduce.py --dataset human_skin --tasks neighborhood_composition_prediction niche_associated_expression
# Training, Test inference and cortical-depth prediction
python reproduce.py --dataset sea_ad --tasks cortical_depth_prediction
# Inspect the command plan without training or writing results
python reproduce.py --dataset human_skin --tasks neighborhood_composition_prediction --dry-run
```

To analyze a completed run separately, replace `DATE/RUN_ID` with the actual directory printed by the launcher:

```bash
python reproduce.py --stage downstream --run-dir outputs/runs/freqniche/human_skin/DATE/RUN_ID --tasks neighborhood_composition_prediction niche_associated_expression
python reproduce.py --stage downstream --run-dir outputs/runs/freqniche/sea_ad/DATE/RUN_ID --tasks cortical_depth_prediction
```

## Independent scripts and outputs

Each task directory provides a `run_<task_name>.py` entry point that can also be invoked directly:

```bash
python experiments_analysis/neighborhood_composition_prediction/run_neighborhood_composition_prediction.py --run-dir RUN
python experiments_analysis/cortical_depth_prediction/run_cortical_depth_prediction.py --run-dir RUN
python experiments_analysis/niche_associated_expression/run_niche_associated_expression.py --run-dir RUN
```

Inputs are defined by the training run's exported `downstream_inputs.json`, which binds the H5AD data, model configuration, barcodes and embeddings through verification checks. New results are written to the run's `downstream/neighborhood_composition_prediction/`, `downstream/cortical_depth_prediction/` or `downstream/niche_associated_expression/` directory. All are ignored by Git.

Both prediction tasks retain cross-validation grouped by Validation donor. Standardization and PCA are fitted only within the corresponding training fold, and Validation and Test donors must not overlap. Normal computation refuses to overwrite existing results; `--plot-only` redraws results associated with the same run.

## Niche-associated expression analysis

This analysis corresponds to the manuscript's within-cell-type niche-associated expression section. It retains the Fibro cell type, Seed 42, Leiden settings, raw-count checks, GMT hashes, donor-level expression effects, pseudobulk changes and pathway-overlap calculations. Numerical routines are implemented in `niche_associated_expression/expression_analysis.py`.

The comparison is between STROMA-enriched and PERIVASC-enriched groups, with default cluster IDs 2 and 0. New training can permute these IDs. Inspect the reported niche proportions to establish the biological roles, and specify them using `--expression-stroma-cluster` and `--expression-perivasc-cluster`. The independent script uses `--stroma-cluster` and `--perivasc-cluster`. Each group must have a majority of its corresponding reference niche, and at least three donors must contribute at least 20 Fibro cells to each group. The analysis stops if these conditions fail; parameters must not be adjusted to obtain favorable effects.

The [analysis protocol](protocols/niche_associated_expression_protocol.md) records the statistical definitions for the original manuscript comparison. The effective sample size is the number of donors, and the analysis is descriptive and post hoc. New runs record their actual cluster IDs, cell counts, donor sets and embedding dimensions rather than reusing historical values.

## Other code used in the main text and supplementary material

| Content | Retained code | Manuscript use |
| --- | --- | --- |
| Key design ablations | `ablation/` | Main-text design ablation analysis |
| Clusterer sensitivity and reference-class-count diagnostics | `clusterer_sensitivity/` | Supplementary clusterer sensitivity experiments |
| Xenium frozen inference time and GPU memory | `benchmark_xenium_frozen_inference.py` | Supplementary unlabeled evaluation and scalability |
| Linear-control verification | `audit_human_skin_linear_controls.py` | Raw PCA and DualFreq-PCA controls |
| Full-slice background and inference assets | `infer_full_slice/` | Inputs to main-text physical-section figures |
| Manuscript figures and shared styles | `plot_figures/`, `figure_style.py` | Main-text and supplementary figures |
| Manuscript table preparation | `prepare_014_paper_inputs.py`, `format_paper_table_ranks.py` | Table input preparation and formatting |

Additional audits and manuscript plotting scripts may require existing baseline embeddings, selection manifests, result tables or R/mclust. These assets are not distributed with the code and are not automatically generated by the three prediction/expression tasks.

Transductive adaptation workflows that were not used in the manuscript, unimplemented graph/expression perturbation proposals, duplicate plotting entry points, additional development tests and duplicate expression launchers tied to historical assets have been removed. Shared plotting functions and expression-effect calculations required by the retained figures remain available.

This organization preserves the manuscript's experimental scope without editing the manuscript itself. Command-plan and synthetic-data checks do not replace a full-scale three-seed rerun.
