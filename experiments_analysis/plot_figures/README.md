# Manuscript figure code

This directory retains plotting scripts used by the current main text and supplementary material. Figures are generated from the corresponding frozen results, embeddings or cached coordinates; result assets are not distributed with the code. Figure names below follow `Paper/FreqNiche-zh/main.tex` and the sections it includes.

| Manuscript figure | Main scripts |
| --- | --- |
| Main text: SEA-AD cross-donor spatial-domain identification | `plot_seaad_spatial_domains.py` |
| Main text: Human Skin cross-donor cellular-niche identification | `plot_human_skin_comparison.py` |
| Main text: neighborhood composition and continuous cortical-depth prediction | `plot_combined_prediction_probes.py`; the composition panels reuse `plot_neighborhood_composition.py` |
| Main text: fibroblast niche-associated expression analysis | `plot_supp_fibro_niche_expression.py` |
| Supplementary: full-Test UMAP, same-donor UMAP, representation diagnostics and multi-dataset comparisons | `restyle_s5_supplementary_analysis.py` and its input-generation scripts `plot_supp_all_methods_panels.py`, `plot_human_skin_comparison.py`, `plot_cross_platform_benchmarks.py` |
| Supplementary: clusterer sensitivity and class-count diagnostics | `restyle_clusterer_and_xenium_analysis.py`, `../clusterer_sensitivity/plot_clusterer_results.py` |
| Supplementary: Xenium unlabeled representations and resource evaluation | `plot_xenium_benchmark_panel.py`, `restyle_clusterer_and_xenium_analysis.py` |

`human_skin_plot_utils.py` retains palette, scatter, legend and annotation functions still used by the main-text figures. The old six-panel plotting entry point, duplicate forwarding entry points, full-slice comparisons not cited by the manuscript and additional 014 representation candidate figures have been removed.

Example commands, run from the repository root:

```bash
python experiments_analysis/plot_figures/plot_seaad_spatial_domains.py --help
python experiments_analysis/plot_figures/plot_human_skin_comparison.py --help
python experiments_analysis/plot_figures/plot_combined_prediction_probes.py --help
```

Default inputs retain the frozen run paths used to generate the manuscript figures. Prediction/expression figures from new training runs are written directly by their task scripts to `<run>/downstream/<task_name>/figures/`. Historical manuscript-layout scripts do not automatically substitute new runs for old results, and regenerating every manuscript figure can require additional assets. Some layout scripts require Times New Roman fonts.
