# Clusterer sensitivity experiments

This directory implements the supplementary clusterer sensitivity analysis using frozen representations from inductive FreqNiche, CellCharter, BANKSY, STAGATE and CellNiche. The FreqNiche transductive-adaptation branch was not included in the manuscript and has been removed; its selection and adaptation manifests are no longer required.

## Protocol

- Compare K-means, R mclust EEE, Louvain and Leiden.
- Select primary-readout parameters by unlabeled stability on ten shared 80% Validation subsamples. Break ties by choosing the smaller parameter, then freeze it for Test.
- Candidate K/G values for K-means/mclust are 5-25. Candidate Louvain/Leiden resolutions are 0.1-2.0, with 15 nearest neighbors.
- Fit PCA20 for mclust on Validation representations only and apply the frozen projection to Test. The actual number of nonempty clusters may be smaller than the requested G.
- The reference-class-count diagnostic fixes target K=10. Graph-clustering resolutions are selected solely by proximity of the actual cluster count to that target; ARI/NMI are not used. Report this diagnostic separately from the primary unlabeled readout.
- Report all four clusterers rather than retaining only the highest-scoring one. Check the environment against the root `requirements.txt` and record the R/mclust versions.

## Execution

This supplementary analysis uses the historical frozen representations, barcodes and bootstrap manifests registered in `clusterer_common.py`. It is separate from the three prediction/expression tasks in `reproduce.py`. Prepare the corresponding run assets separately and install the R package `mclust`.

Run from the repository root. `--suite freqniche` selects the manuscript's inductive model, and `--suite baselines` selects baseline methods:

```bash
python experiments_analysis/clusterer_sensitivity/preflight_clusterer_environment.py --suite freqniche
python experiments_analysis/clusterer_sensitivity/select_clusterer_hyperparameters.py --suite freqniche
python experiments_analysis/clusterer_sensitivity/evaluate_clusterers.py --suite freqniche --authorized
python experiments_analysis/clusterer_sensitivity/preflight_clusterer_environment.py --suite baselines
python experiments_analysis/clusterer_sensitivity/select_clusterer_hyperparameters.py --suite baselines
python experiments_analysis/clusterer_sensitivity/evaluate_clusterers.py --suite baselines --authorized
python experiments_analysis/clusterer_sensitivity/merge_clusterer_results.py
python experiments_analysis/clusterer_sensitivity/plot_clusterer_results.py --dry-run
```

Verify both algorithm environments and input signatures before merging. Existing results are not overwritten. Additional transductive branches in historical reports are excluded from manuscript outputs, while the original report files remain read-only.
