# 聚类算法敏感性实验

对应论文补充材料“聚类算法敏感性实验”，保留归纳式 FreqNiche、CellCharter、BANKSY、STAGATE 与 CellNiche 的冻结表征。未纳入论文的 FreqNiche 转导适配分支已移除，不再要求其选择或适配清单。

## 协议

- 比较 K-means、R mclust EEE、Louvain、Leiden。
- 主读出通过验证集共享十个 80% 子采样的无标签稳定性选择参数；并列时选较小参数，再冻结到测试集。
- K-means/mclust 的 K/G 候选为 5–25；Louvain/Leiden 分辨率候选为 0.1–2.0，近邻数为 15。
- mclust 只在验证表征上拟合 PCA20，冻结投影应用于测试；实际非空簇数可以小于请求的 G。
- 参考类别数诊断固定目标 K=10；图聚类只按实际簇数接近程度选择分辨率，ARI/NMI 不参与选择。它与主无标签读出分开报告。
- 全部四种聚类器均须报告；不得只保留最高分。环境核对使用根目录 `requirements.txt`，并记录 R/mclust 版本。

## 执行

该补充分析仍使用 `clusterer_common.py` 中登记的历史冻结表征、条码和 bootstrap 清单，不属于 `reproduce.py` 的三个预测/表达任务。需要另行准备对应运行资产；R 必须安装 `mclust`。

在项目根目录运行，`--suite freqniche` 为论文中的归纳模型，`--suite baselines` 为基线：

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

合并前检查两端算法环境和输入签名。已有结果不覆盖；历史报告中的额外转导分支在论文输出中排除，原始报告文件保持只读。
