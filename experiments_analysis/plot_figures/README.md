# 论文图件代码

本目录按当前论文正文和补充材料保留绘图代码。图件从相应冻结结果、嵌入或缓存坐标生成；结果资产不随代码上传。本文档所列图名对应 `Paper/FreqNiche-zh/main.tex` 及其实际引用的章节。

| 论文图件 | 主要脚本 |
| --- | --- |
| 正文：SEA-AD 跨供体空间域识别 | `plot_seaad_spatial_domains.py` |
| 正文：Human Skin 跨供体细胞生态位识别 | `plot_human_skin_comparison.py` |
| 正文：邻域细胞组成与连续皮层深度预测 | `plot_combined_prediction_probes.py`；组成部分复用 `plot_neighborhood_composition.py` |
| 正文：成纤维细胞生态位相关表达分析 | `plot_supp_fibro_niche_expression.py` |
| 补充：全测试集 UMAP、同供体 UMAP、表征多视角分析、多数据集综合比较 | `restyle_s5_supplementary_analysis.py`，及对应输入生成脚本 `plot_supp_all_methods_panels.py`、`plot_human_skin_comparison.py`、`plot_cross_platform_benchmarks.py` |
| 补充：聚类算法敏感性和类别数诊断 | `restyle_clusterer_and_xenium_analysis.py`、`../clusterer_sensitivity/plot_clusterer_results.py` |
| 补充：Xenium 无标签表征与资源评估 | `plot_xenium_benchmark_panel.py`、`restyle_clusterer_and_xenium_analysis.py` |

`human_skin_plot_utils.py` 保留仍被正文图件调用的调色、散点、图例和标注函数。旧六面板绘图入口、重复转发入口和论文未引用的全切片比较图、额外 014 表征候选图已删除。

典型命令（从项目根目录执行）：

```bash
python experiments_analysis/plot_figures/plot_seaad_spatial_domains.py --help
python experiments_analysis/plot_figures/plot_human_skin_comparison.py --help
python experiments_analysis/plot_figures/plot_combined_prediction_probes.py --help
```

脚本的默认输入保留生成论文图件时的冻结运行路径。重新训练的预测/表达分析图件由各任务直接写入 `<run>/downstream/<任务名称>/figures/`；历史论文排版脚本不自动将新运行替换成旧结果，也不保证无需补充资产即可重绘全部论文图件。部分排版脚本需要可用的 Times New Roman 字体。
