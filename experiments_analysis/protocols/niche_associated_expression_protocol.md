# 同一细胞类型内的生态位相关表达分析

对应论文实验与结果中的同名小节。以下记录原始比较的分组、描述性效应量和通路重叠口径。新训练的入口为 `niche_associated_expression/run_niche_associated_expression.py`；新运行的供体和簇编号由实际输入及门禁决定。

### 1.1 统计推断边界与数学公理定性
- **样本量约束与无承诺声明**：Human Skin 固定测试队列含 4 个 donor；应用每个 `(donor, cluster, Fibro)` 至少 20 个细胞的门禁后，D028 被排除，实际有效供体仅为 D082、D145、D151，即 $N_{\mathrm{eff}}=3$。在如此有限的 donor 数下，本研究不执行、不承诺供体级显著性发现，也不使用“显著差异基因”措辞。
- **重新定位**：生态位相关表达分析 严格定位为“描述性效应量与跨供体方向一致性分析（Descriptive Effect Size & Directional Concordance Analysis）”。基于 $t$ 分布的 95% 区间只描述效应量不确定性，不作为总体推断或发现声明。

### 1.2 跨供体生态位匹配与最低细胞量门禁
1. **细胞类型与注释字段冻结（唯一指定）**：Human Skin MERFISH 使用唯一字段 `adata.obs['cell_type']`，目标类型固定为 `Fibro`（非 `Fibroblasts`）。
2. **跨供体生态位对应性闭环**：
   - 不能使用任意供体特异的簇号；
   - 使用完整 FreqNiche Trial 24、Seed 42 在 8,000 测试中心上的冻结全局聚类（$r^*=0.8$）；比较 Cluster 2（STROMA-enriched，参考 niche 富集率 58.4%）与 Cluster 0（PERIVASC-enriched，参考 niche 富集率 53.4%）。两组只能称为 enriched cluster，不能等同于纯解剖区域或因果微环境；
3. **有效性过滤门禁（Sample Inclusivity Gate）**：
   - 每一个 `(donor, niche, cell_type)` 单元，必须满足受试单细胞数量 $N_{\text{cells}} \ge 20$；
   - 若某供体在任一生态位中该细胞类型少于 20 个，则该供体作为无效样本排除入集合 $\mathcal{D}_{\text{excluded}}$；冻结有效集合为 $\mathcal{D}_{\text{valid}}=\{\mathrm{D082,D145,D151}\}$，$N_{\text{eff}}=3$；
   - 若 $N_{\text{eff}} < 3$，则强制中止该比较并向用户汇报有效供体不足警告。

### 1.3 描述性效应量度量指标（严谨术语区分）
1. **指标一：供体级平均对数表达差（Mean Log-Expression Difference, $\Delta_d$）**：
   - 对每个供体 $d \in \mathcal{D}_{\text{valid}}$ 内处于生态位 $A$ 与 $B$ 的目标细胞，计算对数标准化表达均值 $\bar{y}_{d, A} = \frac{1}{|C_{d,A}|} \sum_{i \in C_{d,A}} y_i$ 与 $\bar{y}_{d, B}$；
   - 单供体平均对数表达差：$\Delta_d = \bar{y}_{d, A} - \bar{y}_{d, B}$；
   - 严禁将 $\Delta_d$ 混称为严格的 $\log_2\text{FC}$，必须规范定名为“平均对数表达差”；
2. **指标二：真实 Pseudobulk 计数对数折叠变化（Pseudobulk $\log_2\text{FC}_d$）**：
   - 对每个供体 $d \in \mathcal{D}_{\text{valid}}$，分别汇总目标细胞的原始转录本计数（Raw Counts）总和：$u_{d, A} = \sum_{i \in C_{d,A}} x_i$ 与 $u_{d, B} = \sum_{i \in C_{d,B}} x_i$；
   - 按照库大小（Total Counts）进行 CPM 归一化后计算标准对数倍数变化：
     $$\log_2\text{FC}_d = \log_2\left(\frac{\text{CPM}_{d, A} + 1}{\text{CPM}_{d, B} + 1}\right)$$
3. **有效供体宏平均与方向一致性（动态分母 $N_{\text{eff}}$）**：
   - **宏平均效应量**：$\bar{\Delta} = \frac{1}{N_{\text{eff}}} \sum_{d \in \mathcal{D}_{\text{valid}}} \Delta_d$ 及其基于 $t$-分布的 95% 参数置信区间（95% t-based参数置信区间，自由度 $N_{\text{eff}} - 1$）；
   - **跨供体方向一致性率（Directional Concordance Rate）**：
     $$\text{Concordance} = \frac{1}{N_{\text{eff}}} \sum_{d \in \mathcal{D}_{\text{valid}}} \mathbb{I}(\text{sgn}(\Delta_d) == \text{sgn}(\bar{\Delta}))$$
     分母必须严格使用实际有效供体数 $N_{\text{eff}}$，严禁固定除以 5；
4. **功能富集描述性探索**：
   - 仅对方向高度一致（$\text{Concordance} \ge 80\%$）的高效应量基因集进行 GO 生物学通路重叠度说明，背景基因集严格限定为测序芯片覆盖的基因。
