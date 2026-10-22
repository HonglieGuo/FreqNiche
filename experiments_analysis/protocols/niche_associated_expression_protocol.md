# Within-cell-type niche-associated expression analysis

This protocol corresponds to the identically named section in the manuscript's experiments and results. It records the grouping, descriptive effect sizes and pathway-overlap definitions used in the original comparison. For new training runs, use `niche_associated_expression/run_niche_associated_expression.py`; actual donors and cluster IDs are determined from the new inputs and validity checks.

### 1.1 Scope of statistical inference

- **Sample-size constraint:** the fixed Human Skin Test cohort contains four donors. Applying the minimum of 20 cells per `(donor, cluster, Fibro)` group excludes D028. The effective donors are D082, D145 and D151, giving $N_{\mathrm{eff}}=3$. With this limited donor count, the analysis does not perform or claim donor-level significance discovery and does not describe genes as significantly differentially expressed.
- **Descriptive interpretation:** niche-associated expression is a descriptive effect-size and cross-donor directional-concordance analysis. The 95% intervals based on the $t$ distribution describe effect-size uncertainty; they are not used for population-level inference or discovery claims.

### 1.2 Cross-donor niche correspondence and minimum cell counts

1. **Fixed cell type and annotation field:** Human Skin MERFISH uses the single annotation field `adata.obs['cell_type']`, with target type fixed to `Fibro`, not `Fibroblasts`.
2. **Cross-donor niche correspondence:**
   - Do not use arbitrary donor-specific cluster IDs.
   - Use the frozen global clustering from the full FreqNiche Trial 24, Seed 42 model on 8,000 Test centers ($r^*=0.8$). Compare Cluster 2 (STROMA-enriched; 58.4% reference-niche enrichment) with Cluster 0 (PERIVASC-enriched; 53.4% enrichment). Describe these as enriched clusters, not pure anatomical regions or causal microenvironments.
3. **Sample inclusivity gate:**
   - Each `(donor, niche, cell_type)` group must contain at least $N_{\text{cells}} \ge 20$ cells.
   - If a donor has fewer than 20 cells of the target type in either niche, exclude that donor into $\mathcal{D}_{\text{excluded}}$. The frozen valid set is $\mathcal{D}_{\text{valid}}=\{\mathrm{D082,D145,D151}\}$, with $N_{\text{eff}}=3$.
   - If $N_{\text{eff}} < 3$, stop the comparison and report that the number of valid donors is insufficient.

### 1.3 Descriptive effect-size metrics

1. **Donor-level mean log-expression difference ($\Delta_d$):**
   - For target cells in niches $A$ and $B$ within each donor $d \in \mathcal{D}_{\text{valid}}$, calculate the means of log-normalized expression: $\bar{y}_{d, A} = \frac{1}{|C_{d,A}|} \sum_{i \in C_{d,A}} y_i$ and $\bar{y}_{d, B}$.
   - The donor-level difference is $\Delta_d = \bar{y}_{d, A} - \bar{y}_{d, B}$.
   - Call $\Delta_d$ the mean log-expression difference; do not describe it as a strict $\log_2\text{FC}$.
2. **Pseudobulk count log-fold change ($\log_2\text{FC}_d$):**
   - For each donor $d \in \mathcal{D}_{\text{valid}}$, sum raw transcript counts over target cells in each group: $u_{d, A} = \sum_{i \in C_{d,A}} x_i$ and $u_{d, B} = \sum_{i \in C_{d,B}} x_i$.
   - Normalize by library size to counts per million (CPM), then calculate:

     $$\log_2\text{FC}_d = \log_2\left(\frac{\text{CPM}_{d, A} + 1}{\text{CPM}_{d, B} + 1}\right)$$

3. **Valid-donor macro-average and directional concordance (dynamic denominator $N_{\text{eff}}$):**
   - **Macro-average effect:** $\bar{\Delta} = \frac{1}{N_{\text{eff}}} \sum_{d \in \mathcal{D}_{\text{valid}}} \Delta_d$, with a 95% parametric interval based on the $t$ distribution and $N_{\text{eff}} - 1$ degrees of freedom.
   - **Cross-donor directional concordance rate:**

     $$\text{Concordance} = \frac{1}{N_{\text{eff}}} \sum_{d \in \mathcal{D}_{\text{valid}}} \mathbb{I}(\text{sgn}(\Delta_d) == \text{sgn}(\bar{\Delta}))$$

     The denominator must be the actual number of valid donors, $N_{\text{eff}}$, and must never be fixed to 5.
4. **Descriptive exploration of functional overlap:**
   - Describe GO biological pathway overlap only for high-effect gene sets with high directional concordance ($\text{Concordance} \ge 80\%$). Restrict the background gene universe to genes covered by the measured panel.
