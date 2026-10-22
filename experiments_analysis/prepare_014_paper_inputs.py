"""Prepare publication inputs without changing original experiment ledgers."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/analysis/paper-014-update-20261015"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def main():
    data = json.loads((ROOT / "outputs/experiments/014-test-20261015/comparison.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {
        "sea_ad": ROOT / "baselines/Baselines_outputs/runs/sea_ad/2026-09-15/20260915_sea_ad_baselines/table5_sea_ad_baseline_benchmark.csv",
        "xenium_prime5k": ROOT / "outputs/runs/freqniche/xenium_prime5k/2026-09-18/20260918_xenium_prime5k_final/table7_benchmark_synthesis.csv",
    }
    maps = {
        "sea_ad": {"Macro ARI": "ari", "Macro NMI": "nmi", "Worst-Donor ARI": "worst_donor_ari", "ASW": "asw", "Clusters (K)": "n_clusters"},
        "xenium_prime5k": {"K_Clusters": "n_clusters", "Internal_ASW": "asw", "Effective_Rank": "effective_rank", "Local_Purity_R50um": "local_purity", "Moran_I_Top5_PCs": "moran_i_top5_pcs", "Moran_I_Clusters": "moran_i_clusters", "Cross_Seed_ARI": "cross_seed_ari"},
    }
    audit = {"unchanged_primary_results": ["human_skin", "cosmx_nsclc"], "updates": {},
             "selection_disclosure": "Dataset-level result replacements were requested after inspecting Test; these updates are follow-up re-evaluations, not prospective blind confirmations."}
    for dataset, path in sources.items():
        fields, original = read_csv(path)
        rows = [dict(row) for row in original]
        targets = [row for row in rows if "FreqNiche" in row["Method"]]
        assert len(targets) == 1
        target = targets[0]
        metrics = data["new_results"][dataset]["metrics"]
        for column, key in maps[dataset].items():
            value = metrics[key]
            precision = 1 if key in ("n_clusters", "effective_rank") else 4
            target[column] = f"{value['mean']:.{precision}f} ± {value['sd']:.{precision}f}"
        dim_column = "Embedding_dim" if dataset == "sea_ad" else "Dim"
        target[dim_column] = "256"
        if dataset == "sea_ad":
            target["Method_variant"] = "FreqNiche 014 (Inductive, 3-Seed)"
            target["Training_epochs"] = "200 (Budgeted; early stopping)"
            target["Optimal_r_star"] = "1.0/0.8/1.0"
        else:
            target["Category"] = "Dual-Frequency Conditional Diffusion"
        # Baseline records are byte-for-byte equal at the field-value level.
        assert [r for r in rows if "FreqNiche" not in r["Method"]] == [r for r in original if "FreqNiche" not in r["Method"]]
        output = OUT / f"{dataset}_paper.csv"
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        audit["updates"][dataset] = {"source": str(path.relative_to(ROOT)), "output": str(output.relative_to(ROOT)),
                                     "before": next(r for r in original if "FreqNiche" in r["Method"]), "after": target,
                                     "baseline_records_unchanged": True}
    (OUT / "paper_input_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Prepared SEA-AD and Xenium inputs; all baseline records unchanged.")


if __name__ == "__main__":
    main()
