"""Render supplementary clusterer/Xenium results from immutable saved tables."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from experiments_analysis.plot_figures.restyle_s5_supplementary_analysis import setup, clean_name

OUT = ROOT/"Paper/image-gen/s6_s7_analysis_20261021"
COMBINED = ROOT/"baselines/Baselines_outputs/runs/human_skin/2026-09-28/20260928_human_skin_clusterer_sensitivity/combined_v1"
SMOKE = ROOT/"baselines/Baselines_outputs/runs/xenium_prime5k/2026-09-18/20260918_xenium_prime5k_baselines/smoke"
XENIUM = ROOT/"outputs/analysis/paper-014-update-20261015/xenium_prime5k_paper.csv"
INFERENCE = ROOT/"outputs/analysis/xenium_inference_scaling_20261021"
ORDER = ["kmeans", "mclust", "louvain", "leiden"]
TICKS = ["K-means", "mclust", "Louvain", "Leiden"]
SOURCES = {}
OUTPUTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    SOURCES[str(path.relative_to(ROOT))] = sha(path)
    return path


def save(fig, stem, data):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        xmin,xmax=sorted(ax.get_xlim());ymin,ymax=sorted(ax.get_ylim())
        # Locators also create ticks outside the visible view; they are not drawn.
        xticks=[t for t in ax.get_xticklabels() if xmin<=t.get_position()[0]<=xmax]
        yticks=[t for t in ax.get_yticklabels() if ymin<=t.get_position()[1]<=ymax]
        for artist in [ax.get_legend(), ax.title, *xticks, *yticks]:
            if artist is None or not artist.get_visible(): continue
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -1 and box.x1 <= fig.bbox.width+1, (stem, "horizontal overflow", box)
            assert box.y0 >= -1 and box.y1 <= fig.bbox.height+1, (stem, "vertical overflow", box)
    for ext in ["pdf", "svg", "png"]:
        p = OUT/f"{stem}.{ext}"; fig.savefig(p, dpi=600, facecolor="white"); OUTPUTS[p.name] = sha(p)
    plt.close(fig)
    data.update({"sources": dict(SOURCES), "source_results_modified": False,
                 "style": {"font": "Times New Roman", "font_size": 12, "title_size": 15,
                           "raster_dpi": 600, "vector_text": True}, "plot_script_sha256": sha(Path(__file__))})
    (OUT/f"{stem}_provenance.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("SAVED", stem, flush=True)


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=.5, alpha=.18); ax.set_axisbelow(True)


def clusterers():
    original = json.loads((OUT/"original_clusterer_figures_provenance.json").read_text())
    for key, name in [("summary_csv", "summary.csv"), ("all_runs_csv", "all_runs.csv"), ("combined_report", "combined_report.json")]:
        assert sha(read(COMBINED/name)) == original["inputs"][key]["sha256"], name
    summary = pd.read_csv(COMBINED/"summary.csv")
    assert len(summary) in (40, 48) and set(summary.clusterer) == set(ORDER)
    # The paper presents FreqNiche exclusively in its default inductive form.
    summary = summary.loc[~summary.method.eq("FreqNiche-Transductive")].copy()
    assert len(summary) == 40
    primary = summary[summary.track.eq("validation_frozen")]
    diagnostic = summary[summary.track.eq("oracle_true_k")]
    assert len(primary) == len(diagnostic) == 20
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    fig.subplots_adjust(left=.075, right=.985, bottom=.20, top=.89, wspace=.23)
    names = ["FreqNiche-Inductive"]
    for j, (ax, metric, label) in enumerate(zip(axes, ["ari", "nmi"], ["ARI", "NMI"])):
        x = np.arange(4)
        lookup = primary[primary.method.eq(names[0])].set_index("clusterer").loc[ORDER]
        ax.bar(x, lookup["mean_donor_macro_"+metric], .55,
               yerr=lookup["sample_sd_donor_macro_"+metric], color="#397CB2", capsize=3,
               edgecolor="white", linewidth=.5)
        ax.set_xticks(x, ["K-means", "mclust", "Louvain", "Leiden"]); ax.set_ylabel("Donor-macro "+label)
        ax.set_ylim(0, .52 if j == 0 else .64)
        ax.set_title(f"({chr(65+j)}) {label}", pad=10); tidy(ax)
    save(fig, "freqniche_clusterer_sensitivity", {"scope": "Human Skin, default inductive FreqNiche, validation-selected readout", "values":primary[primary.method.isin(names)].to_dict("records")})

    methods = sorted(primary.method.unique())
    displays = ["BANKSY", "CellCharter", "CellNiche", "FreqNiche", "STAGATE"]
    vmax = float(summary.mean_donor_macro_ari.max())
    assert vmax <= original["figure_metadata"]["heatmap_color_scale"]["vmax"]
    fig = plt.figure(figsize=(10.4, 4.6))
    axes = [fig.add_axes([.22, .19, .325, .68]), fig.add_axes([.595, .19, .325, .68])]
    cax = fig.add_axes([.94, .19, .014, .68])
    for i, (ax, frame, title) in enumerate(zip(axes, [primary, diagnostic], ["(A) Validation-selected", "(B) Reference-count diagnostic"])):
        matrix = frame.pivot(index="method", columns="clusterer", values="mean_donor_macro_ari").reindex(index=methods, columns=ORDER)
        sns.heatmap(matrix, ax=ax, annot=True, fmt=".3f", annot_kws={"fontsize":13},
                    cmap="viridis", vmin=0, vmax=vmax, cbar=i==1, cbar_ax=cax if i==1 else None,
                    cbar_kws={"label":""}, linewidths=.5, linecolor="white")
        ax.set_xticklabels(TICKS, rotation=0, fontsize=11.5)
        ax.set_yticklabels(displays if i==0 else [""]*len(displays), rotation=0, fontsize=12.5)
        ax.set_xlabel(""); ax.set_ylabel(""); ax.tick_params(length=0)
        ax.set_title(title, pad=10)
    cax.set_title("ARI",fontsize=11.5,pad=8)
    save(fig, "all_methods_clusterer_heatmap", {"scope":"Same frozen representations, primary and reference-count diagnostic readouts", "values":summary.to_dict("records"), "shared_color_limits":[0,vmax]})


def parse(value):
    a,b = str(value).split("±")
    return float(a),float(b)


def xenium():
    original = json.loads((OUT/"original_fig5_xenium_large_scale_benchmark_provenance.json").read_text())
    paths = {"xenium_table7": XENIUM}
    for key,p in paths.items():
        assert sha(read(p)) == original["source_sha256"][original["sources"][key]]
    df = pd.read_csv(XENIUM); assert len(df)==6
    inference = json.loads(read(INFERENCE/"telemetry.json").read_text())
    audit = json.loads(read(INFERENCE/"audit.json").read_text())
    assert inference["status"] == "complete" and audit["status"] == "passed"
    assert inference["protocol"]["stage"] == "frozen_encoder_inference"
    scaling = pd.DataFrame(inference["summary"]).set_index("cells").loc[[1000,5000,10000]]
    labels = [clean_name(n) for n in df.Method]; x=np.arange(len(df))
    fig,axes = plt.subplots(2,2,figsize=(11.2,7.4))
    fig.subplots_adjust(left=.075,right=.985,bottom=.13,top=.93,wspace=.25,hspace=.60)
    def pair(ax, columns, names, colors):
        for i,(col,name,color) in enumerate(zip(columns,names,colors)):
            data=np.array([parse(v) for v in df[col]])
            ax.bar(x+(i-.5)*.36,data[:,0],.36,yerr=data[:,1],label=name,color=color,
                   edgecolor="white",linewidth=.5,capsize=3)
        ax.set_xticks(x,labels,rotation=28,ha="right")
        ax.set_ylim(0,1.1);ax.legend(loc="upper left",frameon=False,fontsize=11)
    pair(axes[0,0],["Moran_I_Top5_PCs","Moran_I_Clusters"],["Top-5 PCs","Cluster fields"],["#397CB2","#66A182"])
    axes[0,0].set_ylabel("Moran's I increment");axes[0,0].set_title("(A) Spatial autocorrelation",pad=10)
    pair(axes[0,1],["Local_Purity_R50um","Cross_Seed_ARI"],["Local purity","Cross-seed ARI"],["#8D7DB3","#DF8D49"])
    axes[0,1].set_ylabel("Score");axes[0,1].set_title("(B) Continuity and stability",pad=10)

    ax=axes[1,0];xx=np.arange(3)
    # Separate allocator-based frozen inference from historical fit_transform
    # process-memory telemetry. Only this panel's contents/layout are changed.
    pos=ax.get_position();ax.set_position([pos.x0,pos.y0,pos.width-.045,pos.height])
    memory=scaling.cuda_peak_allocated_mib_mean.to_numpy()
    memory_sd=scaling.cuda_peak_allocated_mib_sd.to_numpy()
    runtime=scaling.runtime_sec_mean.to_numpy();runtime_sd=scaling.runtime_sec_sd.to_numpy()
    bars=ax.bar(xx,memory,.48,yerr=memory_sd,capsize=3,color="#397CB2",
                edgecolor="white",linewidth=.5,label="Peak allocated memory")
    ax.set_xticks(xx,["1,000","5,000","10,000"]);ax.set_xlabel("Cells")
    ax.set_ylabel("CUDA peak allocated (MiB)");ax.set_ylim(0,max(memory+memory_sd)*1.55)
    for i,value in enumerate(memory):
        ax.text(i,value+memory_sd[i]+max(memory)*.025,f"{value:.1f}",ha="center",va="bottom",fontsize=10.5,color="#245C8A")
    right=ax.twinx()
    line=right.errorbar(xx,runtime,yerr=runtime_sd,color="#C87536",marker="o",linestyle="--",
                        linewidth=1.4,capsize=3,label="Inference time")
    right.set_ylabel("Inference time (s)",labelpad=3);right.set_ylim(0,max(runtime+runtime_sd)*1.9)
    right.spines[["top"]].set_visible(False)
    ax.set_title("(C) FreqNiche frozen inference",pad=10)
    ax.legend([bars,line],["Peak allocated memory","Inference time"],loc="upper left",frameon=False,fontsize=10.5)

    ax=axes[1,1];rank=np.array([parse(v) for v in df.Effective_Rank]);dims=df.Dim.to_numpy(dtype=int)
    pct=rank[:,0]/dims*100;err=rank[:,1]/dims*100
    ax.bar(x,pct,width=.6,yerr=err,color=["#397CB2" if "FreqNiche" in name else "#9AA6B2" for name in df.Method],capsize=3)
    for i,(value,dimension,percent) in enumerate(zip(rank[:,0],dims,pct)):
        ax.text(i,percent+2,f"{value:.1f}/{dimension}\n{percent:.1f}%",ha="center",va="bottom",fontsize=10.5)
    ax.set_xticks(x,labels,rotation=28,ha="right");ax.set_ylim(0,120)
    ax.set_ylabel("Effective rank / dimension (%)");ax.set_title("(D) Dimension utilization",pad=10)
    for ax in axes.flat:tidy(ax)
    save(fig,"fig5_xenium_large_scale_benchmark",{"scope":"Xenium frozen-encoder evaluation; panel C uses independently audited FreqNiche inference telemetry", "values":df.to_dict("records"),
         "freqniche_inference_scaling":inference["summary"],"freqniche_gpu_ladder_measured":True,
         "panel_c_memory_metric":"CUDA peak allocated MiB, resident encoder included",
         "panel_c_time_metric":inference["protocol"]["timed"],
         "historical_baseline_fit_memory_not_compared":True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only",choices=["clusterers","xenium"])
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);setup()
    if args.only in [None,"clusterers"]:clusterers()
    if args.only in [None,"xenium"]:xenium()
    for rel,digest in SOURCES.items():assert sha(ROOT/rel)==digest,rel
    meta={"analysis":"Supplementary clusterer and Xenium figure restyling", "inputs":SOURCES,"figure_outputs":OUTPUTS,"source_data_unchanged":True}
    (OUT/"restyle_provenance.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("PASS: all source data and reported values preserved",flush=True)


if __name__=="__main__":main()
