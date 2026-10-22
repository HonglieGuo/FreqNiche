"""Render the four manuscript figures in supplementary spatial/representation analysis.

No model training, resolution selection, PCA fitting, or UMAP recomputation.
Saved UMAP coordinates are reused; the older pooled UMAP's raster scatter
layers are preserved exactly because it did not save coordinate arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Paper/image-gen/s5_supplementary_analysis_20261021"
LEGACY = ROOT / "baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final"
RECENT = ROOT / "Paper/image-gen/human_skin_comparison_20261019"
STEMS = ["fig_supp_all_methods_umap_comparison", "fig_supp_human_skin_d028_umap_comparison",
         "fig_supp_freqniche_representation_diagnostics", "fig4_cross_platform_benchmarks"]
SOURCES = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = sha(path)
    return path


def setup():
    for folder in [Path("/mnt/c/Windows/Fonts"), Path("C:/Windows/Fonts")]:
        for name in ["times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"]:
            if (folder / name).exists(): font_manager.fontManager.addfont(str(folder / name))
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12,
                         "axes.titlesize": 15, "axes.labelsize": 12,
                         "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
                         "legend.fontsize": 11.5, "axes.linewidth": .65,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "axes.unicode_minus": False, "mathtext.fontset": "stix"})


def colors(cats, name="tab10"):
    if len(cats) > plt.get_cmap(name).N:
        values = sns.color_palette("husl", len(cats))
    else:
        values = [plt.get_cmap(name)(i) for i in range(len(cats))]
    return dict(zip(cats, values))


def scatter(ax, xy, values, palette, size=4, alpha=.9):
    values = np.asarray(values)
    for c, color in palette.items():
        mask = values == c
        if mask.any():
            ax.scatter(*np.asarray(xy)[mask].T, color=[color], s=size, alpha=alpha,
                       linewidths=0, rasterized=True)


def legend(ax, palette, ncol=3, cluster=False):
    ax.set_axis_off()
    handles = [Line2D([], [], marker="o", linestyle="", markerfacecolor=color,
                      markeredgecolor="none", markersize=4,
                      label=f"C{c}" if cluster else str(c)) for c, color in palette.items()]
    ax.legend(handles=handles, loc="upper center", ncol=ncol, frameon=False,
              handlelength=.65, handletextpad=.25, columnspacing=.65,
              borderaxespad=0, labelspacing=.35, fontsize=11.5)


def shared_legend(fig, palette, y=.025, ncol=5):
    handles = [Line2D([], [], marker="o", linestyle="", markerfacecolor=color,
                      markeredgecolor="none", markersize=4, label=str(c)) for c, color in palette.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, y), ncol=ncol,
               frameon=False, handlelength=.65, handletextpad=.3, columnspacing=1.4,
               fontsize=11.5)


def frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#9099A3"); spine.set_linewidth(.65)


def note(ax, text):
    ax.text(.975, .035, text, ha="right", va="bottom", transform=ax.transAxes,
            fontsize=13.5, linespacing=1.15,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2})


def metrics(truth, pred):
    return {"ari": float(adjusted_rand_score(truth, pred)),
            "nmi": float(normalized_mutual_info_score(truth, pred))}


def grid_with_legends(rows=2, cols=3, size=(10.8, 7.6), ratio=.5):
    fig = plt.figure(figsize=size)
    grid = fig.add_gridspec(rows, cols, left=.025, right=.985, bottom=.015, top=.94,
                           wspace=.16, hspace=.15)
    axes, legs = [], []
    for i in range(rows * cols):
        sub = grid[i//cols, i%cols].subgridspec(2, 1, height_ratios=[1, ratio], hspace=.07)
        ax = fig.add_subplot(sub[0]); axes.append(ax)
        legs.append(fig.add_subplot(sub[1])); frame(ax)
    return fig, axes, legs


def save(fig, stem, payload):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for obj in [*fig.legends, *[ax.get_legend() for ax in fig.axes if ax.get_legend()]]:
        box = obj.get_window_extent(renderer)
        assert box.x0 >= -1 and box.x1 <= fig.bbox.width + 1, (stem, "legend width", box)
        assert box.y0 >= -1 and box.y1 <= fig.bbox.height + 1, (stem, "legend height", box)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(OUT/f"{stem}.{ext}", dpi=600, facecolor="white")
    plt.close(fig)
    payload.update({"sources": dict(SOURCES), "rendering": {
        "font": "Times New Roman", "font_size": 12, "title_size": 15,
        "metric_size": 13.5, "legend_size": 11.5, "scatter_dpi": 600,
        "metric_location": "inside lower right", "vector_text": True,
        "script_sha256": sha(Path(__file__))}})
    (OUT/f"{stem}_provenance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SAVED", stem, flush=True)


def load_labels():
    df = pd.read_csv(record(RECENT/"all_test_labels_seed42.csv"))
    assert len(df) == 8000 and df.cell_barcode.is_unique
    zpath = record(LEGACY/"embeddings/stagate_controlled_seed_42_test_z.npy")
    expected = json.loads((OUT/"original_fig_supp_all_methods_umap_comparison_provenance.json").read_text())
    assert sha(zpath) == expected["embedding_sha256_verified"]["STAGATE"]
    cache = OUT/"stagate_frozen_labels.npy"
    if cache.exists():
        df["pred_STAGATE"] = np.load(cache)
    else:
        # Replay exactly the frozen historical readout, never choose its resolution.
        import anndata as ad
        import scanpy as sc
        z = np.load(zpath); assert z.shape == (8000, 30)
        work = ad.AnnData(X=z)
        sc.pp.neighbors(work, n_neighbors=15, use_rep="X", random_state=42)
        sc.tl.leiden(work, resolution=.2, key_added="pred", random_state=42)
        df["pred_STAGATE"] = work.obs.pred.to_numpy().astype(int)
        np.save(cache, df.pred_STAGATE.to_numpy())
    for key, name in [("FreqNiche-ST", "FreqNiche"), ("CellCharter", "CellCharter"),
                      ("BANKSY", "BANKSY"), ("STAGATE", "STAGATE"), ("CellNiche", "CellNiche")]:
        m = metrics(df.niche, df["pred_"+name]); old = expected["metrics_seed_42_common_readout"][key]
        assert np.isclose(m["ari"], old["ari_vs_niche"], atol=1e-12, rtol=0)
        assert np.isclose(m["nmi"], old["nmi_vs_niche"], atol=1e-12, rtol=0)
    return df




def umap_figures(df):
    niches = sorted(df.niche.unique()); pal = colors(niches)
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.6))
    fig.subplots_adjust(left=.025,right=.985,top=.94,bottom=.12,wspace=.13,hspace=.20)
    old = json.loads((OUT/"original_fig_supp_all_methods_umap_comparison_provenance.json").read_text())
    names=["FreqNiche", "CellCharter", "BANKSY", "STAGATE", "CellNiche", "Raw PCA"]
    layers=json.loads(record(OUT/"pooled_umap_scatter_layers.json").read_text())
    for i,(ax,name) in enumerate(zip(axes.flat,names)):
        layer=layers["scatter_layers"][i];image=plt.imread(record(OUT/layer["file"]))
        ratio=layer["width"]/layer["height"]
        # Original PDF raster layers contain scatter points only, no text or axes.
        ax.imshow(image,extent=(0,ratio,0,1),interpolation="none",origin="upper")
        ax.set_xlim(0,ratio);ax.set_ylim(0,1);frame(ax)
        ax.set_title(f"({chr(65+i)}) {name}",pad=7)
        if name!="Raw PCA":
            m=metrics(df.niche,df["pred_"+name]);note(ax,f"ARI = {m['ari']:.3f}\nNMI = {m['nmi']:.3f}")
    shared_legend(fig,pal,y=.015)
    save(fig,STEMS[0],{"scope":"Historical 8000-cell UMAPs preserved as exact original scatter layers; historical Raw PCA retained", "metrics":old["metrics_seed_42_common_readout"], "umap_recomputed":False})

    names=["Raw PCA","FreqNiche","CellCharter","GraphST","SEDR","BANKSY","SpaGCN","CellNiche"]
    fig,axes=plt.subplots(2,4,figsize=(11.2,7.2))
    fig.subplots_adjust(left=.025,right=.985,top=.94,bottom=.13,wspace=.16,hspace=.23)
    truth=df.loc[df.donor_id.eq("D028")].set_index("cell_barcode")
    old=json.loads((OUT/"original_fig_supp_human_skin_d028_umap_comparison_provenance.json").read_text())
    records=[]
    for i,(ax,name) in enumerate(zip(axes.flat,names)):
        slug=name.lower().replace(" ","_")
        data=pd.read_csv(record(RECENT/f"{slug}_d028_umap.csv"))
        assert len(data)==2000 and data.cell_barcode.is_unique
        assert data.niche.tolist()==truth.loc[data.cell_barcode,"niche"].tolist()
        xy=data[["umap_1","umap_2"]].to_numpy()
        scatter(ax,xy,data.niche,pal,size=5);frame(ax)
        ax.set_aspect("equal",adjustable="datalim");ax.set_box_aspect(.9)
        ax.set_title(f"({chr(65+i)}) {name}",pad=7)
        m=metrics(truth.niche,truth["pred_"+name])
        saved=next(r for r in old["metrics"] if r["method"]==name)
        for k in ["ari","nmi"]:assert np.isclose(m[k],saved[k],atol=1e-12,rtol=0)
        note(ax,f"NMI = {m['nmi']:.3f}");records.append({"method":name,**m})
    shared_legend(fig,pal,y=.015)
    save(fig,STEMS[1],{"scope":"Same 2000 D028 cells, cached UMAP coordinates, shared reference-niche colors", "metrics":records,"umap_recomputed":False})


def diagnostics(df):
    fig,axes,legs=grid_with_legends(rows=2,cols=2,size=(8.8,8.2),ratio=.5)
    xy=df[["umap_1","umap_2"]].to_numpy();records=[]
    for i,(col,name,cmap) in enumerate([("pred_FreqNiche","Predicted clusters","tab20"),
        ("niche","Reference niches","tab10"),("cell_type","Cell types","tab20"),("donor_id","Donors","Set2")]):
        ax=axes[i];values=df[col];cats=sorted(values.unique());pal=colors(cats,cmap)
        scatter(ax,xy,values,pal,size=4);ax.set_aspect("equal");ax.set_title(f"({chr(65+i)}) {name}",pad=7)
        legend(legs[i],pal,ncol=4 if i in [0,2] else 3,cluster=i==0)
        if i==0:note(ax,"K = 16")
        else:
            nmi=float(normalized_mutual_info_score(df.pred_FreqNiche,values))
            note(ax,f"NMI = {nmi:.3f}");records.append({"annotation":name,"nmi":nmi})
    save(fig,STEMS[2],{"scope":"Same cached 8000-center FreqNiche UMAP, four colorings", "metrics":records,"umap_recomputed":False})


def clean_name(name):
    if "FreqNiche" in name:return "FreqNiche"
    if "DualFreq" in name:return "DualFreq-PCA"
    for prefix in ["BANKSY","CellCharter","STAGATE","CellNiche"]:
        if prefix in name:return prefix
    return name


def parse(value):
    parts=str(value).split("±")
    return float(parts[0]),float(parts[1]) if len(parts)>1 else 0.


def benchmarks():
    original=json.loads((OUT/"original_fig4_cross_platform_benchmarks_provenance.json").read_text())
    paths={"human_skin_table1":LEGACY/"table1_full_cohort_primary.csv",
           "sea_ad_table5":ROOT/"outputs/analysis/paper-014-update-20261015/sea_ad_paper.csv",
           "cosmx_table6":ROOT/"outputs/runs/freqniche/cosmx_nsclc/2026-09-17/20260917_cosmx_nsclc_final/table6_cosmx_full_benchmark.csv"}
    for key,path in paths.items():
        assert sha(record(path))==original["source_sha256"][original["sources"][key]]
    skin=pd.read_csv(paths["human_skin_table1"])
    agg=skin.groupby("Method").agg(ari_mean=("Test_Donor_Macro_ARI","mean"),ari_std=("Test_Donor_Macro_ARI","std"),
         nmi_mean=("Test_Donor_Macro_NMI","mean"),nmi_std=("Test_Donor_Macro_NMI","std"),worst_ari=("Worst_Donor_ARI","min")).sort_values("ari_mean",ascending=False)
    sea=pd.read_csv(paths["sea_ad_table5"]);cos=pd.read_csv(paths["cosmx_table6"])
    # Panel C: put FreqNiche first, retaining the order of all other methods.
    is_freqniche=sea["Method"].map(clean_name).eq("FreqNiche")
    sea=pd.concat([sea.loc[is_freqniche],sea.loc[~is_freqniche]],ignore_index=True)
    fig,axes=plt.subplots(2,2,figsize=(11.2,8.8))
    fig.subplots_adjust(left=.07,right=.985,top=.95,bottom=.10,hspace=.52,wspace=.25)
    def pair(ax,names,am,asd,nm,nsd,title):
        x=np.arange(len(names));width=.36
        ax.bar(x-width/2,am,width,yerr=asd,label="ARI",color="#397CB2",capsize=3,linewidth=.6,edgecolor="white")
        ax.bar(x+width/2,nm,width,yerr=nsd,label="NMI",color="#DF8D49",capsize=3,linewidth=.6,edgecolor="white")
        ax.set_xticks(x,[clean_name(n) for n in names],rotation=28,ha="right")
        ax.set_ylabel("Score");ax.set_ylim(0,max(.55,max(np.array(am)+asd),max(np.array(nm)+nsd))+.08)
        ax.legend(loc="lower right",fontsize=11.5,framealpha=.92,edgecolor="none")
        ax.set_title(title,pad=10)
    pair(axes[0,0],agg.index,agg.ari_mean,agg.ari_std,agg.nmi_mean,agg.nmi_std,"(A) Human Skin")
    ax=axes[0,1];x=np.arange(len(agg));vals=agg.worst_ari.to_numpy()
    ax.bar(x,vals,color=["#397CB2" if "FreqNiche" in n else "#9AA6B2" for n in agg.index],width=.6)
    ax.set_xticks(x,[clean_name(n) for n in agg.index],rotation=28,ha="right")
    ax.set_ylabel("Minimum donor ARI");ax.set_ylim(0,max(vals)*1.3);ax.set_title("(B) Human Skin: worst donor",pad=10)
    for i,v in enumerate(vals):ax.text(i,v+.008,f"{v:.3f}",ha="center",va="bottom",fontsize=11.5)
    for ax,data,col,title,ac,nc in [(axes[1,0],sea,"Method","(C) SEA-AD","Macro ARI","Macro NMI"),
                                   (axes[1,1],cos,"Method_variant","(D) CosMx","ARI","NMI")]:
        ari=np.array([parse(v) for v in data[ac]]);nmi=np.array([parse(v) for v in data[nc]])
        pair(ax,data[col],ari[:,0],ari[:,1],nmi[:,0],nmi[:,1],title)
    for ax in axes.flat:
        ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.18,linewidth=.5);ax.set_axisbelow(True)
    save(fig,STEMS[3],{"scope":"Historical comparison retained exactly; not an updated full-method benchmark", "freqniche_versions":original["freqniche_versions"],
                      "skin_aggregates":agg.reset_index().to_dict("records"),"sea_records":sea.to_dict("records"),"cosmx_records":cos.to_dict("records")})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only",choices=["umap","diagnostics","benchmarks"])
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);setup()
    df=load_labels() if args.only!="benchmarks" else None
    if args.only in [None,"umap"]:umap_figures(df)
    if args.only in [None,"diagnostics"]:diagnostics(df)
    if args.only in [None,"benchmarks"]:benchmarks()
    for rel,digest in SOURCES.items():assert sha(ROOT/rel)==digest,rel
    print("PASS: all read-only input hashes unchanged",flush=True)


if __name__=="__main__":main()
