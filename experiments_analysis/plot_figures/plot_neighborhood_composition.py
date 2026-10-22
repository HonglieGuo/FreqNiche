"""Restyle Figure 4 without changing frozen probe choices or benchmark results.

Recover missing plotting arrays using the already selected regression alphas,
verify the original donor metrics, then cache predictions and fitted probes.
No cross-validation, hyperparameter selection, or FreqNiche retraining.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import anndata as ad
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression,Ridge
from experiments_analysis.neighborhood_composition_prediction.run_neighborhood_composition_prediction import compute_neighborhood_ground_truth,evaluate_predictions
from src.freqniche_st.runtime_paths import validated_source_bindings

SUMMARY=ROOT/'experiments_analysis/neighborhood_composition_prediction/results/neighborhood_composition_results.json'
LEGACY=ROOT/'experiments_analysis/neighborhood_composition_prediction/run_neighborhood_composition_prediction.py'
DATA=ROOT/'datasets/h5ad/human_skin_merfish/human_skin_merfish.h5ad'
VAL=ROOT/'outputs/freqniche_wave_a_diagnostics/original_val_barcodes.csv'
TEST=ROOT/'outputs/freqniche_wave_a_diagnostics/wave_a_main/test_centers_mapping.csv'
TRAIN=ROOT/'outputs/runs/freqniche/human_skin/2026-09-08/20260908_human_skin_training_final'
EVAL=ROOT/'outputs/runs/freqniche/human_skin/2026-09-18/20260918_human_skin_evaluation_final'
NAME='fig2_neighborhood_composition_prediction'
SEEDS=[42,123,2024]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def prepare(out,summary):
    paths=[SUMMARY,LEGACY,VAL,TEST,DATA]
    for seed in SEEDS:
        paths.extend([TRAIN/f'trial_24_seed_{seed}/val_embeddings.npy',EVAL/f'embeddings/freqniche_st_v1_trial24_seed_{seed}_test_z.npy'])
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    binding=out/'plot_data_binding.json'
    if binding.exists():
        meta=json.loads(binding.read_text());assert validated_source_bindings(meta['sources'])==sources,'Frozen plotting inputs changed'
        for relative,digest in meta['files'].items():assert sha(out/relative)==digest,relative
        return np.load(out/'plot_arrays.npz'),pd.read_csv(out/'test_centers.csv'),meta
    a=ad.read_h5ad(DATA,backed='r')
    try:
        val_ids=pd.read_csv(VAL).cell_barcode.tolist();frame=pd.read_csv(TEST);test_ids=frame.cell_barcode.tolist()
        assert len(val_ids)==len(test_ids)==8000
        gt_val,cell_types,val_donors=compute_neighborhood_ground_truth(a,val_ids,radius=50.)
        gt_test,test_types,test_donors=compute_neighborhood_ground_truth(a,test_ids,radius=50.)
        assert cell_types==test_types==summary['cell_types'] and len(cell_types)==18
        assert np.array_equal(test_donors,frame.donor_id)
        obs=a.obs.loc[test_ids]
        frame['center_x']=obs.center_x.to_numpy();frame['center_y']=obs.center_y.to_numpy()
        raw=a[val_ids+test_ids,:].X
        raw=raw.toarray() if hasattr(raw,'toarray') else np.asarray(raw)
    finally:a.file.close()
    arrays={'gt_test':gt_test,'gt_val':gt_val};records={};max_difference=0.
    for seed in SEEDS:
        records[str(seed)]={}
        for method in ['freqniche','raw_pca']:
            saved=summary['per_seed_results'][str(seed)][method];alpha=saved['best_alpha']
            reg=LinearRegression() if alpha==0 else Ridge(alpha=alpha,random_state=seed)
            if method=='freqniche':
                x_val=np.load(TRAIN/f'trial_24_seed_{seed}/val_embeddings.npy')
                x_test=np.load(EVAL/f'embeddings/freqniche_st_v1_trial24_seed_{seed}_test_z.npy')
                pipe=Pipeline([('scaler',StandardScaler()),('reg',reg)])
            else:
                x_val,x_test=raw[:8000],raw[8000:]
                pipe=Pipeline([('pca',PCA(n_components=50,random_state=seed)),('scaler',StandardScaler()),('reg',reg)])
            pipe.fit(x_val,gt_val)
            result=evaluate_predictions(gt_test,pipe.predict(x_test),test_donors,cell_types)
            for key in ['macro_mae','macro_js_distance']:
                difference=abs(result[key]-saved[key]);max_difference=max(max_difference,difference)
                assert np.isclose(result[key],saved[key],rtol=1e-5,atol=1e-6),(seed,method,key,result[key],saved[key])
            for donor in summary['unseen_test_donors']:
                for key in ['mae','js_distance']:
                    difference=abs(result['by_donor'][donor][key]-saved['by_donor'][donor][key])
                    max_difference=max(max_difference,difference)
                    assert np.isclose(result['by_donor'][donor][key],saved['by_donor'][donor][key],rtol=1e-5,atol=1e-6)
            arrays[f'{method}_{seed}']=result['y_pred_norm']
            arrays[f'{method}_{seed}_type_mae']=np.array([result['by_cell_type_mae'][ct] for ct in cell_types])
            records[str(seed)][method]={'frozen_alpha':alpha,'macro_mae':result['macro_mae'],'macro_js_distance':result['macro_js_distance']}
            joblib.dump(pipe,out/f'{method}_seed_{seed}_probe.joblib')
            print('Recovered and verified',method,seed,result['macro_mae'],result['macro_js_distance'],flush=True)
    np.savez_compressed(out/'plot_arrays.npz',**arrays);frame.to_csv(out/'test_centers.csv',index=False)
    meta={'sources':sources,'verification':'passed','max_metric_absolute_difference':max_difference,
          'probe_recovery':records,'note':'Refit final linear probes at the original frozen alpha solely to recover missing plotting arrays. No CV/selection or model retraining. Original JSON unchanged.'}
    meta['files']={p.name:sha(p) for p in out.iterdir() if p.name=='plot_arrays.npz' or p.name=='test_centers.csv' or p.suffix=='.joblib'}
    write(binding,meta)
    return np.load(out/'plot_arrays.npz'),frame,meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'Paper/image-gen/neighborhood_composition_figure4_20261019')
    parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    original_hash=sha(SUMMARY);assert original_hash=='466c7579418a7a75b943eb40b8e96585d2c1952b7f0fc9991e2cc4fd20a4c575'
    summary=json.loads(SUMMARY.read_text());arrays,frame,binding=prepare(out,summary)
    for directory in [Path('/mnt/c/Windows/Fonts'),Path('C:/Windows/Fonts')]:
        for f in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
            if (directory/f).exists():font_manager.fontManager.addfont(str(directory/f))
    font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':13,'axes.titlesize':15,
                         'axes.labelsize':13,'xtick.labelsize':12.5,'ytick.labelsize':12.5,
                         'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})
    fig=plt.figure(figsize=(10.8,5.4))
    axA=fig.add_axes([.025,.43,.18,.44]);axB=fig.add_axes([.225,.43,.18,.44])
    axC=fig.add_axes([.52,.19,.25,.67]);axD=fig.add_axes([.85,.19,.135,.67])
    cats=summary['cell_types'];colors={c:plt.get_cmap('tab20')(i) for i,c in enumerate(cats)}
    mask=frame.donor_id.eq('D028').to_numpy();xy=frame.loc[mask,['center_x','center_y']].to_numpy()
    assert mask.sum()==2000
    observed=np.argmax(arrays['gt_test'][mask],axis=1);predicted=np.argmax(arrays['freqniche_42'][mask],axis=1)
    for ax,values,title in [(axA,observed,'(A) Observed'),(axB,predicted,'(B) Predicted')]:
        for i,c in enumerate(cats):
            selected=values==i
            ax.scatter(*xy[selected].T,c=[colors[c]],s=5,alpha=.9,linewidths=0,rasterized=True)
        ax.set_box_aspect(.82);ax.set_aspect('equal',adjustable='datalim');ax.set_anchor('N')
        ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,loc='center',fontsize=15,pad=7)
        for spine in ax.spines.values():spine.set_edgecolor('#cccccc')
    fig.canvas.draw();axB.set_xlim(axA.get_xlim());axB.set_ylim(axA.get_ylim())
    leg=fig.add_axes([.025,.025,.38,.50]);leg.set_axis_off()
    handles=[Line2D([],[],marker='o',linestyle='none',markersize=4,markerfacecolor=colors[c],markeredgecolor='none',label=c) for c in cats]
    leg.legend(handles=handles,loc='upper center',ncol=3,frameon=False,fontsize=12.5,handlelength=.55,handletextpad=.25,columnspacing=.9,labelspacing=.4)
    raw=np.mean([arrays[f'raw_pca_{s}_type_mae'] for s in SEEDS],axis=0)
    freq=np.mean([arrays[f'freqniche_{s}_type_mae'] for s in SEEDS],axis=0)
    y=np.arange(18);height=.36;gray='#929292';blue='#2878ae'
    axC.barh(y-height/2,raw,height,color=gray,linewidth=0,label='Raw PCA')
    axC.barh(y+height/2,freq,height,color=blue,linewidth=0,label='FreqNiche')
    axC.set_yticks(y,cats);axC.invert_yaxis();axC.set_xlabel('MAE');axC.set_title('(C) Cell-type MAE',loc='center',fontsize=15,pad=7)
    axC.set_xlim(0,max(raw.max(),freq.max())*1.12);axC.set_ylim(17.8,-.8)
    axC.set_axisbelow(True);axC.grid(axis='x',color='#e4e4e4',linewidth=.5)
    donors=summary['unseen_test_donors'];x=np.arange(4);width=.36
    for method,offset,color in [('raw_pca',-width/2,gray),('freqniche',width/2,blue)]:
        values=np.array([[summary['per_seed_results'][str(s)][method]['by_donor'][d]['js_distance'] for d in donors] for s in SEEDS])
        mean=values.mean(0);std=values.std(0,ddof=1)
        axD.bar(x+offset,mean,width,yerr=std,color=color,linewidth=0,capsize=2,error_kw={'elinewidth':.7})
    axD.set_xticks(x,donors);axD.tick_params(axis='x',labelsize=12)
    plt.setp(axD.get_xticklabels(),rotation=45,ha='right',rotation_mode='anchor')
    axD.set_ylabel('JS distance');axD.set_ylim(0,.60);axD.set_yticks([0,.2,.4,.6]);axD.set_title('(D) Donor error',loc='center',fontsize=15,pad=7)
    axD.set_axisbelow(True);axD.grid(axis='y',color='#e4e4e4',linewidth=.5)
    for ax in [axC,axD]:
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    fig.legend(handles=[Patch(facecolor=gray,label='Raw PCA'),Patch(facecolor=blue,label='FreqNiche')],
               loc='lower center',bbox_to_anchor=(.75,.005),ncol=2,frameon=False,fontsize=13)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box=text.get_window_extent(renderer)
            assert box.x0>=-1 and box.x1<=fig.bbox.width+1 and box.y0>=-1 and box.y1<=fig.bbox.height+1,(text.get_text(),box)
    for ext in ['pdf','svg','png']:fig.savefig(out/(NAME+'.'+ext),dpi=600,facecolor='white')
    plt.close(fig)
    pd.DataFrame({'cell_type':cats,'raw_pca_mae_mean':raw,'freqniche_mae_mean':freq}).to_csv(out/'cell_type_mae.csv',index=False)
    spatial=frame.loc[mask,['cell_barcode','sample_id','center_x','center_y']].copy()
    spatial['observed_dominant_type']=[cats[i] for i in observed];spatial['predicted_dominant_type']=[cats[i] for i in predicted]
    spatial.to_csv(out/'d028_dominant_types_seed42.csv',index=False)
    meta={'task':summary['task'],'source_results_json':str(SUMMARY.relative_to(ROOT)),'source_results_sha256':original_hash,
          'aggregated_metrics':summary['aggregated_metrics'],'plot_binding':binding,'scope':{'A_B':'D028 same 2000 centers as the previous figure; dominant neighboring cell type only, not a single physical section',
          'C':'18 cell types, three-seed mean MAE on the fixed full test cohort','D':'four test donors, three-seed mean JS distance and sample SD'},
          'rendering':{'layout':'one row x four panels','font':'Times New Roman','title_font_pt':15,'scatter_rasterized':True,'raster_dpi':600,
                       'bar_axes_text_legends_vector':True,'C_orientation':'horizontal grouped bars','script_sha256':sha(Path(__file__))}}
    write(out/(NAME+'_provenance.json'),meta)
    assert sha(SUMMARY)==original_hash,'Frozen original results must remain untouched'
    if args.publish:
        for suffix in ['.pdf','.svg','.png','_provenance.json']:shutil.copyfile(out/(NAME+suffix),ROOT/'Paper/FreqNiche-zh/figures'/(NAME+suffix))
    print('FIGURE_4_COMPLETE',out,flush=True)


if __name__=='__main__':main()
