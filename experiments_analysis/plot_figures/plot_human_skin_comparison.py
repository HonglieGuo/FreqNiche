"""Eight-panel spatial Figure 3 and supplementary UMAP comparisons from saved assets."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from experiments_analysis.plot_figures.human_skin_plot_utils import palette,scatter,legend,annotation,sha

BASE=ROOT/'baselines/Baselines_outputs/runs/human_skin/2026-09-18/20260918_human_skin_benchmark_final'
ADDED=ROOT/'baselines/Baselines_outputs/additional_spatial_assets_20261016/human_skin/additional_spatial'
FREQ=ROOT/'outputs/runs/freqniche/human_skin/2026-09-18/20260918_human_skin_evaluation_final'
DIAGNOSTICS=ROOT/'Paper/image-gen/human_skin_figure3_20261019'
MAIN='fig_human_skin_spatial_domain_umap'
ORDER=['Reference niches','FreqNiche','CellCharter','GraphST','SEDR','BANKSY','SpaGCN','CellNiche']
OLD={'CellCharter':'cellcharter_canonical_seed_42','BANKSY':'banksy_controlled_seed_42','CellNiche':'cellniche_controlled_seed_42'}
UMAP_ORDER=['Raw PCA','FreqNiche','CellCharter','GraphST','SEDR','BANKSY','SpaGCN','CellNiche']
LINEAR=ROOT/'baselines/Baselines_outputs/runs/human_skin/2026-10-17/20261017_linear_controls'


def metrics(truth,pred):
    return {'ari':float(adjusted_rand_score(truth,pred)),'nmi':float(normalized_mutual_info_score(truth,pred))}


def load(out):
    frame=pd.read_csv(DIAGNOSTICS/'plot_data.csv')
    assert len(frame)==8000 and frame.cell_barcode.is_unique
    frame=frame.rename(columns={'pred_freqniche':'pred_FreqNiche'})
    sources={};embeddings={};resolutions={'FreqNiche':.8}
    def record(path):
        sources[str(path.relative_to(ROOT))]=sha(path)
    record(DIAGNOSTICS/'plot_data.csv');record(DIAGNOSTICS/'data_binding.json')
    freq_path=FREQ/'embeddings/freqniche_st_v1_trial24_seed_42_test_z.npy'
    record(freq_path);embeddings['FreqNiche']=np.load(freq_path)
    ids=frame.cell_barcode.tolist();donor_mask=frame.donor_id.eq('D028').to_numpy()
    expected_ids=frame.loc[donor_mask,'cell_barcode'].tolist()
    for name,stem in OLD.items():
        path=BASE/'embeddings'/((stem.replace('_seed_42','')+'_test_z.npy') if name=='BANKSY' else stem+'_test_z.npy')
        result_path=BASE/'results'/(stem+'.json')
        record(path);record(result_path)
        report=json.loads(result_path.read_text())['common_readout']
        resolution=report['stability_selection']['optimal_r_star'];resolutions[name]=resolution
        z=np.load(path);assert len(z)==8000 and np.isfinite(z).all()
        embeddings[name]=z
        a=ad.AnnData(X=z)
        sc.pp.neighbors(a,n_neighbors=15,use_rep='X',random_state=42)
        sc.tl.leiden(a,resolution=resolution,key_added='pred',random_state=42)
        frame['pred_'+name]=a.obs.pred.to_numpy().astype(int)
        original=report['test_evaluation']
        check=[]
        for donor,block in frame.groupby('donor_id',sort=False):
            m=metrics(block.niche,block['pred_'+name]);check.append(m)
            saved=original['per_donor_metrics'][donor]
            for key in ['ari','nmi']:assert np.isclose(m[key],saved[key],atol=1e-12,rtol=0),(name,donor,key,m,saved)
        for key in ['ari','nmi']:
            assert np.isclose(np.mean([m[key] for m in check]),original['test_donor_macro_'+key],atol=1e-12,rtol=0)
        print('Verified frozen readout:',name,flush=True)
    for name in ['GraphST','SEDR','SpaGCN']:
        directory=ADDED/name.lower()/'seed_42'
        result=json.loads((directory/'test_result.json').read_text());record(directory/'test_result.json')
        resolutions[name]=result['leiden_resolution'];parts=[]
        for donor in ['D028','D082','D145','D151']:
            bundle=directory/'assets/test'/donor
            manifest=json.loads((bundle/'artifact_manifest.json').read_text())
            record(bundle/'artifact_manifest.json')
            path=bundle/'assignments.csv';record(path)
            assert sha(path)==manifest['files']['assignments.csv']['sha256']
            block=pd.read_csv(path);parts.append(block)
            m=metrics(block.reference_label,block.predicted_cluster)
            for key in ['ari','nmi']:assert np.isclose(m[key],result['per_donor'][donor][key],atol=1e-12,rtol=0)
            if donor=='D028':
                path=bundle/'embedding.npy';record(path)
                assert sha(path)==manifest['files']['embedding.npy']['sha256']
                z=np.load(path)
                cells=pd.read_csv(bundle/'cells.csv');assert cells.cell_barcode.tolist()==block.cell_barcode.tolist()
                indices=pd.Index(cells.cell_barcode).get_indexer(expected_ids);assert (indices>=0).all()
                embeddings[name]=z[indices]
        joined=pd.concat(parts).set_index('cell_barcode')
        assert joined.index.is_unique and set(joined.index)==set(ids)
        joined=joined.loc[ids]
        for key in ['donor_id','sample_id']:assert np.array_equal(joined[key].to_numpy(),frame[key])
        assert np.array_equal(joined.reference_label.to_numpy(),frame.niche)
        frame['pred_'+name]=joined.predicted_cluster.to_numpy()
        print('Verified donor-wise assignments:',name,flush=True)
    # Adopt the same Raw PCA assets as Table 2, not the historical val-8k PCA.
    path=LINEAR/'raw_pca/test_embedding.npy';record(path)
    retained=json.loads((LINEAR/'paper_asset_binding.json').read_text())['files']
    retained={k.replace('\\','/'):v for k,v in retained.items()}
    assert sha(path)==retained['raw_pca/test_embedding.npy']['sha256']
    centers_path=LINEAR/'test_centers.csv';record(centers_path)
    cells=pd.read_csv(centers_path);raw=np.load(path)
    assert len(raw)==len(cells)==8000 and cells.cell_barcode.is_unique
    indices=pd.Index(cells.cell_barcode).get_indexer(ids);assert (indices>=0).all()
    embeddings['Raw PCA']=raw[indices]
    result_path=LINEAR/'raw_pca/seed_42/test_result.json';record(result_path)
    result=json.loads(result_path.read_text())
    assignment_path=LINEAR/'raw_pca/seed_42/assignments.csv';record(assignment_path)
    assert sha(assignment_path)==result['assignments_sha256']
    block=pd.read_csv(assignment_path).set_index('cell_barcode').loc[ids]
    assert np.array_equal(block.niche,frame.niche)
    for key in ['donor_id','sample_id']:assert np.array_equal(block[key],frame[key])
    frame['pred_Raw PCA']=block.predicted_cluster.to_numpy()
    resolutions['Raw PCA']=result['frozen_r_star']
    for donor,group in frame.groupby('donor_id',sort=False):
        m=metrics(group.niche,group['pred_Raw PCA'])
        for key in ['ari','nmi']:assert np.isclose(m[key],result['per_donor_metrics'][donor][key],atol=1e-12,rtol=0)
    print('Verified adopted Raw PCA assets',flush=True)
    # FreqNiche cache is also checked against the adopted seed-42 donor metrics.
    result_path=FREQ/'results/freqniche_st_v1_trial24_seed_42.json';record(result_path)
    original=json.loads(result_path.read_text())['common_readout']['test_evaluation']
    for donor,block in frame.groupby('donor_id',sort=False):
        m=metrics(block.niche,block.pred_FreqNiche)
        for key in ['ari','nmi']:assert np.isclose(m[key],original['per_donor_metrics'][donor][key],atol=1e-12,rtol=0)
    frame.to_csv(out/'all_test_labels_seed42.csv',index=False)
    selected=frame[donor_mask & frame.sample_id.str.endswith('region_1').to_numpy()].copy()
    assert len(selected)==669
    selected.to_csv(out/'slice_labels_seed42.csv',index=False)
    background_path=FREQ/'full_slice/d028_region1_full_11293_pred.csv';record(background_path)
    background=pd.read_csv(background_path);assert len(background)==11293
    return frame,selected,background,embeddings,sources,resolutions


def setup():
    for directory in [Path('/mnt/c/Windows/Fonts'),Path('C:/Windows/Fonts')]:
        for name in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
            if (directory/name).exists():font_manager.fontManager.addfont(str(directory/name))
    font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'pdf.fonttype':42,
                         'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.55,'axes.unicode_minus':False})


def axes_grid(rows,cols,size,leg_ratio=.5):
    fig=plt.figure(figsize=size)
    grid=fig.add_gridspec(rows,cols,left=.025,right=.985,bottom=.025,top=.925,hspace=.15,wspace=.15)
    axes=[];legs=[]
    for i in range(rows*cols):
        sub=grid[i//cols,i%cols].subgridspec(2,1,height_ratios=[1,leg_ratio],hspace=.10)
        ax=fig.add_subplot(sub[0]);axes.append(ax);legs.append(fig.add_subplot(sub[1]))
        ax.set_box_aspect(.82);ax.set_aspect('equal',adjustable='datalim');ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_edgecolor('#cccccc')
    return fig,axes,legs


def save(fig,stem,out,metadata,publish):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for ax in fig.axes:
        item=ax.get_legend()
        if item is not None:
            box=item.get_window_extent(renderer)
            assert box.x0>=0 and box.x1<=fig.bbox.width and box.y0>=0,'Legend exceeds figure'
    for ext in ['pdf','svg','png']:fig.savefig(out/(stem+'.'+ext),dpi=600,facecolor='white')
    plt.close(fig)
    metadata['rendering']={'font':'Times New Roman','raster_dpi':600,'scatter_rasterized':True,
                           'vector_text_axes_legends':True,'titles_centered':True,'legends_below_panels':True,
                           'script_sha256':sha(Path(__file__))}
    (out/(stem+'_provenance.json')).write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    if publish:
        for suffix in ['.pdf','.svg','.png','_provenance.json']:
            shutil.copyfile(out/(stem+suffix),ROOT/'Paper/FreqNiche-zh/figures'/(stem+suffix))
    print('SAVED',stem,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'Paper/image-gen/human_skin_comparison_20261019')
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--umap-only',action='store_true',help='Update supplementary comparison only; preserve main Figure 3 and single-model diagnostics')
    args=parser.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);setup()
    frame,selected,background,z,sources,resolutions=load(out)
    niches=sorted(frame.niche.unique());nc=palette(niches,'tab10')
    if not args.umap_only:
        fig,axes,legs=axes_grid(2,4,(8.8,5.8),.55)
        contour=background[['center_x','center_y']].to_numpy();xy=selected[['center_x','center_y']].to_numpy()
        record=[]
        maps=['tab10','tab20','Set2','tab20b','tab20c','Dark2','tab20','Paired']
        for i,name in enumerate(ORDER):
            ax=axes[i];ax.set_title(f'({chr(65+i)}) {name}',fontsize=12,loc='center',pad=7)
            ax.scatter(*contour.T,c='#dedede',s=1.4,alpha=.55,linewidths=0,rasterized=True)
            if i==0:
                cats=[c for c in niches if c in selected.niche.values]
                scatter(ax,xy,selected.niche,cats,nc,6,.95);legend(legs[i],cats,nc,ncol=2)
            else:
                values=selected['pred_'+name];cats=sorted(values.unique())
                colors=palette(sorted(frame.pred_FreqNiche.unique()),'tab20') if name=='FreqNiche' else palette(cats,maps[i])
                assert len({colors[c] for c in cats})==len(cats),'Insufficient unique colors'
                scatter(ax,xy,values,cats,colors,6,.95)
                legend(legs[i],cats,colors,[f'C{c}' for c in cats],ncol=min(4,len(cats)))
                m=metrics(selected.niche,values);record.append({'method':name,**m,'cells':669,'active_clusters':len(cats)})
                ax.text(.97,.035,f"ARI = {m['ari']:.3f}\nNMI = {m['nmi']:.3f}",transform=ax.transAxes,
                        ha='right',va='bottom',fontsize=10.5,linespacing=1.1,
                        bbox={'facecolor':'white','edgecolor':'none','alpha':.88,'pad':1.2})
        fig.canvas.draw()
        for ax in axes[1:]:ax.set_xlim(axes[0].get_xlim());ax.set_ylim(axes[0].get_ylim())
        pd.DataFrame(record).to_csv(out/'slice_metrics_seed42.csv',index=False)
        save(fig,MAIN,out,{'scope':'D028 region_1 fixed 669 evaluation centers, 11293 background cells',
            'panel_order':ORDER,'seed':42,'metrics':record,'sources':sources,'frozen_resolutions':resolutions,
            'note':'No fitting/selection. Local metrics are not donor-macro Table 2 means; cluster colors independent across methods. New baselines use original donor-wise readouts.'},args.publish)
    # Same donor for all UMAPs: donor-wise latent axes must not be concatenated.
    mask=frame.donor_id.eq('D028').to_numpy();truth=frame.loc[mask,'niche'].to_numpy()
    fig,axes,legs=axes_grid(2,4,(8.8,6.2),.58)
    umap_records=[]
    for i,name in enumerate(UMAP_ORDER):
        embedding=z[name] if name in ['GraphST','SEDR','SpaGCN'] else z[name][mask]
        assert len(embedding)==2000
        slug=name.lower().replace(' ','_')
        cache=out/(slug+'_d028_umap.npy');binding=out/(slug+'_umap_binding.json')
        digest=hashlib.sha256(embedding.tobytes()).hexdigest()
        if cache.exists():
            assert json.loads(binding.read_text())['input_sha256']==digest
            coords=np.load(cache)
        else:
            work=ad.AnnData(X=embedding)
            sc.pp.neighbors(work,n_neighbors=30,use_rep='X',random_state=42)
            sc.tl.umap(work,min_dist=.3,random_state=42);coords=work.obsm['X_umap']
            np.save(cache,coords);binding.write_text(json.dumps({'input_sha256':digest,'n_neighbors':30,'min_dist':.3,'seed':42}),encoding='utf-8')
        pd.DataFrame({'cell_barcode':frame.loc[mask,'cell_barcode'],'niche':truth,'umap_1':coords[:,0],'umap_2':coords[:,1]}).to_csv(out/(slug+'_d028_umap.csv'),index=False)
        axes[i].set_title(f'({chr(65+i)}) {name}',fontsize=12,loc='center',pad=8)
        scatter(axes[i],coords,truth,niches,nc,3.5)
        m=metrics(truth,frame.loc[mask,'pred_'+name]);annotation(axes[i],f"NMI = {m['nmi']:.3f}")
        legend(legs[i],niches,nc,ncol=2);umap_records.append({'method':name,'input_dimensions':embedding.shape[1],**m})
    save(fig,'fig_supp_human_skin_d028_umap_comparison',out,{'scope':'same 2000 D028 centers in all methods, reference-niche colors shared',
        'metrics':umap_records,'panel_order':UMAP_ORDER,'layout':'2 rows x 4 columns','sources':sources,'projection':{'n_neighbors':30,'min_dist':.3,'seed':42},
        'note':'Each method projected independently; no latent alignment, no clustering on UMAP. Displayed NMI uses original frozen high-dimensional readout.'},args.publish)
    if not args.umap_only:
        # Retain the four-color FreqNiche diagnostics, outside the main comparison.
        fig,axes,legs=axes_grid(2,2,(7.0,6.5),.48)
        coords=frame[['umap_1','umap_2']].to_numpy()
        for i,(col,title,cmap) in enumerate([('pred_FreqNiche','Predicted clusters','tab20'),('niche','Reference niches','tab10'),
                                           ('cell_type','Cell types','tab20'),('donor_id','Donors','Set2')]):
            cats=sorted(frame[col].unique());colors=palette(cats,cmap)
            axes[i].set_title(f'({chr(65+i)}) {title}',fontsize=12,loc='center',pad=8)
            scatter(axes[i],coords,frame[col],cats,colors,3)
            legend(legs[i],cats,colors,[f'C{c}' for c in cats] if i==0 else None,ncol=4 if i in [0,2] else 3)
            if i==0:annotation(axes[i],'K = 16')
            else:annotation(axes[i],f"NMI = {normalized_mutual_info_score(frame[col],frame.pred_FreqNiche):.3f}")
        save(fig,'fig_supp_freqniche_representation_diagnostics',out,{'scope':'original 8000-center FreqNiche UMAP, four annotations',
            'sources':sources,'note':'Pooled 8000-center NMI, not donor-macro; donor structure retained as a diagnostic.'},args.publish)
    pd.DataFrame(umap_records).to_csv(out/'d028_umap_metrics_seed42.csv',index=False)
    print(pd.DataFrame(umap_records).to_string(index=False),flush=True)


if __name__=='__main__':main()
