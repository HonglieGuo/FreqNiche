#!/usr/bin/env python3
"""CosMx six-variant ablation; no depth task or automatic manuscript edits.

Frozen Validation selection precedes Test access for all 18 variant/seed pairs.
Reusable weights are copied into the new run; new weights, embeddings and cell
assignments are saved with hashes and checked by independent checkpoint reload.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time


class Tee:
    def __init__(self, console, log): self.console, self.log = console, log
    def write(self, value): self.console.write(value); self.log.write(value); self.log.flush()
    def flush(self): self.console.flush(); self.log.flush()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.freqniche_st.runtime_paths import project_path, validated_source_bindings
COSMX_RUN = ROOT/'outputs/runs/freqniche/cosmx_nsclc/2026-09-17/20260917_cosmx_nsclc_final'
COSMX_NO_COORD_RUN = ROOT/'outputs/runs/ablations/cosmx_nsclc/2026-09-23/20260923_cosmx_nsclc_nocoord'
DATA = ROOT/'datasets/h5ad/cosmx_human_nsclc/cosmx_human_nsclc.h5ad'
SEEDS = (42, 123, 2024)
VARIANTS = ('full', 'no_coords', 'no_low', 'no_high', 'raw_only', 'deterministic')
LABELS = dict(zip(VARIANTS, ('Full FreqNiche', 'w/o Coordinates',
    'w/o Low-frequency condition', 'w/o High-frequency condition', 'Raw-only', 'Deterministic')))
MODEL = dict(in_features=960, k_target=32, d_model=128, n_enc_layers=3,
             n_heads=8, mlp_ratio=2.0, lambda_var=1.0, lambda_cov=0.04, mode='full_freq')
PROTOCOL = dict(seeds=list(SEEDS), variants=list(VARIANTS), radius_um=50., k_max=64,
                m_max=32, rho=.5, p_center_mask=.3, learning_rate=.001,
                weight_decay=.001, gradient_clip=1., resolutions=[.2,.4,.6,.8,1.],
                k_min=5, k_max_clusters=25, readout_seed=42, inference_batch_size=128,
                test_donor='Lung9', model=MODEL, diffusion_layers=2, timesteps=50)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sign(value):
    result = copy.deepcopy(value)
    result['signature'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return result


def verify_signature(value, field='signature'):
    result = copy.deepcopy(value); signature = result.pop(field)
    if hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest() != signature:
        raise ValueError(f'Invalid {field}')


def runtime():
    global np, pd, sc, sp, torch, DataLoader, SubgraphDataset, collate_subgraphs
    global FreqNicheDiTModel, NicheTransformerReconModel, TrainOnlyWhitening
    global build_global_spatial_graph, adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp
    import torch
    from torch.utils.data import DataLoader
    from src.freqniche_st.dataset import SubgraphDataset, collate_subgraphs
    from src.freqniche_st.models.freqniche_model import FreqNicheDiTModel
    from src.freqniche_st.models.transformer_recon import NicheTransformerReconModel
    from src.freqniche_st.whitening import TrainOnlyWhitening
    from src.freqniche_st.graph import build_global_spatial_graph
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def source_paths():
    return [Path(__file__).resolve(),
            *sorted((ROOT/'src/freqniche_st').glob('*.py')),
            *sorted((ROOT/'src/freqniche_st/models').glob('*.py'))]


def check_files(base, files):
    if Path(base).resolve() == ROOT:
        validated_source_bindings(files)
        return
    for relative, digest in files.items():
        if sha(Path(base)/relative) != digest: raise ValueError(f'Asset changed: {base}/{relative}')


def model_for(variant, config=None):
    config = dict(MODEL if config is None else config)
    if variant == 'deterministic': return NicheTransformerReconModel(**config, n_dec_layers=2)
    return FreqNicheDiTModel(**config, n_dit_layers=2, timesteps=50)


def intervene(batch, variant, device):
    """Same intervention for training, validation objective and inference."""
    keys = ('X_vis', 'delta_s_vis', 'vis_pad_mask', 'L_vis_cond', 'H_vis_cond',
            'H_0_target', 'delta_s_target', 'target_mask')
    used = {key: batch[key].to(device) for key in keys if key in batch}
    if variant in ('no_low', 'raw_only'): used['L_vis_cond'] = torch.zeros_like(used['L_vis_cond'])
    if variant in ('no_high', 'raw_only'): used['H_vis_cond'] = torch.zeros_like(used['H_vis_cond'])
    if variant == 'no_coords':
        for key in ('delta_s_vis', 'delta_s_target'):
            if key in used: used[key] = torch.zeros_like(used[key])
    return used


def reuse_sources(audit):
    result = {}
    for variant, base in [('full', COSMX_RUN), ('no_coords', COSMX_NO_COORD_RUN)]:
        selection = load(base/'resolution_selection.json')
        verify_signature(selection, 'selection_manifest_sha256')
        if selection['bound_input_assets']['data_audit_manifest']['sha256'] != sha(COSMX_RUN/'cosmx_data_audit_manifest.json'):
            raise ValueError('Reusable checkpoint belongs to a different data contract')
        for seed in SEEDS:
            bound = selection['bound_input_assets']['checkpoints'][str(seed)]
            checkpoint = project_path(bound['path']); history_path = checkpoint.parent/'training_history.json'
            if sha(checkpoint) != bound['sha256']: raise ValueError(f'Checkpoint changed: {checkpoint}')
            history = load(history_path)
            if history['checkpoint_sha256'] != bound['sha256']: raise ValueError('History/checkpoint mismatch')
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            hp = payload['hyperparameters']; expected = {k:v for k,v in MODEL.items() if k != 'lambda_var'}
            expected.update(n_dit_layers=2, timesteps=50, batch_size=64, lr=.001, weight_decay=.001, radius_um=50.)
            if any(hp.get(k) != v for k,v in expected.items()): raise ValueError(f'Model configuration mismatch: {checkpoint}')
            if payload['seed'] != seed: raise ValueError('Seed mismatch')
            checked_model=model_for(variant)
            checked_model.load_state_dict(payload['model_state_dict'],strict=True)
            del checked_model
            if variant == 'no_coords':
                item = selection['bound_input_assets']['attestations'][str(seed)]; path = project_path(item['path'])
                if sha(path) != item['sha256']: raise ValueError('NoCoord attestation changed')
                attestation = load(path); verify_signature(attestation, 'self_signature_sha256')
                bound_assets = attestation['bound_assets']
                expected_assets = {'model_checkpoint_sha256': sha(checkpoint),
                    'training_history_sha256': sha(history_path), 'dataset_h5ad_sha256': audit['h5ad_sha256'],
                    'graph_cache_P_sha256': audit['train_graph_cache']['P_sha256'],
                    'graph_cache_A_sha256': audit['train_graph_cache']['A_sha256'],
                    'whitener_sha256': audit['whitener']['sha256']}
                if any(bound_assets[k] != v for k,v in expected_assets.items()): raise ValueError('NoCoord inputs differ')
            result[f'{variant}/{seed}'] = {'checkpoint': str(checkpoint.relative_to(ROOT)),
                'checkpoint_sha256': bound['sha256'], 'history': str(history_path.relative_to(ROOT)),
                'history_sha256': sha(history_path), 'selection': str((base/'resolution_selection.json').relative_to(ROOT)),
                'selection_sha256': sha(base/'resolution_selection.json')}
            if variant=='no_coords':
                result[f'{variant}/{seed}'].update(attestation=str(path.relative_to(ROOT)),attestation_sha256=sha(path))
    return result


def preflight(args):
    audit = load(COSMX_RUN/'cosmx_data_audit_manifest.json')
    paths = {DATA: audit['h5ad_sha256'], COSMX_RUN/'whitener.pt': audit['whitener']['sha256'],
        COSMX_RUN/'cache/cosmx_train_graph_A.npz': audit['train_graph_cache']['A_sha256'],
        COSMX_RUN/'cache/cosmx_train_graph_P.npz': audit['train_graph_cache']['P_sha256']}
    for split, name in [('train','train'), ('validation','val'), ('test','test')]:
        path = COSMX_RUN/f'{name}_centers.json'; cohort = load(path)
        if len(cohort['cell_barcodes']) != audit['cohorts'][split]['count']: raise ValueError('Center count changed')
        for ordered, key in [(False, 'sorted_barcode_sha256'), (True, 'ordered_barcode_sha256')]:
            items = cohort['cell_barcodes'] if ordered else sorted(cohort['cell_barcodes'])
            digest = hashlib.sha256(json.dumps(items, separators=(',',':')).encode()).hexdigest()
            if digest != audit['cohorts'][split][key]: raise ValueError('Center order changed')
        paths[path] = audit['cohorts'][split]['file_sha256']
    paths[COSMX_RUN/'val_bootstrap_folds.json'] = audit['cohorts']['val_bootstrap_folds']['file_sha256']
    for path, expected in paths.items():
        if sha(path) != expected: raise ValueError(f'Frozen input changed: {path}')
    read_whitener(COSMX_RUN/'whitener.pt')
    folds = load(COSMX_RUN/'val_bootstrap_folds.json')
    for i in range(10):
        indices = np.asarray(folds[f'fold_{i}']['indices_in_val_centers'])
        if len(indices)!=8000 or len(np.unique(indices))!=8000 or indices.min()<0 or indices.max()>=10000:
            raise ValueError('Invalid Validation subsampling fold')
    reused = reuse_sources(audit) if not args.no_reuse_existing else {}
    if reused and (args.max_epochs,args.patience,args.batch_size,args.min_delta)!=(50,7,64,1e-4):
        raise ValueError('Weight reuse requires original 50/7/64/1e-4 budget; use --no-reuse-existing for a different budget')
    return audit, {str(p.relative_to(ROOT)):h for p,h in paths.items()}, reused


def save_array(path, value):
    temp=Path(path).with_name(Path(path).name+'.tmp')
    with temp.open('wb') as stream: np.save(stream,value,allow_pickle=False)
    temp.replace(path)


def prepare_context(out, split, audit):
    base=out/'inputs'/split; manifest=base/'manifest.json'
    if manifest.exists():
        meta=load(manifest); check_files(base,meta['files'])
        if meta['dataset_sha256']!=audit['h5ad_sha256']: raise ValueError('Cached dataset mismatch')
    else:
        base.mkdir(parents=True,exist_ok=True); a=sc.read_h5ad(DATA,backed='r')
        try:
            if a.n_vars!=960 or not np.array_equal(a.obsm['spatial'],a.obsm['spatial_um']): raise ValueError('Coordinate/panel contract changed')
            indices=np.flatnonzero(a.obs['split'].to_numpy()==split)
            name={'train':'train','validation':'val','test':'test'}[split]
            cohort=load(COSMX_RUN/f'{name}_centers.json'); center_globals=np.asarray(cohort['global_indices'],dtype=np.int64)
            locations=np.searchsorted(indices,center_globals)
            if np.any(locations>=len(indices)) or not np.array_equal(indices[locations],center_globals): raise ValueError('Centers outside requested split')
            if list(a.obs_names[center_globals])!=cohort['cell_barcodes']: raise ValueError('Center row alignment mismatch')
            sub=a[indices].to_memory(); coordinates=np.asarray(sub.obsm['spatial_um'],dtype=np.float32)
            if not sub.obs.spatial_unit.eq('micron').all(): raise ValueError('Coordinate units changed')
            if split=='test' and set(sub.obs.donor_id.astype(str))!={'Lung9'}: raise ValueError('Unexpected test donor')
            sc.pp.normalize_total(sub,target_sum=1e4); sc.pp.log1p(sub)
            expression=sub.X.toarray() if sp.issparse(sub.X) else np.asarray(sub.X)
            expression=np.asarray(expression,dtype=np.float32)
            if not np.isfinite(expression).all(): raise ValueError('Nonfinite normalized expression')
            save_array(base/'X.npy',expression); del expression
            save_array(base/'coordinates.npy',coordinates); save_array(base/'center_local_indices.npy',locations)
            cells=pd.DataFrame({'cell_barcode':list(sub.obs_names),'global_index':indices,
                'donor_id':sub.obs.donor_id.astype(str).to_numpy(),'sample_id':sub.obs.sample_id.astype(str).to_numpy()})
            cells.to_csv(base/'cells.csv',index=False); centers=cells.iloc[locations].copy()
            if split=='test': centers['reference_niche']=sub.obs['niche'].iloc[locations].astype(str).to_numpy()
            centers.to_csv(base/'centers.csv',index=False)
            pd.DataFrame({'gene':list(sub.var_names)}).to_csv(base/'genes.csv',index=False)
            if split=='train':
                for key in ['A','P']: shutil.copyfile(COSMX_RUN/f'cache/cosmx_train_graph_{key}.npz',base/f'{key}.npz')
            else:
                P,A=build_global_spatial_graph(coordinates,sub.obs.sample_id.to_numpy(),radius=50.,max_neighbors=None)
                sp.save_npz(base/'A.npz',A); sp.save_npz(base/'P.npz',P)
            del sub
        finally: a.file.close()
        files={p.name:sha(p) for p in base.iterdir() if p.is_file() and p.name!='manifest.json'}
        write(manifest,{'split':split,'dataset_sha256':audit['h5ad_sha256'],'n_context':len(indices),
            'n_centers':len(locations),'normalization':'normalize_total(1e4), log1p','files':files})
    X=np.load(base/'X.npy',mmap_mode='r'); A,P=sp.load_npz(base/'A.npz'),sp.load_npz(base/'P.npz')
    if A.shape!=(len(X),len(X)) or not np.allclose(np.asarray(P.sum(1)).ravel(),1,atol=1e-5): raise ValueError('Graph shape/normalization mismatch')
    sample=pd.read_csv(base/'cells.csv',dtype={'sample_id':str}).sample_id.to_numpy()
    for first in range(0,len(X),10000):
        edges=A[first:first+10000].tocoo()
        if not np.all(sample[first+edges.row]==sample[edges.col]): raise ValueError('Cross-sample edges found')
    return {'X_global':X,'A_global':A,'P_global':P,'coords_global':np.load(base/'coordinates.npy'),
            'center_indices':np.load(base/'center_local_indices.npy')}


def make_dataset(context, *, seed, whitener=None, train=False, objective=False):
    return SubgraphDataset(**context,whitener=whitener,k_max=64,m_max=32,rho=.5,p_center_mask=.3,
                           is_train=train,is_val_objective=objective,seed=seed)


def read_whitener(path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    result=TrainOnlyWhitening(k_components=32,lambda_min=1e-4); result.__dict__.update(state); result.is_fitted=True
    if result.n_components_fitted_!=32 or result.whitening_matrix_.shape!=(960,32): raise ValueError('Whitening shape mismatch')
    return result


def torch_save(path,value):
    temp=path.with_name(path.name+'.tmp'); torch.save(value,temp); temp.replace(path)


def load_model(directory,variant,device):
    model=model_for(variant).to(device)
    state=torch.load(directory/'model_best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state['model_state_dict'],strict=True)
    return model


def train_seed(args,directory,variant,seed,train_context,val_context,whitener,device):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if device.type=='cuda': torch.cuda.manual_seed_all(seed)
    tr=make_dataset(train_context,seed=seed,whitener=whitener,train=True)
    va=make_dataset(val_context,seed=seed,whitener=whitener,objective=True)
    tr_loader=DataLoader(tr,batch_size=args.batch_size,shuffle=True,collate_fn=collate_subgraphs,num_workers=0)
    va_loader=DataLoader(va,batch_size=args.batch_size,shuffle=False,collate_fn=collate_subgraphs,num_workers=0)
    model=model_for(variant).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    history=[]; best=float('inf'); best_epoch=0; wait=0; start=1; last=directory/'checkpoint_last.pt'
    if last.exists():
        if not args.resume: raise FileExistsError(f'Partial seed exists; use --resume: {directory}')
        state=torch.load(last,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model_state_dict']); optimizer.load_state_dict(state['optimizer'])
        history=state['history']; best=state['best']; best_epoch=state['best_epoch']; wait=state['wait']; start=state['epoch']+1
        torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
        tr.rng.set_state(state['dataset_rng'])
        if device.type=='cuda' and state['cuda_rng'] is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
    for epoch in range(start,args.max_epochs+1):
        if wait>=args.patience: break
        began=time.perf_counter(); model.train(); losses=[]
        for batch in tr_loader:
            optimizer.zero_grad(set_to_none=True); result=model(**intervene(batch,variant,device)); loss=result['loss']
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); validation=[]; generator=torch.Generator(device=device).manual_seed(seed+99999)
        with torch.no_grad():
            for batch in va_loader:
                used=intervene(batch,variant,device)
                result=model(**used) if variant=='deterministic' else model(**used,generator=generator)
                loss=result['loss_recon' if variant=='deterministic' else 'loss_diff']
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite validation loss')
                validation.append(float(loss))
        val=float(np.mean(validation)); improved=val<best-args.min_delta
        if improved:
            best=val; best_epoch=epoch; wait=0
            torch_save(directory/'model_best.pt',{'variant':variant,'seed':seed,'model_config':MODEL,
                'best_epoch':best_epoch,'best_val_loss':best,'model_state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}})
        else: wait+=1
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_objective':val,'seconds':time.perf_counter()-began}
        history.append(row); write(directory/'history.json',history)
        torch_save(last,{'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
            'history':history,'best':best,'best_epoch':best_epoch,'wait':wait,'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
            'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'dataset_rng':tr.rng.get_state()})
        print(variant,seed,row,'best_epoch',best_epoch,flush=True)
    if not (directory/'model_best.pt').exists(): raise RuntimeError('No valid checkpoint saved')
    write(directory/'training.json',{'variant':variant,'seed':seed,'origin':'trained_in_this_run',
        'best_epoch':best_epoch,'best_validation_objective':best,'epochs':len(history),
        'checkpoint_sha256':sha(directory/'model_best.pt'),'history_sha256':sha(directory/'history.json')})


def extract(model,context,variant,device):
    loader=DataLoader(make_dataset(context,seed=42),batch_size=128,shuffle=False,collate_fn=collate_subgraphs,num_workers=0)
    arrays=[]; model.eval()
    with torch.no_grad():
        for batch in loader:
            used=intervene(batch,variant,device)
            z=model.encode_niche(**{k:used[k] for k in ['X_vis','delta_s_vis','vis_pad_mask','L_vis_cond','H_vis_cond']})
            arrays.append(z.cpu().numpy())
    result=np.concatenate(arrays)
    if result.shape!=(10000,128) or not np.isfinite(result).all(): raise ValueError('Embedding shape/finite check failed')
    return result


def select_resolution(z,folds,seed):
    """Same label-free stability rule as the frozen CosMx evaluation."""
    indices=[np.asarray(folds[f'fold_{i}']['indices_in_val_centers'],dtype=np.int64) for i in range(10)]
    scores={}; ks={}
    for resolution in PROTOCOL['resolutions']:
        predictions=[]
        for i,idx in enumerate(indices):
            a=sc.AnnData(X=z[idx].copy()); sc.pp.neighbors(a,n_neighbors=15,use_rep='X',random_state=seed+i)
            sc.tl.leiden(a,resolution=resolution,key_added='cluster',random_state=seed+i)
            predictions.append(a.obs.cluster.to_numpy().astype(int))
        ks[resolution]=float(np.mean([len(np.unique(p)) for p in predictions])); pairs=[]
        for i in range(10):
            for j in range(i+1,10):
                common,pi,pj=np.intersect1d(indices[i],indices[j],return_indices=True)
                if len(common)>10: pairs.append(float(adjusted_rand_score(predictions[i][pi],predictions[j][pj])))
        scores[resolution]=float(np.mean(pairs)) if pairs else 0.
        print('VALIDATION_STABILITY',seed,resolution,'score',scores[resolution],'mean_K',ks[resolution],flush=True)
    valid=[r for r in PROTOCOL['resolutions'] if 5<=ks[r]<=25]
    if valid: best=max(valid,key=lambda r:(scores[r],-r))
    else: best=min(PROTOCOL['resolutions'],key=lambda r:(max(5-ks[r],ks[r]-25,0),-scores[r]))
    return best,{'stability_scores':{str(k):v for k,v in scores.items()},'mean_clusters':{str(k):v for k,v in ks.items()},'valid_resolutions':valid}


def completed_selection(directory):
    selected=load(directory/'selection.json'); verify_signature(selected); check_files(directory,selected['assets'])
    return selected


def train_validation(args,out,audit,reused,device):
    if sha(out/'inputs/whitener.pt')!=audit['whitener']['sha256']: raise ValueError('Copied whitener changed')
    val_context=prepare_context(out,'validation',audit); train_context=None; whitener=read_whitener(out/'inputs/whitener.pt')
    folds=load(COSMX_RUN/'val_bootstrap_folds.json')
    for variant in VARIANTS:
        for seed in SEEDS:
            directory=out/variant/f'seed_{seed}'; directory.mkdir(parents=True,exist_ok=True)
            write(out/'status.json',{'status':'running','phase':'train_validation','variant':variant,'seed':seed})
            if (directory/'selection.json').exists(): completed_selection(directory); continue
            training_path=directory/'training.json'
            if not training_path.exists():
                source=reused.get(f'{variant}/{seed}')
                if source:
                    shutil.copyfile(project_path(source['checkpoint']),directory/'model_best.pt'); shutil.copyfile(project_path(source['history']),directory/'history.json')
                    snapshots={'source_selection.json':source['selection']}
                    if 'attestation' in source: snapshots['source_attestation.json']=source['attestation']
                    for name,path in snapshots.items(): shutil.copyfile(ROOT/path,directory/name)
                    old=load(project_path(source['history']))
                    write(training_path,{'variant':variant,'seed':seed,'origin':'verified_existing_weights','source':source,
                        'best_epoch':old['best_epoch'],'epochs':len(old['train_loss_history']),
                        'checkpoint_sha256':sha(directory/'model_best.pt'),'history_sha256':sha(directory/'history.json'),
                        'source_snapshots':{name:sha(directory/name) for name in snapshots},
                        'note':'NoCoord intervention is explicit during training/inference; checkpoint mode metadata is not used to infer it.'})
                else:
                    if train_context is None: train_context=prepare_context(out,'train',audit)
                    train_seed(args,directory,variant,seed,train_context,val_context,whitener,device)
            training=load(training_path)
            check_files(directory,{'model_best.pt':training['checkpoint_sha256'],'history.json':training['history_sha256']})
            check_files(directory,training.get('source_snapshots',{}))
            model=load_model(directory,variant,device); z=extract(model,val_context,variant,device); del model
            save_array(directory/'validation_embeddings.npy',z)
            shutil.copyfile(out/'inputs/validation/centers.csv',directory/'validation_centers.csv')
            r_star,report=select_resolution(z,folds,seed)
            selected={'variant':variant,'seed':seed,'r_star':r_star,'report':report,
                'assets':{name:sha(directory/name) for name in ['model_best.pt','history.json','training.json','validation_embeddings.npy','validation_centers.csv',*training.get('source_snapshots',{})]}}
            write(directory/'selection.json',sign(selected)); print('VALIDATION_COMPLETE',variant,seed,'r*',r_star,flush=True)
    entries={f'{v}/seed_{s}/selection.json':sha(out/v/f'seed_{s}/selection.json') for v in VARIANTS for s in SEEDS}
    entries['inputs/validation/manifest.json']=sha(out/'inputs/validation/manifest.json')
    if (out/'inputs/train/manifest.json').exists(): entries['inputs/train/manifest.json']=sha(out/'inputs/train/manifest.json')
    write(out/'freeze.json',sign({'entries':entries,'run_manifest_sha256':sha(out/'run_manifest.json')}))
    write(out/'status.json',{'status':'validation_complete','completed_selections':18,'test_evaluated':False})


def cluster(z,resolution):
    a=sc.AnnData(X=z.copy()); sc.pp.neighbors(a,n_neighbors=15,use_rep='X',random_state=42)
    sc.tl.leiden(a,resolution=resolution,key_added='cluster',random_state=42)
    return a.obs.cluster.to_numpy().astype(int)


def test_phase(args,out,audit,device):
    if not args.authorized: raise PermissionError('Test phase requires --authorized (the launcher supplies it)')
    freeze=load(out/'freeze.json'); verify_signature(freeze); check_files(out,freeze['entries'])
    if sha(out/'run_manifest.json')!=freeze['run_manifest_sha256']: raise ValueError('Run manifest changed after freeze')
    for variant in VARIANTS:
        for seed in SEEDS: completed_selection(out/variant/f'seed_{seed}')
    # No Test expression, graph or labels loaded before all 18 selections verify.
    context=prepare_context(out,'test',audit)
    centers=pd.read_csv(out/'inputs/test/centers.csv',dtype={'cell_barcode':str,'reference_niche':str}); reference=centers.reference_niche.to_numpy()
    for variant in VARIANTS:
        for seed in SEEDS:
            directory=out/variant/f'seed_{seed}'
            write(out/'status.json',{'status':'running','phase':'test','variant':variant,'seed':seed})
            if (directory/'test_result.json').exists(): check_files(directory,load(directory/'test_result.json')['assets']); continue
            selected=completed_selection(directory); model=load_model(directory,variant,device)
            z=extract(model,context,variant,device); del model; save_array(directory/'test_embeddings.npy',z)
            predictions=cluster(z,selected['r_star']); assignment=centers.copy(); assignment['predicted_cluster']=predictions
            assignment.to_csv(directory/'test_assignments.csv',index=False)
            ari=float(adjusted_rand_score(reference,predictions)); nmi=float(normalized_mutual_info_score(reference,predictions))
            idx=np.random.RandomState(42).choice(len(z),min(5000,len(z)),replace=False)
            asw=float(silhouette_score(z[idx],reference[idx])) if 1<len(np.unique(reference[idx]))<len(idx) else None
            result={'variant':variant,'seed':seed,'r_star':selected['r_star'],'ari':ari,'nmi':nmi,
                'reference_niche_asw':asw,'n_clusters':int(len(np.unique(predictions))),
                'selection_sha256':sha(directory/'selection.json'),
                'assets':{name:sha(directory/name) for name in ['test_embeddings.npy','test_assignments.csv']}}
            write(directory/'test_result.json',result); print('TEST_COMPLETE',variant,seed,'ARI',ari,'NMI',nmi,flush=True)


def audit_run(out, *, complete=False):
    if (out/'asset_audit.json').exists():
        previous=load(out/'asset_audit.json')
        for package in previous['assets']:
            check_files(out/package['variant']/f"seed_{package['seed']}",package['files'])
    run=load(out/'run_manifest.json'); verify_signature(run); check_files(ROOT,run['code_sha256'])
    freeze=load(out/'freeze.json'); verify_signature(freeze); check_files(out,freeze['entries'])
    if freeze['run_manifest_sha256']!=sha(out/'run_manifest.json'): raise ValueError('Frozen run changed')
    if sha(out/'inputs/whitener.pt')!={str(project_path(k).relative_to(ROOT)):v for k,v in run['frozen_inputs'].items()}[str((COSMX_RUN/'whitener.pt').relative_to(ROOT))]: raise ValueError('Whitening changed')
    for split in ['train','validation','test']:
        path=out/'inputs'/split/'manifest.json'
        if path.exists(): check_files(path.parent,load(path)['files'])
    centers=pd.read_csv(out/'inputs/test/centers.csv',dtype={'cell_barcode':str,'reference_niche':str}); rows=[]; summary=[]; packages=[]
    for variant in VARIANTS:
        predictions=[]
        for seed in SEEDS:
            directory=out/variant/f'seed_{seed}'; selected=completed_selection(directory); result=load(directory/'test_result.json')
            if result['selection_sha256']!=sha(directory/'selection.json') or result['r_star']!=selected['r_star']: raise ValueError('Test selection binding mismatch')
            check_files(directory,result['assets'])
            assignment=pd.read_csv(directory/'test_assignments.csv',dtype={'cell_barcode':str,'reference_niche':str})
            if not assignment.cell_barcode.equals(centers.cell_barcode) or not assignment.reference_niche.equals(centers.reference_niche): raise ValueError('Cell order/reference changed')
            pred=assignment.predicted_cluster.to_numpy(); predictions.append(pred)
            for key,score in [('ari',adjusted_rand_score),('nmi',normalized_mutual_info_score)]:
                if not np.isclose(result[key],score(assignment.reference_niche,pred),atol=1e-12,rtol=0): raise ValueError('Metric replay mismatch')
            for name in ['validation_embeddings.npy','test_embeddings.npy']:
                z=np.load(directory/name,mmap_mode='r')
                if z.shape!=(10000,128) or not np.isfinite(z).all(): raise ValueError('Invalid saved embedding')
            model=load_model(directory,variant,torch.device('cpu')); del model
            rows.append({k:result[k] for k in ['variant','seed','r_star','ari','nmi','reference_niche_asw','n_clusters']})
            packages.append({'variant':variant,'seed':seed,'files':{name:sha(directory/name) for name in
                ['model_best.pt','training.json','history.json','validation_embeddings.npy','validation_centers.csv',
                 'test_embeddings.npy','test_assignments.csv','selection.json','test_result.json']}})
        results=rows[-3:]; record={'variant':variant,'display_name':LABELS[variant]}
        for key in ['ari','nmi','n_clusters','reference_niche_asw']:
            values=[r[key] for r in results if r[key] is not None]
            record[key+'_mean']=float(np.mean(values)) if values else None
            record[key+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else None
        pairs=[{'seed_a':SEEDS[i],'seed_b':SEEDS[j],'ari':float(adjusted_rand_score(predictions[i],predictions[j]))} for i in range(3) for j in range(i+1,3)]
        record['cross_seed_ari']=float(np.mean([p['ari'] for p in pairs])); record['cross_seed_pairs']=pairs; summary.append(record)
    write(out/'cosmx_ablation_summary.json',{'dataset':'cosmx_nsclc','task':'reference_niche_identification',
        'test_donor':'Lung9','n_test_centers':10000,'seeds':list(SEEDS),'post_hoc':True,
        'note':'Single test donor; seed SD is not donor uncertainty. No depth endpoint. Cross-seed ARI is the descriptive mean of dependent seed pairs.',
        'results':summary})
    pd.DataFrame(rows).to_csv(out/'result.csv',index=False)
    pd.DataFrame([{k:v for k,v in row.items() if k!='cross_seed_pairs'} for row in summary]).to_csv(out/'summary.csv',index=False)
    write(out/'asset_audit.json',{'status':'passed','packages':18,'checkpoint_reload':'strict CPU reload passed',
        'metric_replay':'ARI/NMI recomputed from aligned assignments','assets':packages})
    if complete: write(out/'status.json',{'status':'complete','packages':18,'variants':6,'seeds':3,'audit':'passed'})
    print('ASSET_AUDIT_PASSED',len(packages),flush=True)


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['all','train_validation','test','audit','preflight'],default='all')
    parser.add_argument('--output-dir',type=Path,required=True); parser.add_argument('--device',default='cuda')
    parser.add_argument('--resume',action='store_true'); parser.add_argument('--no-reuse-existing',action='store_true')
    parser.add_argument('--authorized',action='store_true'); parser.add_argument('--plan',action='store_true',help='Plan only; no data access or writes')
    parser.add_argument('--max-epochs',type=int,default=50); parser.add_argument('--patience',type=int,default=7)
    parser.add_argument('--batch-size',type=int,default=64); parser.add_argument('--min-delta',type=float,default=1e-4)
    return parser.parse_args()


def main():
    args=parse_args(); out=args.output_dir.resolve(); allowed=(ROOT/'outputs/runs/ablations/cosmx_nsclc').resolve()
    if out==allowed or not out.is_relative_to(allowed): raise ValueError(f'Use a new dated directory below {allowed}')
    if min(args.max_epochs,args.patience,args.batch_size)<=0 or args.min_delta<0: raise ValueError('Invalid training budget')
    if args.plan:
        print(json.dumps({'output_dir':str(out),'phase':args.phase,'variants':LABELS,'seeds':SEEDS,
            'reuse_existing_full_and_no_coords':not args.no_reuse_existing,'new_training_runs':18 if args.no_reuse_existing else 12,
            'primary_metrics':['ARI','NMI'],'auxiliary':['cross_seed_ARI','reference_niche_ASW'],'no_training_or_data_access':True},indent=2)); return
    runtime()
    if args.phase=='audit': audit_run(out,complete=True); return
    audit,inputs,reused=preflight(args)
    if args.phase=='preflight': print('PREFLIGHT_PASSED: frozen inputs and reusable checkpoints verified; no training or output writes'); return
    if args.phase in ['all','test'] and not args.authorized: raise PermissionError('Test evaluation requires --authorized')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    fresh=not (out/'run_manifest.json').exists()
    if fresh and args.phase=='test': raise FileNotFoundError('No frozen run to test')
    if fresh and out.exists() and any(out.iterdir()): raise FileExistsError('Nonempty output without a valid run manifest')
    if not fresh and not args.resume and args.phase!='test': raise FileExistsError('Run exists; use --resume with original run ID/date')
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'.run.lock').open('w') as lock, (out/'run.log').open('a',buffering=1) as log:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        console_out,console_err=sys.stdout,sys.stderr
        # Keep the console visible while persisting diagnostics for failed runs.
        sys.stdout=Tee(console_out,log); sys.stderr=Tee(console_err,log)
        try:
            config={'protocol':PROTOCOL,'budget':{'max_epochs':args.max_epochs,'patience':args.patience,
                'batch_size':args.batch_size,'min_delta':args.min_delta},'reuse_existing':not args.no_reuse_existing,
                'frozen_inputs':inputs,'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in source_paths()},'reused_sources':reused}
            if fresh:
                write(out/'run_manifest.json',sign(config)); (out/'inputs').mkdir(exist_ok=True)
                shutil.copyfile(COSMX_RUN/'whitener.pt',out/'inputs/whitener.pt'); shutil.copyfile(COSMX_RUN/'val_bootstrap_folds.json',out/'inputs/val_bootstrap_folds.json')
                write(out/'environment.json',{'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
                    'numpy':np.__version__,'scanpy':sc.__version__,'device':str(device)})
            else:
                old=load(out/'run_manifest.json'); verify_signature(old)
                if {k:v for k,v in old.items() if k!='signature'}!=config: raise ValueError('Code, inputs or configuration changed; do not mix runs')
            if args.phase in ['all','train_validation']: train_validation(args,out,audit,reused,device)
            if args.phase in ['all','test']: test_phase(args,out,audit,device); audit_run(out,complete=True)
        except Exception as error:
            prior=load(out/'status.json') if (out/'status.json').exists() else {}
            write(out/'status.json',{**prior,'status':'failed','error':f'{type(error).__name__}: {error}'}); raise
        finally:
            sys.stdout,sys.stderr=console_out,console_err


if __name__=='__main__': main()
