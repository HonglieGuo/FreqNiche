"""Render composition and cortical-depth probes together; optionally publish.

Uses the verified composition cache and frozen depth-probe alpha. Final depth
predictions are recovered once, checked against the original results and cached.
Original result JSON is read-only. --publish updates the approved manuscript figure.
"""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
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
from sklearn.linear_model import Ridge, LinearRegression
from experiments_analysis.plot_figures.plot_neighborhood_composition import sha, write, SEEDS, SUMMARY
from experiments_analysis.cortical_depth_prediction.run_cortical_depth_prediction import evaluate_depth_predictions
from src.freqniche_st.runtime_paths import project_path, validated_source_bindings

COMPOSITION = ROOT/'Paper/image-gen/neighborhood_composition_figure4_20261019'
DEPTH = ROOT/'experiments_analysis/cortical_depth_prediction/results/cortical_depth_results.json'
SEA_AD_RUN = ROOT/'outputs/runs/freqniche/sea_ad/2026-09-15/20260915_sea_ad_final'
DATA = ROOT/'datasets/h5ad/sea_ad/sea_ad_mtg_merfish.h5ad'
NAME = 'fig_combined_composition_depth_prediction_preview'
GRAY, BLUE, RED = '#929292', '#2878ae', '#d96363'


def depth_arrays(out, summary):
    paths = [DEPTH, DATA, SEA_AD_RUN/'val_centers.json', SEA_AD_RUN/'test_centers.json',
             SEA_AD_RUN/'embeddings/val_embeddings_seed_42.npy', SEA_AD_RUN/'embeddings/test_embeddings_seed_42.npy']
    sources = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    binding = out/'depth_plot_binding.json'
    if binding.exists():
        meta = json.loads(binding.read_text())
        assert validated_source_bindings(meta['sources']) == sources, 'Depth sources changed'
        for name, digest in meta['files'].items():
            assert sha(out/name) == digest, name
        return np.load(out/'depth_plot_arrays.npz'), meta
    val_ids = json.loads((SEA_AD_RUN/'val_centers.json').read_text())['cell_barcodes']
    test_ids = json.loads((SEA_AD_RUN/'test_centers.json').read_text())['cell_barcodes']
    a = ad.read_h5ad(DATA, backed='r')
    try:
        obs = a.obs.loc[test_ids]
        y_val = a.obs.loc[val_ids, 'Normalized depth from pia'].to_numpy(dtype=np.float32)
        y_test = obs['Normalized depth from pia'].to_numpy(dtype=np.float32)
        donors = obs.donor_id.to_numpy().astype(str)
        coords = np.asarray(a.obsm['spatial'][a.obs.index.get_indexer(test_ids)])
    finally:
        a.file.close()
    alpha = summary['per_seed_results']['42']['freqniche']['best_alpha']
    reg = LinearRegression() if alpha == 0 else Ridge(alpha=alpha, random_state=42)
    pipe = Pipeline([('scaler', StandardScaler()), ('reg', reg)])
    pipe.fit(np.load(paths[-2]), y_val)
    result = evaluate_depth_predictions(y_test, pipe.predict(np.load(paths[-1])), donors)
    saved = summary['per_seed_results']['42']['freqniche']
    differences = []
    for key in ['macro_spearman_rho', 'macro_mae', 'macro_r2']:
        differences.append(abs(result[key]-saved[key]))
        assert np.isclose(result[key], saved[key], rtol=1e-5, atol=1e-6), key
    for donor in summary['test_unseen_donors']:
        for key in ['spearman_rho', 'mae', 'r2']:
            differences.append(abs(result['by_donor'][donor][key]-saved['by_donor'][donor][key]))
            assert np.isclose(result['by_donor'][donor][key], saved['by_donor'][donor][key], rtol=1e-5, atol=1e-6), (donor, key)
    np.savez_compressed(out/'depth_plot_arrays.npz', observed=y_test,
                        predicted=result['y_pred_clipped'], xy=coords, donor=donors)
    pd.DataFrame({'cell_barcode': test_ids, 'donor': donors, 'x': coords[:, 0], 'y': coords[:, 1],
                  'observed_depth': y_test, 'predicted_depth_seed42': result['y_pred_clipped']}).to_csv(out/'depth_predictions_seed42.csv', index=False)
    joblib.dump(pipe, out/'depth_freqniche_seed42_probe.joblib')
    meta = {'sources': sources, 'frozen_alpha': alpha, 'verification': 'passed',
            'maximum_absolute_metric_difference': max(differences),
            'note': 'Final probe refit at original alpha solely to recover plotting predictions. No model training, CV or parameter selection.'}
    meta['files'] = {name: sha(out/name) for name in ['depth_plot_arrays.npz', 'depth_predictions_seed42.csv', 'depth_freqniche_seed42_probe.joblib']}
    write(binding, meta)
    print('DEPTH_RECOVERY_VERIFIED', max(differences), flush=True)
    return np.load(out/'depth_plot_arrays.npz'), meta


def spatial_style(ax, title):
    ax.set_box_aspect(.82)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_anchor('N')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, pad=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'Paper/image-gen/combined_prediction_probes_20261019')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    original_hashes = {str(p.relative_to(ROOT)): sha(p) for p in [SUMMARY, DEPTH]}
    comp_summary, depth_summary = json.loads(SUMMARY.read_text()), json.loads(DEPTH.read_text())
    binding = json.loads((COMPOSITION/'plot_data_binding.json').read_text())
    for name in ['plot_arrays.npz', 'test_centers.csv']:
        assert sha(COMPOSITION/name) == binding['files'][name]
    assert binding['sources'][str(SUMMARY.relative_to(ROOT))] == original_hashes[str(SUMMARY.relative_to(ROOT))]
    comp = np.load(COMPOSITION/'plot_arrays.npz')
    frame = pd.read_csv(COMPOSITION/'test_centers.csv')
    depth, depth_binding = depth_arrays(out, depth_summary)
    for base in [Path('/mnt/c/Windows/Fonts'), Path('C:/Windows/Fonts')]:
        for name in ['times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf']:
            if (base/name).exists():
                font_manager.fontManager.addfont(str(base/name))
    font_manager.findfont(font_manager.FontProperties(family='Times New Roman'), fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 12,
                         'axes.titlesize': 14, 'axes.labelsize': 12,
                         'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5,
                         'axes.linewidth': .6, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.unicode_minus': False,
                         'mathtext.fontset': 'stix'})
    # Reserve a separate right-hand gutter for both method legends.
    fig = plt.figure(figsize=(13.5, 9.8))
    plot_width_fraction = 12/fig.get_figwidth()
    def plot_box(box):
        x, y, width, height = box
        return [x*plot_width_fraction, y, width*plot_width_fraction, height]
    fig.text(.5, .965, 'Human Skin | Neighborhood composition prediction', fontsize=15, weight='bold', ha='center')
    fig.text(.5, .475, 'SEA-AD | Continuous cortical depth prediction', fontsize=15, weight='bold', ha='center')
    axes = [fig.add_axes(plot_box(box)) for box in [
        [.115, .59, .25, .32], [.385, .59, .25, .32],
        [.70, .59, .17, .32], [.93, .59, .125, .32],
        [.075, .155, .215, .28], [.305, .155, .215, .28],
        [.60, .155, .165, .28], [.845, .155, .14, .28]]]
    a, b, c, d, e, f, g, h = axes
    cats = comp_summary['cell_types']
    colors = {ct: plt.get_cmap('tab20')(i) for i, ct in enumerate(cats)}
    mask = frame.donor_id.eq('D028').to_numpy()
    xy = frame.loc[mask, ['center_x', 'center_y']].to_numpy()
    assert mask.sum() == 2000
    for ax, values, title in [(a, comp['gt_test'][mask].argmax(1), '(A) Observed'),
                               (b, comp['freqniche_42'][mask].argmax(1), '(B) Predicted')]:
        for i, ct in enumerate(cats):
            ax.scatter(*xy[values == i].T, color=colors[ct], s=5, alpha=.9, linewidths=0, rasterized=True)
        spatial_style(ax, title)
        ax.set_box_aspect(.32*fig.get_figheight()/(ax.get_position().width*fig.get_figwidth()))
        ax.set_anchor('S')
    fig.canvas.draw()
    b.set_xlim(a.get_xlim()); b.set_ylim(a.get_ylim())
    legend_ax = fig.add_axes(plot_box([.005, .595, .10, .34])); legend_ax.axis('off')
    cell_legend = legend_ax.legend(handles=[Line2D([], [], marker='o', linestyle='none', markersize=4,
                                   markerfacecolor=colors[ct], markeredgecolor='none', label=ct) for ct in cats],
                     loc='upper left', ncol=1, frameon=False, fontsize=11.5,
                     handlelength=.55, handletextpad=.4, borderaxespad=0, labelspacing=.2)
    raw = np.mean([comp[f'raw_pca_{seed}_type_mae'] for seed in SEEDS], axis=0)
    freq = np.mean([comp[f'freqniche_{seed}_type_mae'] for seed in SEEDS], axis=0)
    y = np.arange(len(cats))
    c.barh(y-.18, raw, .36, color=GRAY)
    c.barh(y+.18, freq, .36, color=BLUE)
    c.set_yticks(y, cats); c.set_ylim(17.8, -.8)
    c.set_xlabel('MAE'); c.set_xlim(0, max(raw.max(), freq.max())*1.12)
    c.set_title('(C) Cell-type MAE', pad=7)
    donors = comp_summary['unseen_test_donors']; x = np.arange(len(donors))
    for method, offset, color in [('raw_pca', -.18, GRAY), ('freqniche', .18, BLUE)]:
        values = np.array([[comp_summary['per_seed_results'][str(seed)][method]['by_donor'][donor]['js_distance'] for donor in donors] for seed in SEEDS])
        d.bar(x+offset, values.mean(0), .36, yerr=values.std(0, ddof=1), color=color,
              capsize=2, error_kw={'elinewidth': .7})
    d.set_xticks(x, donors, rotation=45, ha='right', rotation_mode='anchor')
    d.set_ylim(0, .6); d.set_yticks([0, .2, .4, .6]); d.set_ylabel('JS distance')
    d.set_title('(D) Donor error', pad=7)
    method_legend = fig.legend(handles=[Patch(facecolor=GRAY, label='Raw PCA'), Patch(facecolor=BLUE, label='FreqNiche')],
               loc='upper center', bbox_to_anchor=(.79, .905), ncol=1, frameon=False, fontsize=12,
               handlelength=1, handletextpad=.45, borderaxespad=0, labelspacing=.55)
    depth_mask = depth['donor'] == 'H21.33.001'
    true, pred = depth['observed'][depth_mask], depth['predicted'][depth_mask]
    assert depth_mask.sum() == 2000
    for ax, values, title in [(e, true, '(E) Observed depth'), (f, pred, '(F) Predicted depth')]:
        sc = ax.scatter(*depth['xy'][depth_mask].T, c=values, cmap='viridis', vmin=0, vmax=1,
                        s=5, alpha=.9, linewidths=0, rasterized=True)
        spatial_style(ax, title)
        # Match G/H frame height without stretching physical coordinates.
        ax.set_box_aspect(.28*fig.get_figheight()/(ax.get_position().width*fig.get_figwidth()))
        ax.set_anchor('S')
    fig.canvas.draw()
    f.set_xlim(e.get_xlim()); f.set_ylim(e.get_ylim())
    colorbar_ax = fig.add_axes(plot_box([.05, .195, .010, .20]))
    colorbar = fig.colorbar(sc, cax=colorbar_ax, orientation='vertical', ticks=[0, .5, 1])
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor('face')
    colorbar.ax.yaxis.set_ticks_position('left')
    colorbar.ax.set_yticklabels(['0 (pia)', '0.5', '1 (WM)'], fontsize=10.5)
    colorbar.ax.tick_params(length=2, pad=3)
    colorbar_ax.set_title('Depth', fontsize=11.5, pad=7)
    colorbar.outline.set_linewidth(.5)
    g.scatter(true, pred, color=BLUE, s=5, alpha=.35, linewidths=0, rasterized=True)
    g.plot([0, 1], [0, 1], '--', color=RED, lw=1)
    g.set_xlim(-.03, 1.03); g.set_ylim(-.03, 1.03)
    g.set_xticks([0, .5, 1]); g.set_yticks([0, .5, 1])
    g.set_xlabel('Observed depth'); g.set_ylabel('Predicted depth')
    g.set_title('(G) Depth agreement', pad=7)
    local = depth_summary['per_seed_results']['42']['freqniche']['by_donor']['H21.33.001']
    g.text(.96, .04, f"Spearman ρ = {local['spearman_rho']:.4f}\nR² = {local['r2']:.4f}",
           ha='right', va='bottom', transform=g.transAxes, fontsize=11.5,
           bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': 'none', 'pad': 2})
    donors = depth_summary['test_unseen_donors']; x = np.arange(len(donors)); width = .25
    for method, offset, color in [('coord', -width, RED), ('raw_pca', 0, GRAY), ('freqniche', width, BLUE)]:
        values = np.array([[depth_summary['per_seed_results'][str(seed)][method]['by_donor'][donor]['spearman_rho'] for donor in donors] for seed in SEEDS])
        h.bar(x+offset, values.mean(0), width, yerr=values.std(0, ddof=1), color=color,
              capsize=1.5, error_kw={'elinewidth': .7})
    # Preserve negative coordinate-only correlations rather than clipping at -0.35.
    all_rho = [depth_summary['per_seed_results'][str(seed)][method]['by_donor'][donor]['spearman_rho']
               for seed in SEEDS for method in ['coord', 'raw_pca', 'freqniche'] for donor in donors]
    h.set_ylim(min(-.05, min(all_rho)-.08), 1.05)
    h.axhline(0, color='#999999', lw=.6)
    h.set_xticks(x, donors, rotation=55, ha='right', rotation_mode='anchor')
    h.tick_params(axis='x', labelsize=10.5)
    h.set_ylabel('Spearman correlation'); h.set_title('(H) Donor correlation', pad=7)
    for ax in [c, d, g, h]:
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='x' if ax is c else 'y', color='#e4e4e4', lw=.5)
    fig.legend(handles=[Patch(facecolor=RED, label='Coordinate-only'), Patch(facecolor=GRAY, label='Raw PCA'),
                        Patch(facecolor=BLUE, label='FreqNiche')], loc='center left', bbox_to_anchor=(.895, .295),
               ncol=1, frameon=False, fontsize=12, handlelength=1, handletextpad=.45,
               borderaxespad=0, labelspacing=.8)
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    assert not cell_legend.get_window_extent(renderer).overlaps(method_legend.get_window_extent(renderer))
    assert cell_legend.get_window_extent(renderer).x1 < a.get_window_extent(renderer).x0
    assert b.get_window_extent(renderer).x1 < min(t.get_window_extent(renderer).x0 for t in c.get_yticklabels())
    assert all(abs(ax.get_position().y0-c.get_position().y0) < 1e-8 for ax in [a, b, d])
    assert all(abs(ax.get_position().y0-g.get_position().y0) < 1e-8 for ax in [e, f, h])
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1, (text.get_text(), box)
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(out/(NAME+'.'+ext), dpi=600, facecolor='white')
    fig.savefig(out/'preview.png', dpi=160, facecolor='white')
    plt.close(fig)
    write(out/(NAME+'_provenance.json'), {'status': 'preview_only', 'source_results_sha256': original_hashes,
          'composition_cache': str(COMPOSITION.relative_to(ROOT)), 'composition_binding': binding,
          'depth_binding': depth_binding, 'layout': 'two task rows, four panels per row, independent widths',
          'A_B': 'D028 all 2000 fixed centers, seed42, dominant neighbor cell type (not full composition or single section)',
          'E_F_G': 'H21.33.001 all 2000 fixed centers, seed42; same scope as previous depth figure',
          'C_D_H': 'Three-seed means; D/H error bars are sample SD; saved original results',
          'rendering': {'font': 'Times New Roman', 'scatter_raster_dpi': 600, 'text_axes_bars_legends_vector': True,
                        'A_B_legend_rows': 18, 'A_B_legend_columns': 1,
                        'A_B_legend_position': 'single column at far left, before panel A; A/B shifted right at unchanged size',
                        'A_B_width_increase_vs_three_row_legend_percent': 100*(.25/.215-1),
                        'A_B_C_D_frames': 'common top and bottom; spatial coordinates retain equal aspect',
                        'E_F_colorbar': 'shared vertical bar on left side',
                        'task_headings': 'centered over figure width',
                        'C_D_method_legend': 'upper blank area between C and D',
                        'C_position': 'shifted right; C-D gap reduced from 0.08 to 0.06 of original plot width',
                        'G_H_method_legend': 'vertical shared legend in right-side gutter',
                        'E_F_G_H_frames': 'common top and bottom; spatial coordinates retain equal aspect',
                        'A_B_width_increase_vs_initial_percent': 100*(.25/.18-1),
                        'E_F_width_increase_vs_initial_percent': 100*(.215/.18-1)},
          'script_sha256': sha(Path(__file__))})
    for relative, digest in original_hashes.items():
        assert sha(project_path(relative)) == digest
    if args.publish:
        target = ROOT/'Paper/FreqNiche-zh/figures/fig_neighborhood_composition_cortical_depth_prediction'
        for ext in ['.pdf', '.png', '.svg']:
            shutil.copyfile(out/(NAME+ext), Path(str(target)+ext))
        meta = json.loads((out/(NAME+'_provenance.json')).read_text())
        meta.update(status='published_in_manuscript', manuscript_figure_label='fig:prediction_probes',
                    manuscript_figure_number=4, published_from=str((out/(NAME+'.pdf')).relative_to(ROOT)))
        write(Path(str(target)+'_provenance.json'), meta)
    print('COMBINED_PREVIEW_COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
