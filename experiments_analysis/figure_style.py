"""
experiments_analysis/figure_style.py
Unified publication-ready plotting style for FreqNiche-ST.
Follows Nature/Cell guidelines: clear typography, colorblind-friendly palettes,
vector PDF export, and automatic metadata sidecars.
"""

import os
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Color palettes (Nature/Cell style colorblind-friendly)
PALETTE_METHODS = {
    'FreqNiche': '#1f77b4',          # Deep Blue (Proposed)
    'FreqNiche (Ours)': '#1f77b4',
    'FreqNiche v1.0': '#1f77b4',
    'FreqNiche v1.0 Final': '#1f77b4',
    'FreqNiche-ST': '#1f77b4',       # Backward compatibility
    'FreqNiche-ST (Ours)': '#1f77b4',
    'Raw PCA': '#7f7f7f',            # Slate Grey (Lower bound)
    'DualFreq': '#2ca02c',           # Forest Green (Linear filter)
    'DualFreq (px_resdiff)': '#2ca02c',
    'BANKSY': '#ff7f0e',             # Amber Orange
    'BANKSY-Controlled': '#ff7f0e',
    'CellCharter': '#d62728',        # Crimson Red
    'CellCharter-PCA': '#d62728',
    'CellCharter-Canonical': '#d62728',
    'STAGATE': '#9467bd',            # Purple
    'STAGATE-Controlled': '#9467bd',
    'CellNiche': '#8c564b',          # Brown
    'CellNiche-Controlled': '#8c564b',
    'Coordinate-only': '#bcbd22',    # Olive
}

PALETTE_CELL_TYPES = [
    '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
    '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC',
    '#86BCB6', '#E7CA60', '#D37295', '#FABFD2', '#B6992D'
]

import matplotlib.font_manager as fm

def set_publication_style():
    """Apply unified publication-quality styling to matplotlib/seaborn."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Register Times New Roman and SimSun fonts
    candidate_font_paths = [
        '/mnt/c/Windows/Fonts/simsun.ttc',
        '/mnt/c/Windows/Fonts/times.ttf',
        'C:/Windows/Fonts/simsun.ttc',
        'C:/Windows/Fonts/times.ttf'
    ]
    for p in candidate_font_paths:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
            except Exception:
                pass

    mpl.rcParams['font.family'] = ['Times New Roman', 'SimSun']
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['font.size'] = 8.5
    mpl.rcParams['axes.labelsize'] = 9.0
    mpl.rcParams['axes.titlesize'] = 10.0
    mpl.rcParams['xtick.labelsize'] = 8.0
    mpl.rcParams['ytick.labelsize'] = 8.0
    mpl.rcParams['legend.fontsize'] = 8.0
    mpl.rcParams['legend.title_fontsize'] = 8.5
    mpl.rcParams['figure.titlesize'] = 11.0
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['grid.linewidth'] = 0.4
    mpl.rcParams['grid.alpha'] = 0.5
    mpl.rcParams['lines.linewidth'] = 1.2
    mpl.rcParams['lines.markersize'] = 4.5
    mpl.rcParams['pdf.fonttype'] = 42  # TrueType for editable text
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'

def save_publication_figure(fig, fig_basename, output_dir, metadata=None):
    """
    Save figure as vector PDF, vector SVG, and high-res 600 DPI PNG,
    and output a companion JSON file tracking data provenance and parameters.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{fig_basename}.pdf")
    svg_path = os.path.join(output_dir, f"{fig_basename}.svg")
    png_path = os.path.join(output_dir, f"{fig_basename}.png")
    meta_path = os.path.join(output_dir, f"{fig_basename}_provenance.json")

    # Save PDF (vector, rasterized scatter layers at 600 DPI)
    fig.savefig(pdf_path, format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.04)
    # Save SVG (vector, rasterized scatter layers at 600 DPI)
    fig.savefig(svg_path, format='svg', dpi=600, bbox_inches='tight', pad_inches=0.04)
    # Save PNG (600 DPI high-resolution bitmap)
    fig.savefig(png_path, format='png', dpi=600, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)

    if metadata is not None:
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[OK] Figure saved: {pdf_path}")
    print(f"[OK] Vector SVG saved: {svg_path}")
    print(f"[OK] 600 DPI PNG saved: {png_path}")
    if metadata:
        print(f"[OK] Provenance saved: {meta_path}")
