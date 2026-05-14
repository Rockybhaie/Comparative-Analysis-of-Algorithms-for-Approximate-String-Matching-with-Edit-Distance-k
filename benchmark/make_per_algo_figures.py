"""
Generate per-algorithm figures and reorganize the figures/ folder into:

    figures/
        all/         -> existing comparative figures (all 4 algos together)
        brute/       -> Khalid's brute-force-only scaling figure
        sellers/     -> Kashish's Sellers'-only scaling figure
        pigeonhole/  -> Muskan's pigeonhole-only scaling figure
        seed/        -> Abdullah's seed-and-extend-only scaling figure

Reads the saved CSVs in results/ — no re-benchmarking is needed.
"""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parent
RES     = ROOT / 'results'
FIG     = ROOT / 'figures'
FIG.mkdir(exist_ok=True)

ALGOS = ['brute', 'sellers', 'pigeonhole', 'seed']
ALGO_LABELS = {
    'brute':      'Brute Force (Khalid)',
    'sellers':    "Sellers' DP (Kashish)",
    'pigeonhole': 'Pigeonhole D&C (Muskan)',
    'seed':       'Seed-and-Extend (Abdullah)',
}
ALGO_COLOR = {'brute':'#d62728','sellers':'#1f77b4','pigeonhole':'#2ca02c','seed':'#ff7f0e'}

# ---------------------------------------------------------------------------
# 1. Move the existing comparative figures into figures/all/
# ---------------------------------------------------------------------------
ALL_DIR = FIG / 'all'
ALL_DIR.mkdir(exist_ok=True)
for png in FIG.glob('*.png'):
    target = ALL_DIR / png.name
    if target.exists(): target.unlink()
    shutil.move(str(png), target)
    print(f'moved {png.name} -> all/')

# ---------------------------------------------------------------------------
# 2. Per-algorithm subfolders
# ---------------------------------------------------------------------------
for algo in ALGOS:
    (FIG / algo).mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 3. Load the per-experiment CSVs (drop heavy `starts` column)
# ---------------------------------------------------------------------------
def load_no_starts(name):
    return pd.read_csv(RES / name, usecols=lambda c: c != 'starts')

df_n     = load_no_starts('exp2_scaling_n.csv')
df_m     = load_no_starts('exp3_scaling_m.csv')
df_k     = load_no_starts('exp4_scaling_k.csv')
df_sigma = load_no_starts('exp5_sigma.csv')

def agg(df, *group_cols):
    g = df.dropna(subset=['us']).groupby(['algo', *group_cols])
    return g['us'].agg(['mean','std','count']).reset_index().rename(
        columns={'mean':'us_mean','std':'us_std','count':'n_seeds'})

# Aggregate (mean ±SE across seeds, dna only for clean single-curve plots)
agg_n     = agg(df_n[df_n.family=='dna'], 'n')
agg_m     = agg(df_m[df_m.family=='dna'], 'm_req')
agg_k     = agg(df_k[df_k.family=='dna'], 'k_req')
agg_sigma = agg(df_sigma, 'sigma')

# ---------------------------------------------------------------------------
# 4. Render per-algorithm 2x2 grid
# ---------------------------------------------------------------------------
def panel(ax, sub, x, xlabel, title, logx=True, logy=True):
    if sub.empty:
        ax.text(0.5, 0.5, 'no data\n(algorithm skipped\non these inputs)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=9, color='gray')
        ax.set_title(title, fontsize=10); ax.set_xlabel(xlabel)
        return
    sub = sub.sort_values(x)
    se  = sub['us_std'] / np.sqrt(sub['n_seeds'].replace(0, 1))
    ax.errorbar(sub[x], sub['us_mean'], yerr=se, marker='o', capsize=3,
                color=ALGO_COLOR[sub['algo'].iloc[0]])
    if logx: ax.set_xscale('log')
    if logy: ax.set_yscale('log')
    ax.set_xlabel(xlabel); ax.set_ylabel('Runtime μs (log)')
    ax.set_title(title, fontsize=10)
    ax.grid(True, which='both', ls=':', alpha=0.4)

for algo in ALGOS:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(f'{ALGO_LABELS[algo]} — scaling across all dimensions\n'
                 '(family=dna for n/m/k panels; mean ±SE across seeds)',
                 fontsize=12, y=1.00)

    panel(axes[0, 0], agg_n[agg_n.algo == algo],
          x='n',     xlabel='Text length n (log)',
          title='vs text length n   (m=20, k=2)')
    panel(axes[0, 1], agg_m[agg_m.algo == algo],
          x='m_req', xlabel='Pattern length m',
          title='vs pattern length m   (n=20k, k=2)', logx=False)
    panel(axes[1, 0], agg_k[agg_k.algo == algo],
          x='k_req', xlabel='Error budget k',
          title='vs error budget k   (n=20k, m=32)', logx=False)
    panel(axes[1, 1], agg_sigma[agg_sigma.algo == algo],
          x='sigma', xlabel='Alphabet size σ (log₂)',
          title='vs alphabet size σ   (n=20k, m=20, k=2)')

    plt.tight_layout()
    out = FIG / algo / f'algo_{algo}.png'
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out.relative_to(ROOT)}')

print('\nDone. Folder layout:')
for sub in sorted(FIG.iterdir()):
    if sub.is_dir():
        print(f'  figures/{sub.name}/')
        for f in sorted(sub.glob('*.png')):
            print(f'      {f.name}')
