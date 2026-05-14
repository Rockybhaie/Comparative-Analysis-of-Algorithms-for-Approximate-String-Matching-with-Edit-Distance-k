"""
Generate analytical / cross-cutting figures for the report.

Output: FINAL/figures/analysis/  (and a copy under benchmark/figures/analysis/)

All figures are derived from saved CSVs in benchmark/results/ — no
re-benchmarking is needed. Run time: a few seconds.
"""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT  = Path(__file__).resolve().parent
RES   = ROOT / 'results'
OUT   = ROOT / 'figures' / 'analysis'
FINAL = ROOT.parent / 'FINAL' / 'figures' / 'analysis'
for d in (OUT, FINAL):
    d.mkdir(parents=True, exist_ok=True)

ALGOS = ['brute','sellers','pigeonhole','seed']
LBL   = {'brute':'Brute Force (Khalid)','sellers':"Sellers' DP (Kashish)",
         'pigeonhole':'Pigeonhole D&C (Muskan)','seed':'Seed-and-Extend (Abdullah)'}
COL   = {'brute':'#d62728','sellers':'#1f77b4','pigeonhole':'#2ca02c','seed':'#ff7f0e'}

def save(fig, name):
    for d in (OUT, FINAL):
        fig.savefig(d / name, dpi=140, bbox_inches='tight')
    print(f'  -> figures/analysis/{name}')
    plt.close(fig)

def load_no_starts(p):
    return pd.read_csv(RES / p, usecols=lambda c: c != 'starts')

# ---------------------------------------------------------------------------
# Figure 1: pure analytical big-O comparison
# ---------------------------------------------------------------------------
def fig_theoretical():
    n  = np.logspace(2, 7, 200)        # n from 100 to 10^7
    m, k = 20, 2
    sigma = 4
    # Drop unit constants — we just want the shapes
    t_brute   = n * (3 ** k)           # O(n * 3^k)        worst-case branching
    t_sellers = n * m                  # O(n * m)          DP
    t_pigeon  = n * k                  # O(n * k)          average-case filter
    q = max(2, m // (k + 2))
    t_seed    = n + n * m / (sigma ** q)   # O(n + n*m/σ^q) seeds + extensions
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.loglog(n, t_brute,   label="O(n·3$^k$)  — Brute Force",      color=COL['brute'])
    ax.loglog(n, t_sellers, label="O(n·m)      — Sellers' DP",       color=COL['sellers'])
    ax.loglog(n, t_pigeon,  label="O(n·k)      — Pigeonhole",        color=COL['pigeonhole'])
    ax.loglog(n, t_seed,    label="O(n + n·m/σ$^q$) — Seed-and-Extend", color=COL['seed'])
    ax.set_xlabel('Text length n (log)'); ax.set_ylabel('Operations (log, arbitrary units)')
    ax.set_title(f'Theoretical asymptotic complexity  (m={m}, k={k}, σ={sigma}, q≈{q})')
    ax.grid(True, which='both', ls=':', alpha=0.5); ax.legend(fontsize=9)
    save(fig, 'fig_theoretical_complexity.png')

# ---------------------------------------------------------------------------
# Figure 2: empirical scaling vs theoretical reference  (2x2 grid, one per algo)
# ---------------------------------------------------------------------------
def fig_empirical_vs_theoretical():
    df = load_no_starts('exp2_scaling_n.csv')
    df = df[(df.family == 'dna') & (~df.skipped.astype(bool))]
    agg = (df.dropna(subset=['us']).groupby(['algo','n'])['us']
              .agg(['mean','std','count']).reset_index())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("Empirical scaling vs theoretical complexity  (family=dna, m=20, k=2)",
                 fontsize=12, y=1.00)
    for ax, algo in zip(axes.flat, ALGOS):
        sub = agg[agg.algo == algo].sort_values('n')
        if sub.empty or len(sub) < 2:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            ax.set_title(LBL[algo], fontsize=10); continue
        se = sub['std'] / np.sqrt(sub['count'].replace(0,1))
        ax.errorbar(sub.n, sub['mean'], yerr=se, marker='o', ms=5,
                    color=COL[algo], capsize=3, label='measured (mean ±SE)')
        # Empirical power-law fit T(n) = a * n^b
        positive = sub[sub['mean'] > 0]
        if len(positive) >= 2:
            b, log_a = np.polyfit(np.log(positive.n), np.log(positive['mean']), 1)
            n_fit = np.linspace(sub.n.min(), sub.n.max(), 50)
            ax.plot(n_fit, np.exp(log_a) * n_fit**b, '--', color=COL[algo],
                    alpha=0.6, label=f'fit  T∝n$^{{{b:.2f}}}$')
        # Theoretical reference: linear in n (slope 1) anchored at midpoint
        mid = sub.iloc[len(sub)//2]
        if mid['mean'] > 0:
            n_ref = np.linspace(sub.n.min(), sub.n.max(), 50)
            ax.plot(n_ref, mid['mean'] * (n_ref / mid['n']),
                    ':', color='black', alpha=0.5, label='theory  T∝n')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('n (log)'); ax.set_ylabel('μs (log)')
        ax.set_title(LBL[algo], fontsize=10)
        ax.grid(True, which='both', ls=':', alpha=0.4)
        ax.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    save(fig, 'fig_empirical_vs_theoretical.png')

# ---------------------------------------------------------------------------
# Figure 3: speed–recall Pareto (the headline trade-off)
# ---------------------------------------------------------------------------
def fig_pareto():
    runs = load_no_starts('exp4_scaling_k.csv')
    acc  = pd.read_csv(RES / 'exp8_accuracy_vs_k.csv')
    agg_us = (runs.dropna(subset=['us']).groupby(['algo','k_req'])['us']
                  .mean().reset_index().rename(columns={'k_req':'k','us':'mean_us'}))
    merged = agg_us.merge(acc, on=['algo','k'], how='inner')

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for algo in ALGOS:
        sub = merged[merged.algo == algo].sort_values('k')
        if sub.empty: continue
        ax.plot(sub.mean_us, sub.recall, marker='o', color=COL[algo],
                label=LBL[algo], linewidth=2)
        for _, r in sub.iterrows():
            ax.annotate(f"k={int(r.k)}", (r.mean_us, r.recall),
                        fontsize=7, alpha=0.6,
                        xytext=(4, 4), textcoords='offset points')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlabel('Mean runtime per case (μs, log)')
    ax.set_ylabel('Recall (vs Sellers ground truth)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Speed vs recall — the central trade-off  (n=20k, m=32, family=dna)')
    ax.grid(True, ls=':', alpha=0.5); ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    save(fig, 'fig_pareto_speed_recall.png')

# ---------------------------------------------------------------------------
# Figure 4: runtime distribution per algorithm (box plot)
# ---------------------------------------------------------------------------
def fig_runtime_distribution():
    df = load_no_starts('all_runs.csv')
    df = df.dropna(subset=['us'])
    df = df[df.us > 0]                  # log scale needs positive values
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [df[df.algo == a]['us'].values for a in ALGOS]
    bp = ax.boxplot(data, labels=[LBL[a].split(' (')[0] for a in ALGOS],
                    showfliers=True, patch_artist=True, widths=0.6)
    for patch, a in zip(bp['boxes'], ALGOS):
        patch.set_facecolor(COL[a]); patch.set_alpha(0.5)
    ax.set_yscale('log'); ax.set_ylabel('Runtime μs (log)')
    ax.set_title('Runtime distribution across all cases  (3,440 benchmarked runs)')
    ax.grid(True, axis='y', ls=':', alpha=0.5)
    plt.tight_layout()
    save(fig, 'fig_runtime_distribution.png')

# ---------------------------------------------------------------------------
# Figure 5: speedup factor over Sellers' as k grows
# ---------------------------------------------------------------------------
def fig_speedup():
    df = load_no_starts('exp4_scaling_k.csv')
    df = df.dropna(subset=['us'])
    agg = (df.groupby(['algo','k_req'])['us'].mean().reset_index()
                                                 .rename(columns={'k_req':'k'}))
    base = agg[agg.algo == 'sellers'].set_index('k')['us']
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in ALGOS:
        if algo == 'sellers': continue
        sub = agg[agg.algo == algo].sort_values('k').set_index('k')
        speedup = base / sub['us']           # >1 means faster than Sellers
        speedup = speedup.replace([np.inf, -np.inf], np.nan).dropna()
        ax.plot(speedup.index, speedup.values, marker='o', color=COL[algo],
                label=LBL[algo], linewidth=2)
    ax.axhline(1.0, color='black', ls=':', alpha=0.5, label="Sellers' baseline")
    ax.set_yscale('log')
    ax.set_xlabel('Error budget k'); ax.set_ylabel("Speedup factor vs Sellers' (log)")
    ax.set_title("Speedup over Sellers' DP as k grows  (n=20k, m=32, family=dna)")
    ax.grid(True, which='both', ls=':', alpha=0.4); ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, 'fig_speedup_vs_sellers.png')

# ---------------------------------------------------------------------------
# Figure 6: match-count consistency — do algorithms agree on # of matches?
# ---------------------------------------------------------------------------
def fig_match_consistency():
    df = load_no_starts('exp4_scaling_k.csv')
    df = df.dropna(subset=['matches'])
    pivot = df.groupby(['k_req','algo'])['matches'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in ALGOS:
        sub = pivot[pivot.algo == algo].sort_values('k_req')
        ax.plot(sub.k_req, sub.matches, marker='o', color=COL[algo],
                label=LBL[algo], linewidth=2)
    ax.set_yscale('symlog')
    ax.set_xlabel('Error budget k'); ax.set_ylabel('Average matches reported per case (symlog)')
    ax.set_title('Match-count behaviour as k grows  (n=20k, m=32, family=dna)')
    ax.grid(True, which='both', ls=':', alpha=0.4); ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, 'fig_match_count_consistency.png')

if __name__ == '__main__':
    print('Generating analysis figures into FINAL/figures/analysis/ :')
    fig_theoretical()
    fig_empirical_vs_theoretical()
    fig_pareto()
    fig_runtime_distribution()
    fig_speedup()
    fig_match_consistency()
    print('\nDone.')
