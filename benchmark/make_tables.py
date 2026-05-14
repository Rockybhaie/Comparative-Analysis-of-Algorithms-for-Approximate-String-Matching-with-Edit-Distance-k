"""
Generate report-ready tables (CSV + Markdown) from the saved benchmark CSVs.

Output: FINAL/tables/  (and benchmark/tables/).
Each table is written twice: <name>.csv (for spreadsheets / data) and
<name>.md (a copy-paste-ready Markdown table for the report).
"""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

ROOT  = Path(__file__).resolve().parent
RES   = ROOT / 'results'
OUT   = ROOT / 'tables'
FINAL = ROOT.parent / 'FINAL' / 'tables'
for d in (OUT, FINAL):
    d.mkdir(parents=True, exist_ok=True)

ALGOS = ['brute','sellers','pigeonhole','seed']
LBL   = {'brute':'Brute Force','sellers':"Sellers' DP",
         'pigeonhole':'Pigeonhole D&C','seed':'Seed-and-Extend'}

def write(df, name, index=False):
    """Save df as both CSV and Markdown to OUT and FINAL."""
    for d in (OUT, FINAL):
        df.to_csv(d / f'{name}.csv', index=index)
        with open(d / f'{name}.md', 'w', encoding='utf-8') as f:
            f.write(df.to_markdown(index=index))
            f.write('\n')
    print(f'  -> tables/{name}.csv  +  {name}.md')

def load_no_starts(p):
    return pd.read_csv(RES / p, usecols=lambda c: c != 'starts')

# =============================================================================
# Table 1 — Theoretical complexity (textbook reference)
# =============================================================================
def t1_theoretical():
    df = pd.DataFrame([
        {'Algorithm':'Brute Force (Khalid)',
         'Best-case time':'O(n)',
         'Average-case time':'O(n·m)',
         'Worst-case time':'O(n·3^k) ≈ O(n·3^m)',
         'Space':'O(m) recursion depth',
         'Optimality':'Exact'},
        {'Algorithm':"Sellers' DP (Kashish)",
         'Best-case time':'O(n·m)',
         'Average-case time':'O(n·m)',
         'Worst-case time':'O(n·m)',
         'Space':'O(m) (rolling rows)',
         'Optimality':'Exact'},
        {'Algorithm':'Pigeonhole D&C (Muskan)',
         'Best-case time':'O(n)',
         'Average-case time':'O(n·k) (small k, large σ)',
         'Worst-case time':'O(n·m·k)',
         'Space':'O(m)',
         'Optimality':'Exact'},
        {'Algorithm':'Seed-and-Extend (Abdullah)',
         'Best-case time':'O(n + m)',
         'Average-case time':'O(n + (n·m)/σ^q)',
         'Worst-case time':'O(n·m)',
         'Space':'O(n) q-gram index',
         'Optimality':'Approximate (Hamming only)'},
    ])
    write(df, 'table1_theoretical_complexity')

# =============================================================================
# Table 2 — Empirical complexity (fitted exponent on n)
# =============================================================================
def t2_empirical_exponents():
    df = load_no_starts('exp2_scaling_n.csv')
    df = df[(df.family=='dna') & (~df.skipped.astype(bool))].dropna(subset=['us'])
    agg = df.groupby(['algo','n'])['us'].mean().reset_index()
    rows = []
    for algo in ALGOS:
        sub = agg[(agg.algo==algo) & (agg.us>0)].sort_values('n')
        if len(sub) >= 2:
            slope, _ = np.polyfit(np.log(sub.n), np.log(sub.us), 1)
            rows.append({'Algorithm': LBL[algo],
                         'Empirical exponent on n': round(float(slope), 3),
                         'Theoretical exponent on n': 1.0,
                         'Δ (empirical − theory)': round(float(slope) - 1.0, 3),
                         'Data points': int(len(sub))})
        else:
            rows.append({'Algorithm': LBL[algo],
                         'Empirical exponent on n': None,
                         'Theoretical exponent on n': 1.0,
                         'Δ (empirical − theory)': None,
                         'Data points': int(len(sub))})
    write(pd.DataFrame(rows), 'table2_empirical_exponents')

# =============================================================================
# Table 3 — Per-algorithm runtime summary across the entire study
# =============================================================================
def t3_runtime_summary():
    df = load_no_starts('all_runs.csv').dropna(subset=['us'])
    rows = []
    for algo in ALGOS:
        sub = df[df.algo==algo]
        rows.append({
            'Algorithm':       LBL[algo],
            'Cases run':       int(sub['us'].count()),
            'Cases skipped':   int(df[(df.algo==algo) & (df.skipped.fillna(False).astype(bool))].shape[0]),
            'Min (μs)':        int(sub['us'].min()),
            'Median (μs)':     int(sub['us'].median()),
            'Mean (μs)':       int(sub['us'].mean()),
            'P95 (μs)':        int(sub['us'].quantile(0.95)),
            'Max (μs)':        int(sub['us'].max()),
            'Std (μs)':        int(sub['us'].std()),
        })
    write(pd.DataFrame(rows), 'table3_runtime_summary')

# =============================================================================
# Table 4 — Mean runtime by (algorithm × family)
# =============================================================================
def t4_runtime_by_family():
    df = load_no_starts('exp1_families.csv').dropna(subset=['us'])
    pivot = (df.groupby(['family','algo'])['us'].mean()
                .unstack('algo').round(0).astype('Int64'))
    pivot = pivot[ALGOS].rename(columns=LBL).reset_index()
    pivot.columns = ['Family'] + list(pivot.columns[1:])
    write(pivot, 'table4_runtime_by_family_us')

# =============================================================================
# Table 5 — Accuracy (precision / recall) by k
# =============================================================================
def t5_accuracy_vs_k():
    df = pd.read_csv(RES/'exp8_accuracy_vs_k.csv')
    df['precision'] = df['precision'].round(3)
    df['recall']    = df['recall'].round(3)
    pr = df.pivot(index='k', columns='algo', values='precision')[ALGOS] \
            .rename(columns={a:f'{LBL[a]} P' for a in ALGOS})
    rc = df.pivot(index='k', columns='algo', values='recall')[ALGOS] \
            .rename(columns={a:f'{LBL[a]} R' for a in ALGOS})
    out = pd.concat([pr, rc], axis=1).reset_index()
    write(out, 'table5_accuracy_vs_k')

# =============================================================================
# Table 6 — Algorithm-specific worst cases (Experiment 7)
# =============================================================================
def t6_worst_cases():
    df = load_no_starts('exp7_adversarial.csv').dropna(subset=['us'])
    pivot = (df.pivot_table(index='case', columns='algo', values='us',
                            aggfunc='first')[ALGOS]
              .rename(columns=LBL).round(0).astype('Int64'))
    pivot.index.name = 'Adversarial input'
    pivot = pivot.reindex(['control_dna','brute_worst','pigeon_worst','seed_worst'])
    write(pivot.reset_index(), 'table6_adversarial_us')

# =============================================================================
# Table 7 — Effect of error budget k on runtime (μs)
# =============================================================================
def t7_runtime_vs_k():
    df = load_no_starts('exp4_scaling_k.csv').dropna(subset=['us'])
    pivot = (df.groupby(['k_req','algo'])['us'].mean()
                .unstack('algo')[ALGOS].round(0).astype('Int64'))
    pivot.index.name = 'k'
    pivot = pivot.rename(columns=LBL).reset_index()
    write(pivot, 'table7_runtime_vs_k_us')

# =============================================================================
# Table 8 — Overall ranking (composite)
# =============================================================================
def t8_ranking():
    runs = load_no_starts('all_runs.csv').dropna(subset=['us'])
    acc  = pd.read_csv(RES/'all_accuracy.csv')
    skipped = (load_no_starts('all_runs.csv')
                .groupby('algo')['skipped']
                .apply(lambda s: int(s.fillna(False).astype(bool).sum())))
    rows = []
    for algo in ALGOS:
        sub_r = runs[runs.algo==algo]['us']
        sub_a = acc [acc .algo==algo]
        rows.append({
            'Algorithm':           LBL[algo],
            'Median runtime (μs)': int(sub_r.median()),
            'Mean runtime (μs)':   int(sub_r.mean()),
            'Mean precision':      round(float(sub_a.precision.mean()), 3),
            'Mean recall':         round(float(sub_a.recall.mean()),    3),
            'Cases skipped':       int(skipped.get(algo, 0)),
            'Verdict': {
                'brute':      'Pedagogical / tiny inputs only',
                'sellers':    'Reference exact algorithm — predictable',
                'pigeonhole': 'Fastest exact for small k; slow at large k',
                'seed':       'Fastest overall; trades recall for speed',
            }[algo],
        })
    write(pd.DataFrame(rows), 'table8_overall_ranking')

if __name__ == '__main__':
    print('Generating tables into FINAL/tables/ :')
    t1_theoretical()
    t2_empirical_exponents()
    t3_runtime_summary()
    t4_runtime_by_family()
    t5_accuracy_vs_k()
    t6_worst_cases()
    t7_runtime_vs_k()
    t8_ranking()
    print('\nDone.')
