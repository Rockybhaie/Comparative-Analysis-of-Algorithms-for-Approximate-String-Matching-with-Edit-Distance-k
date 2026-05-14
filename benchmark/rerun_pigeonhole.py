"""
Focused rerun: re-execute only the Pigeonhole algorithm on every benchmark
case after the verifier fix in bench.cpp, while preserving the existing
Brute/Sellers/Seed-and-Extend results untouched.

Strategy
--------
The notebook uses deterministic seeds, so re-running with the same
(family, n, m, k, seed) tuple reproduces the same text/pattern exactly.
We:
    1. parse every per-experiment CSV in results/
    2. find each row where algo == 'pigeonhole'
    3. regenerate the input via the same gen_text + plant_pattern code
    4. invoke bench.exe pigeonhole on the inputs
    5. overwrite the row's us, matches, starts, skipped
    6. write the CSV back
    7. rebuild the aggregated CSVs (all_runs, all_accuracy, leaderboard)
    8. invoke make_tables.py and make_table_visuals.py
    9. regenerate the inline notebook figures (exp1..exp4, fig_pareto, etc.)
   10. copy the relevant PNGs into presentation/assets/

This avoids rerunning Brute Force (12+ min of timeouts) and keeps the other
three algorithms' raw measurements intact.
"""
from __future__ import annotations

import ast
import math
import os
import random
import shutil
import string
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT      = Path(__file__).resolve().parent
BENCH_EXE = ROOT / ('bench.exe' if os.name == 'nt' else 'bench')
DATA_DIR  = ROOT / 'data';     DATA_DIR.mkdir(exist_ok=True)
RES       = ROOT / 'results';  RES.mkdir(exist_ok=True)
FIG       = ROOT / 'figures';  FIG.mkdir(exist_ok=True)

ASSETS    = ROOT.parent / 'presentation' / 'assets'
ASSETS.mkdir(parents=True, exist_ok=True)

# Match the notebook
SEEDS  = [42, 7, 1729, 31337, 271828]
ALGOS  = ['brute', 'sellers', 'pigeonhole', 'seed']
LBL    = {'brute':'Brute Force (Khalid)', 'sellers':"Sellers' DP (Kashish)",
          'pigeonhole':'Pigeonhole D&C (Muskan)', 'seed':'Seed-and-Extend (Abdullah)'}
COL    = {'brute':'#d62728', 'sellers':'#1f77b4',
          'pigeonhole':'#2ca02c', 'seed':'#ff7f0e'}

FAMILIES_ALL = ['dna','english','binary','repetitive','low_entropy_dna',
                'adversarial','real_dna','real_english']


# =============================================================================
# 1. INPUT GENERATORS  (ported verbatim from comparative_analysis.ipynb)
# =============================================================================
MARKOV_DNA = {
    'A': {'A':0.30,'C':0.18,'G':0.21,'T':0.31},
    'C': {'A':0.27,'C':0.27,'G':0.21,'T':0.25},
    'G': {'A':0.23,'C':0.32,'G':0.27,'T':0.18},
    'T': {'A':0.16,'C':0.23,'G':0.30,'T':0.31},
}
SHAKESPEARE = (
    "to be or not to be that is the question whether tis nobler in the mind "
    "to suffer the slings and arrows of outrageous fortune or to take arms "
    "against a sea of troubles and by opposing end them to die to sleep no "
    "more and by a sleep to say we end the heart ache and the thousand natural "
    "shocks that flesh is heir to tis a consummation devoutly to be wished to "
    "die to sleep to sleep perchance to dream ay there is the rub for in that "
    "sleep of death what dreams may come when we have shuffled off this mortal "
    "coil must give us pause there is the respect that makes calamity of so "
    "long life for who would bear the whips and scorns of time the oppressors "
    "wrong the proud mans contumely the pangs of despised love the laws delay "
    "the insolence of office and the spurns that patient merit of the unworthy "
    "takes when he himself might his quietus make with a bare bodkin who would "
    "these fardels bear to grunt and sweat under a weary life but that the "
    "dread of something after death the undiscovered country from whose bourn "
    "no traveller returns puzzles the will and makes us rather bear those ills "
    "we have than fly to others that we know not of thus conscience does make "
    "cowards of us all and thus the native hue of resolution is sicklied oer "
    "with the pale cast of thought and enterprises of great pith and moment "
    "with this regard their currents turn awry and lose the name of action")

def markov_dna(n, rng):
    cur = rng.choice('ACGT'); out = [cur]
    for _ in range(n - 1):
        d = MARKOV_DNA[cur]
        cur = rng.choices(list(d), weights=list(d.values()))[0]
        out.append(cur)
    return ''.join(out)

def tile_shakespeare(n):
    reps = (n // len(SHAKESPEARE)) + 1
    return (SHAKESPEARE * reps)[:n]

def gen_text(family, n, rng):
    if family == 'dna':              return ''.join(rng.choices('ACGT', k=n))
    if family == 'english':          return ''.join(rng.choices(string.ascii_lowercase, k=n))
    if family == 'binary':           return ''.join(rng.choices('01', k=n))
    if family == 'repetitive':
        s = list('A' * n)
        for _ in range(max(1, n // 200)):
            s[rng.randrange(n)] = 'T'
        return ''.join(s)
    if family == 'low_entropy_dna':  return ''.join(rng.choices('ACGT', weights=[40,10,10,40], k=n))
    if family == 'adversarial':
        motif = 'ACGTACGTACGTACGT'
        s = list(rng.choices('ACGT', k=n))
        for _ in range(max(1, n // (len(motif)*4))):
            pos = rng.randrange(0, max(1, n - len(motif)))
            mu = list(motif); mu[rng.randrange(len(motif))] = rng.choice('ACGT')
            s[pos:pos+len(motif)] = mu
        return ''.join(s)
    if family == 'real_dna':         return markov_dna(n, rng)
    if family == 'real_english':     return tile_shakespeare(n)
    raise ValueError(family)

def alphabet(family):
    return {'dna':'ACGT','english':string.ascii_lowercase,'binary':'01',
            'repetitive':'AT','low_entropy_dna':'ACGT','adversarial':'ACGT',
            'real_dna':'ACGT','real_english':None}[family]

def plant_pattern(text, m, k, family, rng):
    n = len(text)
    if m >= n: return text[:max(1,m)], 0
    pos = rng.randrange(0, n - m)
    pat = list(text[pos:pos+m])
    alpha = alphabet(family)
    if alpha is None:
        alpha = ''.join(sorted(set(text)))
    for _ in range(k):
        op = rng.choice(['sub','ins','del']) if len(pat) > 1 else 'sub'
        i = rng.randrange(len(pat))
        if op == 'sub':
            choices = [c for c in alpha if c != pat[i]]
            if choices: pat[i] = rng.choice(choices)
        elif op == 'ins':
            pat.insert(i, rng.choice(alpha))
        else:
            pat.pop(i)
    return ''.join(pat), pos

def gen_uniform(sigma, n, rng):
    base = (string.ascii_letters + string.digits + '!@#$%^&*()_+[]{}|;:,.<>/?`~-=' )[:sigma]
    return ''.join(rng.choices(base, k=n))


# =============================================================================
# 2. BENCHMARK INVOCATION
# =============================================================================
def write_text(p, s):
    p.write_text(s, encoding='ascii', errors='replace'); return p

def run_pigeon(text, pattern, k, repeats=3, timeout=60):
    """Run pigeonhole on a single (text, pattern, k); best-of-N us."""
    tT = write_text(DATA_DIR / '_T.txt', text)
    tP = write_text(DATA_DIR / '_P.txt', pattern)
    best_us, last_starts, last_matches = math.inf, None, None
    for _ in range(repeats):
        try:
            r = subprocess.run(
                [str(BENCH_EXE), 'pigeonhole', str(tT), str(tP), str(k)],
                capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        if r.returncode != 0:
            raise RuntimeError(f'pigeonhole failed: {r.stderr}')
        lines = r.stdout.strip().splitlines()
        head  = lines[0].split(',')
        starts = [int(x) for x in lines[1].split(',')] if len(lines) > 1 and lines[1] else []
        us = int(head[5])
        if us < best_us:
            best_us, last_starts, last_matches = us, starts, int(head[4])
    return dict(us=best_us, matches=last_matches, starts=last_starts)


# =============================================================================
# 3. INPUT REGENERATION  (matches notebook's bench_seeds protocol)
# =============================================================================
def regenerate_inputs(family, n_req, m_req, k_req, seed):
    """Reproduces (text, pattern) deterministically from a seed."""
    rng = random.Random(seed)
    text = gen_text(family, n_req, rng)
    pat, _ = plant_pattern(text, m_req, k_req, family, rng)
    return text, pat

def regenerate_inputs_sigma(sigma, n, m, k, seed):
    """For exp5 sigma sweep — uses gen_uniform + custom plant."""
    rng = random.Random(seed)
    text = gen_uniform(sigma, n, rng)
    pos  = rng.randrange(0, n - m)
    pat  = list(text[pos:pos+m])
    alpha = ''.join(sorted(set(text)))
    for _ in range(k):
        op = rng.choice(['sub','ins','del']) if len(pat) > 1 else 'sub'
        i = rng.randrange(len(pat))
        if op == 'sub':
            ch = [c for c in alpha if c != pat[i]]
            if ch: pat[i] = rng.choice(ch)
        elif op == 'ins':
            pat.insert(i, rng.choice(alpha))
        else:
            pat.pop(i)
    return text, ''.join(pat)


# =============================================================================
# 4. CSV-DRIVEN RERUN
# =============================================================================
def rerun_csv_pigeonhole(csv_path, regen_fn, label='?'):
    """Update only pigeonhole rows in csv_path; rest untouched.
    regen_fn(row) -> (text, pattern, k).  Returns count of rows updated.
    """
    df = pd.read_csv(csv_path)
    pig = df[df.algo == 'pigeonhole'].copy()
    print(f'  {label}: {len(pig)} pigeonhole rows to rerun')
    updates = 0
    for idx, row in pig.iterrows():
        try:
            text, pattern, k_use = regen_fn(row)
        except Exception as e:
            print(f'    !! could not regen row {idx}: {e}')
            continue
        result = run_pigeon(text, pattern, k_use)
        if result is None:
            df.at[idx, 'us']      = np.nan
            df.at[idx, 'matches'] = np.nan
            df.at[idx, 'starts']  = '[]'
            df.at[idx, 'skipped'] = True
        else:
            df.at[idx, 'us']      = result['us']
            df.at[idx, 'matches'] = result['matches']
            df.at[idx, 'starts']  = str(result['starts'])
            df.at[idx, 'skipped'] = False
        updates += 1
        if updates % 25 == 0:
            print(f'    ... {updates}/{len(pig)}')
    df.to_csv(csv_path, index=False)
    return updates


def regen_e1234(row):
    family = row['family']
    n      = int(row['n'])
    m_req  = int(row['m_req'])
    k_req  = int(row['k_req'])
    seed   = int(row['seed'])
    text, pat = regenerate_inputs(family, n, m_req, k_req, seed)
    return text, pat, k_req


def regen_e5(row):
    """Family column is 'sigma{N}'."""
    fam = row['family']
    sigma = int(str(fam).replace('sigma', ''))
    n     = int(row['n'])
    m_req = int(row['m_req'])
    k_req = int(row['k_req'])
    seed  = int(row['seed'])
    text, pat = regenerate_inputs_sigma(sigma, n, m_req, k_req, seed)
    return text, pat, k_req


def regen_e6(row):
    """Edge cases - reconstruct from label."""
    rng = random.Random(0)
    base = gen_text('dna', 5_000, rng)
    cases = {
        'm_gt_n':           ('AAAAA'*20,                  'A'*200,             1),
        'k_ge_m':           (base,                        'ACGT',              4),
        'no_match':         ('A'*5_000,                   'CGCGCGCGCGCG',      1),
        'pattern_at_start': (base,                        base[:20],           1),
        'pattern_at_end':   (base,                        base[-20:],          1),
        'pattern_eq_text':  (base[:200],                  base[:200],          0),
    }
    label = row['label']
    text, pat, k_use = cases[label]
    return text, pat, k_use


def regen_e7(row):
    """Adversarial cases - reconstruct exact inputs."""
    rng = random.Random(0)
    t4 = gen_text('dna', 10_000, rng)
    pat4, _ = plant_pattern(t4, 20, 2, 'dna', rng)
    cases = {
        'brute_worst':  ('A'*8_000,                                 'A'*16,                2),
        'pigeon_worst': ('AT'* (10_000 // 2),                       'ATATATATATATATAT',    3),
        'seed_worst':   ('ACAC' * (10_000 // 4),                    'ACAC'*5,              2),
        'control_dna':  (t4,                                        pat4,                  2),
    }
    label = row['label']
    text, pat, k_use = cases[label]
    return text, pat, k_use


def step_rerun():
    print('\n[1/4] Rerunning pigeonhole over every experiment CSV ...')
    rerun_csv_pigeonhole(RES / 'exp1_families.csv',  regen_e1234, label='exp1_families')
    rerun_csv_pigeonhole(RES / 'exp2_scaling_n.csv', regen_e1234, label='exp2_scaling_n')
    rerun_csv_pigeonhole(RES / 'exp3_scaling_m.csv', regen_e1234, label='exp3_scaling_m')
    rerun_csv_pigeonhole(RES / 'exp4_scaling_k.csv', regen_e1234, label='exp4_scaling_k')
    rerun_csv_pigeonhole(RES / 'exp5_sigma.csv',     regen_e5,    label='exp5_sigma')
    rerun_csv_pigeonhole(RES / 'exp6_edge.csv',      regen_e6,    label='exp6_edge')
    rerun_csv_pigeonhole(RES / 'exp7_adversarial.csv', regen_e7,  label='exp7_adversarial')


# =============================================================================
# 5. ACCURACY  (windowed evaluation against Sellers' DP ground truth)
# =============================================================================
def _parse_starts(s):
    if isinstance(s, list): return s
    if pd.isna(s) or s in ('', '[]'): return []
    try:
        return ast.literal_eval(s) if isinstance(s, str) and s.startswith('[') else \
               [int(x) for x in str(s).split(',') if x]
    except Exception:
        return []

def windowed_match(predicted, gt, tol):
    """O(P log P + G log G) windowed 1-to-1 matching.
    A predicted position is a TP iff it is within ±tol of some still-unused
    ground-truth position.  Greedy left-to-right pairing on sorted lists."""
    pred_sorted = sorted(predicted)
    gt_sorted   = sorted(gt)
    used = [False] * len(gt_sorted)
    tp, j_start = 0, 0
    for p in pred_sorted:
        # advance window start past gt values that are < p - tol (unreachable)
        while j_start < len(gt_sorted) and gt_sorted[j_start] < p - tol:
            j_start += 1
        j = j_start
        while j < len(gt_sorted) and gt_sorted[j] <= p + tol:
            if not used[j]:
                used[j] = True; tp += 1; break
            j += 1
    fp = len(pred_sorted) - tp
    fn = len(gt_sorted)   - tp
    pr = tp / (tp + fp) if (tp + fp) else 1.0
    rc = tp / (tp + fn) if (tp + fn) else 1.0
    return pr, rc

def accuracy_table(df):
    rows = []
    for key, grp in df.groupby(['label','n','m_req','k_req']):
        lab, n, m_req, k_req = key
        gt = grp[grp.algo == 'sellers']
        if gt.empty or bool(gt.iloc[0]['skipped']): continue
        gt_starts = _parse_starts(gt.iloc[0]['starts'])
        for _, r in grp.iterrows():
            if bool(r['skipped']): continue
            pr, rc = windowed_match(_parse_starts(r['starts']), gt_starts, tol=int(k_req))
            rows.append(dict(label=lab, family=r['family'], algo=r['algo'],
                             n=int(n), m=int(m_req), k=int(k_req),
                             precision=pr, recall=rc))
    return pd.DataFrame(rows)


def step_recompute_accuracy():
    print('\n[2/4] Recomputing accuracy CSVs ...')
    # accuracy per k (from exp4) and per family (from exp1).
    df_e1 = pd.read_csv(RES / 'exp1_families.csv')
    df_e2 = pd.read_csv(RES / 'exp2_scaling_n.csv')
    df_e3 = pd.read_csv(RES / 'exp3_scaling_m.csv')
    df_e4 = pd.read_csv(RES / 'exp4_scaling_k.csv')

    acc_k = accuracy_table(df_e4)
    acc_k_agg = acc_k.groupby(['k','algo'])[['precision','recall']].mean().reset_index()
    acc_k_agg.to_csv(RES / 'exp8_accuracy_vs_k.csv', index=False)
    print(f'  exp8_accuracy_vs_k.csv  rows={len(acc_k_agg)}')

    acc_fam = accuracy_table(df_e1)
    acc_fam_agg = acc_fam.groupby(['family','algo'])[['precision','recall']].mean().reset_index()
    acc_fam_agg.to_csv(RES / 'exp8_accuracy_by_family.csv', index=False)
    print(f'  exp8_accuracy_by_family.csv  rows={len(acc_fam_agg)}')

    all_acc = pd.concat([accuracy_table(df_e1),
                         accuracy_table(df_e2),
                         accuracy_table(df_e3),
                         accuracy_table(df_e4)], ignore_index=True)
    all_acc.to_csv(RES / 'all_accuracy.csv', index=False)
    print(f'  all_accuracy.csv  rows={len(all_acc)}')

    # Rebuild all_runs.csv
    parts = []
    for exp_csv, tag in [('exp1_families.csv','1_families'),
                          ('exp2_scaling_n.csv','2_scaling_n'),
                          ('exp3_scaling_m.csv','3_scaling_m'),
                          ('exp4_scaling_k.csv','4_scaling_k'),
                          ('exp5_sigma.csv','5_sigma'),
                          ('exp6_edge.csv','6_edge'),
                          ('exp7_adversarial.csv','7_adversarial')]:
        p = RES / exp_csv
        if p.exists():
            d = pd.read_csv(p, usecols=lambda c: c != 'starts')
            d['experiment'] = tag
            parts.append(d)
    all_runs = pd.concat(parts, ignore_index=True)
    all_runs.to_csv(RES / 'all_runs.csv', index=False)
    print(f'  all_runs.csv  rows={len(all_runs)}')

    # Leaderboard
    speed = (all_runs.dropna(subset=['us']).groupby('algo')
             .agg(median_us=('us','median'), mean_us=('us','mean'),
                  max_us=('us','max'), cases_run=('us','count'),
                  cases_skipped=('skipped', lambda s: int(s.fillna(False).astype(bool).sum()))))
    acc = (all_acc.groupby('algo')
           .agg(precision=('precision','mean'),
                recall   =('recall','mean')))
    leaderboard = speed.join(acc).reindex(ALGOS).round(3)
    leaderboard.to_csv(RES / 'leaderboard.csv')
    print('  leaderboard.csv updated')
    print(leaderboard)


# =============================================================================
# 6. PLOTTING  (port of inline notebook plotting cells)
# =============================================================================
def _agg_seeds(df, *group_cols):
    g = df.dropna(subset=['us']).groupby(['algo', *group_cols])
    return g['us'].agg(['mean','std','count']).reset_index().rename(
        columns={'mean':'us_mean','std':'us_std','count':'n_seeds'})

def step_regenerate_figures():
    print('\n[3/4] Regenerating PNG figures ...')

    # ---- exp1: 8-family bar chart -------------------------------------------
    df1   = pd.read_csv(RES / 'exp1_families.csv', usecols=lambda c: c != 'starts')
    agg1  = _agg_seeds(df1, 'family')
    fig, ax = plt.subplots(figsize=(12,5))
    x = np.arange(len(FAMILIES_ALL)); width = 0.2
    for i, a in enumerate(ALGOS):
        sub = agg1[agg1.algo==a].set_index('family').reindex(FAMILIES_ALL)
        se  = sub.us_std / np.sqrt(sub.n_seeds.replace(0,1))
        ax.bar(x + (i-1.5)*width, sub.us_mean.fillna(0), width,
               yerr=se.fillna(0), label=LBL[a], color=COL[a], capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(FAMILIES_ALL, rotation=20, ha='right')
    ax.set_yscale('log'); ax.set_ylabel('Runtime (μs, log scale)')
    ax.set_title('Runtime across 8 input families  (n=5000, m=16, k=2; mean ±SE over 5 seeds)')
    ax.legend(fontsize=8); ax.grid(True, axis='y', ls=':', alpha=0.5)
    plt.tight_layout(); fig.savefig(FIG / 'exp1_families.png', dpi=140); plt.close(fig)
    print('  exp1_families.png')

    # ---- exp2: 8-panel n-scaling --------------------------------------------
    df2  = pd.read_csv(RES / 'exp2_scaling_n.csv', usecols=lambda c: c != 'starts')
    agg2 = _agg_seeds(df2, 'family', 'n')
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    for ax, fam in zip(axes.flat, FAMILIES_ALL):
        for a in ALGOS:
            sub = agg2[(agg2.family==fam) & (agg2.algo==a)].sort_values('n')
            if sub.empty: continue
            se = sub.us_std / np.sqrt(sub.n_seeds.replace(0,1))
            ax.errorbar(sub.n, sub.us_mean, yerr=se, marker='o', ms=4,
                        label=LBL[a].split(' (')[0], color=COL[a])
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(fam, fontsize=9)
        ax.grid(True, which='both', ls=':', alpha=0.4)
    for ax in axes[-1, :]: ax.set_xlabel('n (log)')
    for ax in axes[:, 0]:  ax.set_ylabel('μs (log)')
    axes[0,0].legend(fontsize=7, loc='upper left')
    fig.suptitle('Scaling with n by family  (m=20, k=2; mean ±SE)', y=1.02)
    plt.tight_layout(); fig.savefig(FIG / 'exp2_scaling_n.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('  exp2_scaling_n.png')

    # ---- exp3: m-scaling line plot ------------------------------------------
    df3  = pd.read_csv(RES / 'exp3_scaling_m.csv', usecols=lambda c: c != 'starts')
    agg3 = _agg_seeds(df3, 'm_req')
    fig, ax = plt.subplots(figsize=(9,5))
    for a in ALGOS:
        sub = agg3[agg3.algo==a].sort_values('m_req')
        if sub.empty: continue
        se  = sub.us_std / np.sqrt(sub.n_seeds.replace(0,1))
        ax.errorbar(sub.m_req, sub.us_mean, yerr=se, marker='o',
                    label=LBL[a], color=COL[a])
    ax.set_yscale('log'); ax.set_xlabel('Pattern length m'); ax.set_ylabel('Runtime μs (log)')
    ax.set_title('Scaling with m  (n=20000, k=2, family=dna; mean ±SE over 5 seeds)')
    ax.grid(True, which='both', ls=':', alpha=0.5); ax.legend()
    plt.tight_layout(); fig.savefig(FIG / 'exp3_scaling_m.png', dpi=140); plt.close(fig)
    print('  exp3_scaling_m.png')

    # ---- exp4: k-scaling line plot ------------------------------------------
    df4  = pd.read_csv(RES / 'exp4_scaling_k.csv', usecols=lambda c: c != 'starts')
    agg4 = _agg_seeds(df4, 'k_req')
    fig, ax = plt.subplots(figsize=(9,5))
    for a in ALGOS:
        sub = agg4[agg4.algo==a].sort_values('k_req')
        if sub.empty: continue
        se  = sub.us_std / np.sqrt(sub.n_seeds.replace(0,1))
        ax.errorbar(sub.k_req, sub.us_mean, yerr=se, marker='o',
                    label=LBL[a], color=COL[a])
    ax.set_yscale('log'); ax.set_xlabel('Error budget k'); ax.set_ylabel('Runtime μs (log)')
    ax.set_title('Scaling with k  (n=20000, m=32, family=dna; mean ±SE over 5 seeds)')
    ax.grid(True, which='both', ls=':', alpha=0.5); ax.legend()
    plt.tight_layout(); fig.savefig(FIG / 'exp4_scaling_k.png', dpi=140); plt.close(fig)
    print('  exp4_scaling_k.png')


def step_analysis_and_tables():
    print('\n[3/4b] Running make_analysis_figures + make_tables + make_table_visuals ...')
    for script in ['make_analysis_figures.py', 'make_tables.py',
                    'make_table_visuals.py', 'make_per_algo_figures.py']:
        path = ROOT / script
        if not path.exists():
            print(f'  skip (missing): {script}')
            continue
        r = subprocess.run([sys.executable, str(path)], cwd=str(ROOT),
                           capture_output=True, text=True)
        print(f'  -- {script} -- rc={r.returncode}')
        if r.returncode != 0:
            print('     stdout:', r.stdout[-1000:])
            print('     stderr:', r.stderr[-1000:])


# =============================================================================
# 7. COPY PNGs INTO presentation/assets/
# =============================================================================
PNGS_FOR_DECK = [
    # exp1..exp4 get moved into figures/all/ by make_per_algo_figures.py
    ('exp1_families.png',                FIG / 'all' / 'exp1_families.png'),
    ('exp2_scaling_n.png',               FIG / 'all' / 'exp2_scaling_n.png'),
    ('exp3_scaling_m.png',               FIG / 'all' / 'exp3_scaling_m.png'),
    ('exp4_scaling_k.png',               FIG / 'all' / 'exp4_scaling_k.png'),
    ('fig_empirical_vs_theoretical.png', FIG / 'analysis' / 'fig_empirical_vs_theoretical.png'),
    ('fig_pareto_speed_recall.png',      FIG / 'analysis' / 'fig_pareto_speed_recall.png'),
    ('table5_accuracy_vs_k.png',         ROOT / 'tables' / 'visuals' / 'table5_accuracy_vs_k.png'),
    ('table8_overall_ranking.png',       ROOT / 'tables' / 'visuals' / 'table8_overall_ranking.png'),
    ('algo_seed.png',                    FIG / 'seed' / 'algo_seed.png'),
]

def step_copy_to_assets():
    print('\n[4/4] Copying regenerated PNGs into presentation/assets/ ...')
    for dest_name, src in PNGS_FOR_DECK:
        if not src.exists():
            print(f'  !! missing source: {src}')
            continue
        shutil.copy(src, ASSETS / dest_name)
        print(f'  -> assets/{dest_name}')


# =============================================================================
# 8. MAIN
# =============================================================================
def main():
    assert BENCH_EXE.exists(), 'bench.exe not found — recompile first'
    skip_rerun = '--skip-rerun' in sys.argv
    if not skip_rerun:
        step_rerun()
    else:
        print('[1/4] (skipped — using existing CSVs)')
    step_recompute_accuracy()
    step_regenerate_figures()
    step_analysis_and_tables()
    step_copy_to_assets()
    print('\nAll done.')


if __name__ == '__main__':
    main()
