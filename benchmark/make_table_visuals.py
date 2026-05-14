"""
Render each table from make_tables.py as a polished PNG figure suitable for
the report (no spreadsheet copy-paste needed).

Output: FINAL/tables/visuals/  (and benchmark/tables/visuals/)
"""
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

ROOT  = Path(__file__).resolve().parent
TBL   = ROOT / 'tables'
OUT   = TBL / 'visuals'
FINAL = ROOT.parent / 'FINAL' / 'tables' / 'visuals'
for d in (OUT, FINAL):
    d.mkdir(parents=True, exist_ok=True)

# Soft, report-friendly palette
HEADER_BG = '#2c3e50'
HEADER_FG = 'white'
ROW_ALT   = ['#ffffff', '#f4f6f9']
GRID_COL  = '#cfd6e0'
ALGO_TINT = {
    'Brute Force':      '#fbe9ea',
    'Brute Force (Khalid)': '#fbe9ea',
    "Sellers' DP":      '#e6eef9',
    "Sellers' DP (Kashish)": '#e6eef9',
    'Pigeonhole D&C':   '#e7f3e7',
    'Pigeonhole D&C (Muskan)': '#e7f3e7',
    'Seed-and-Extend':  '#fdeede',
    'Seed-and-Extend (Abdullah)': '#fdeede',
}

def _wrap_cells(df, wrap_widths):
    """Pre-wrap long string cells so they fit. wrap_widths is a dict
       {column_name: max_chars_per_line}. Other columns are left unchanged."""
    df = df.copy()
    for col, width in wrap_widths.items():
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda s: '\n'.join(textwrap.wrap(s, width=width)) if s else s)
    return df

def render_table(df, title, subtitle=None, filename=None,
                 col_widths=None, wrap_widths=None,
                 fontsize=10, row_height=0.6, figwidth=None):
    """Render df as a polished PNG.
       - col_widths : list of fractional widths summing to ~1 (or None for equal).
       - wrap_widths: {col_name: max_chars} → pre-wraps long text cells.
       - row_height : axis-fraction-per-row; bigger if cells will wrap.
    """
    if wrap_widths:
        df = _wrap_cells(df, wrap_widths)
    df = df.fillna('—').astype(str)

    nrows, ncols = df.shape
    if figwidth is None:
        figwidth = max(8, 1.6 * ncols + 1.5)
    # Allow rows that contain wrapped text to be taller.
    extra = sum(max(c.count('\n') for c in row) for row in df.values.tolist())
    figheight = 1.4 + row_height * (nrows + 1) + 0.35 * extra

    # Header is a fixed 1.1 inch tall band so title/subtitle never overlap.
    figheight = max(figheight, 2.2)
    fig = plt.figure(figsize=(figwidth, figheight))
    head_inches = 1.0
    head_frac   = head_inches / figheight
    ax_head = fig.add_axes([0.0, 1 - head_frac, 1.0, head_frac])
    ax_head.axis('off')
    ax_head.text(0.015, 0.72, title, fontsize=fontsize + 3, fontweight='bold',
                 ha='left', va='center', color='#1f2937')
    if subtitle:
        ax_head.text(0.015, 0.28, subtitle, fontsize=fontsize - 1,
                     color='#5a6776', style='italic', ha='left', va='center')

    # Table axes — placed just below the header band.
    ax = fig.add_axes([0.015, 0.03, 0.97, 1 - head_frac - 0.05])
    ax.axis('off')

    cells = df.values.tolist()
    cols  = list(df.columns)
    if col_widths is None:
        col_widths = [1.0 / ncols] * ncols

    table = ax.table(cellText=cells, colLabels=cols, cellLoc='center',
                     loc='center', colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.0)

    # Compute uniform row heights so wrapped cells don't misalign.
    base_h = 0.06
    line_h = 0.04
    row_heights = {0: 0.075}                                    # header
    for r in range(1, nrows + 1):
        max_lines = max(cells[r - 1][c].count('\n') for c in range(ncols))
        row_heights[r] = base_h + line_h * max_lines

    # Style cells
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COL)
        cell.set_linewidth(0.6)
        cell.set_height(row_heights[r])
        if r == 0:
            cell.set_facecolor(HEADER_BG)
            cell.get_text().set_color(HEADER_FG)
            cell.get_text().set_weight('bold')
        else:
            row_text = cells[r - 1]
            algo_name = row_text[0]
            tint = ALGO_TINT.get(algo_name, ROW_ALT[(r - 1) % 2])
            cell.set_facecolor(tint)
            if c > 0:
                cell.get_text().set_ha('right')
            else:
                cell.get_text().set_ha('left')
                cell.get_text().set_weight('bold')

    for d in (OUT, FINAL):
        fig.savefig(d / filename, dpi=160, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)
    print(f'  -> tables/visuals/{filename}')

# =============================================================================
# Render each table from the CSVs produced by make_tables.py
# =============================================================================
def t1():
    df = pd.read_csv(TBL / 'table1_theoretical_complexity.csv')
    render_table(df,
        title='Table 1 — Theoretical complexity',
        subtitle='Textbook best / average / worst-case time and space for each algorithm',
        filename='table1_theoretical_complexity.png',
        col_widths=[0.22, 0.13, 0.20, 0.18, 0.15, 0.12],
        wrap_widths={'Algorithm':22, 'Average-case time':22, 'Worst-case time':18,
                     'Space':18, 'Optimality':16},
        fontsize=10, figwidth=15)

def t2():
    df = pd.read_csv(TBL / 'table2_empirical_exponents.csv')
    render_table(df,
        title='Table 2 — Empirical vs theoretical exponent on n',
        subtitle='Power-law fit T(n) ∝ n^b on Experiment 2 (DNA, m=20, k=2). Theory: b = 1.0.',
        filename='table2_empirical_exponents.png',
        col_widths=[0.30, 0.18, 0.18, 0.18, 0.16],
        fontsize=10, figwidth=12)

def t3():
    df = pd.read_csv(TBL / 'table3_runtime_summary.csv')
    render_table(df,
        title='Table 3 — Runtime summary across all 3,440 cases',
        subtitle='Aggregated over every experiment, family, seed, and parameter combination',
        filename='table3_runtime_summary.png',
        col_widths=[0.20] + [0.10]*8,
        fontsize=10, figwidth=14)

def t4():
    df = pd.read_csv(TBL / 'table4_runtime_by_family_us.csv')
    render_table(df,
        title='Table 4 — Mean runtime by input family (μs)',
        subtitle='n = 5,000, m = 16, k = 2; mean over 5 random seeds',
        filename='table4_runtime_by_family_us.png',
        col_widths=[0.20, 0.20, 0.20, 0.20, 0.20],
        fontsize=10, figwidth=12)

def t5():
    df = pd.read_csv(TBL / 'table5_accuracy_vs_k.csv')
    # Tidy column names: shorten algorithm labels
    df.columns = [c.replace('Brute Force','Brute')
                   .replace("Sellers' DP",'Sellers')
                   .replace('Pigeonhole D&C','Pigeon')
                   .replace('Seed-and-Extend','Seed') for c in df.columns]
    render_table(df,
        title='Table 5 — Precision (P) and recall (R) vs error budget k',
        subtitle="Sellers' DP is the ground truth (windowed ±k 1-to-1 matching). DNA, n=20k, m=32.",
        filename='table5_accuracy_vs_k.png',
        fontsize=9, figwidth=14)

def t6():
    df = pd.read_csv(TBL / 'table6_adversarial_us.csv')
    render_table(df,
        title='Table 6 — Algorithm-specific worst-case inputs (μs)',
        subtitle="Runtime on hand-crafted inputs that target each algorithm's structural weakness",
        filename='table6_adversarial_us.png',
        col_widths=[0.20] + [0.20]*4,
        fontsize=10, figwidth=12)

def t7():
    df = pd.read_csv(TBL / 'table7_runtime_vs_k_us.csv')
    render_table(df,
        title='Table 7 — Mean runtime (μs) vs error budget k',
        subtitle='DNA family, n = 20k, m = 32; "—" indicates the algorithm was auto-skipped (infeasible)',
        filename='table7_runtime_vs_k_us.png',
        col_widths=[0.10, 0.225, 0.225, 0.225, 0.225],
        fontsize=10, figwidth=13)

def t8():
    df = pd.read_csv(TBL / 'table8_overall_ranking.csv')
    # Hand-shorten verdicts so they wrap cleanly within the column.
    short = {
        'Pedagogical / tiny inputs only':            'Pedagogical only;\ntiny inputs',
        'Reference exact algorithm — predictable':   'Reference exact;\npredictable',
        'Fastest exact for small k, fragile at large k':
            'Fastest exact for\nsmall k; fragile\nat large k',
        'Fastest exact for small k; slow at large k':
            'Fastest exact for\nsmall k; slow\nat large k',
        'Fastest overall; trades recall for speed':
            'Fastest overall;\ntrades recall for\nspeed',
    }
    df['Verdict'] = df['Verdict'].map(lambda s: short.get(s, s))
    render_table(df,
        title='Table 8 — Overall ranking',
        subtitle='Composite leaderboard aggregated over the entire study',
        filename='table8_overall_ranking.png',
        col_widths=[0.18, 0.12, 0.12, 0.11, 0.10, 0.10, 0.27],
        fontsize=10, figwidth=16, row_height=0.7)

if __name__ == '__main__':
    print('Rendering tables as PNGs into FINAL/tables/visuals/ :')
    for fn in (t1, t2, t3, t4, t5, t6, t7, t8):
        fn()
    print('\nDone.')
