# Comparative Analysis of Algorithms for Approximate String Matching with Edit Distance ≤ k

> **CSE 317 — Algorithms: Design and Analysis · Spring 2026**
> Group project comparing four algorithmic paradigms on the same fundamental problem.

[![Report](https://img.shields.io/badge/report-PDF-blue)](report/)
[![Algorithms](https://img.shields.io/badge/algorithms-4-green)]()
[![Cases](https://img.shields.io/badge/cases-1086-orange)]()
[![Language](https://img.shields.io/badge/C%2B%2B-17-blue)]()

---

## Problem

Given a text $T$ of length $n$, a pattern $P$ of length $m$, and an integer $k \ge 0$,
report **every position** in $T$ where some substring is within edit (Levenshtein)
distance $k$ of $P$. This generalises exact substring search (KMP, Boyer–Moore)
and powers everything from spell-checkers to genome alignment.

We implement and benchmark four algorithms drawn from four different paradigms,
then compare them empirically on **1,086 test cases** spanning eight input
families, six text lengths, nine pattern lengths, and nine error budgets.

---

## Team & Algorithm Ownership

| Member               | Algorithm                  | Paradigm                  |
| -------------------- | -------------------------- | ------------------------- |
| **Abdullah Khalid**  | Recursive Brute Force      | Branch-and-Bound          |
| **Kashish Anil Kumar** | Sellers' Dynamic Programming | Dynamic Programming     |
| **Muskan Pawan**     | Pigeonhole Filter          | Divide-and-Conquer        |
| **Abdullah Irfan**   | Seed-and-Extend            | Greedy / Approximation    |

---

## Headline Results

| Metric                                  | Brute    | Sellers  | Pigeonhole | Seed     |
| --------------------------------------- | -------- | -------- | ---------- | -------- |
| Empirical exponent on $n$ (power-law)   | **3.45** | **1.37** | **1.41**   | **1.38** |
| Mean precision $\bar{P}$                | 0.90     | **1.00** | **1.00**   | 1.00     |
| Mean recall $\bar{R}$                   | 0.94     | **1.00** | **1.00**   | **0.24** |
| Mean runtime (µs, all cases)            | 2,087    | 3,347    | 59,579     | 7,810    |

**Key takeaway:** Sellers' DP is the only universally accurate-and-fast algorithm
when nothing is known about $k$ in advance; Pigeonhole edges it out at small $k$ but
collapses as $k$ grows; Seed-and-Extend is fast but misses indel matches by design
(the cause of its $\bar{R} = 0.24$). Full details, complexity tables, and
experiment-by-experiment analysis are in **[report/](report/)**.

---

## Folder Layout

```
.
├── README.md                                            ← you are here
├── Project Details.pdf                                  Course assignment brief
│
├── Project ideas with list of names of team member/    First-milestone proposal PDF
│
├── All group members and their repective algorithms/   Per-member standalone source
│   ├── Abdullah Khalid Recursive Brute Force/
│   ├── Kashish Anil Kumar Sellers' Dynamic Programming/
│   ├── Muskan Pawan Pigeonhole Filter (D&C)/
│   └── Abdullah Irfan Seed-and-Extend/
│
├── benchmark/                                           Unified C++ harness + Python analysis
│   ├── data/                                            Input text/pattern files
│   ├── results/                                         Raw experiment CSVs
│   ├── tables/                                          Formatted summary tables
│   │   └── visuals/                                     PNG renders of those tables
│   └── figures/                                         All plots used in the report
│       ├── all/                                         Cross-algorithm experiment plots
│       ├── analysis/                                    Pareto, theory-vs-empirical, etc.
│       └── brute/, pigeonhole/, seed/, sellers/         Per-algorithm detail panels
│
├── presentation/                                        Mid-semester slide deck (PDF)
│
└── report/                                              Final compiled report PDF
```

### What's in each folder

| Folder | Contents |
| ------ | -------- |
| `Project ideas with list of names of team member/` | First-milestone proposal PDF naming the team and the chosen problem. |
| `All group members and their repective algorithms/` | Four subfolders, one per author, each containing a single C++ source file with that member's standalone reference implementation. These are the original turn-ins, kept verbatim. |
| `benchmark/` | The reproducible benchmark. Contains the unified C++ harness (`bench.cpp`, `bench.exe`), the master Jupyter notebook (`comparative_analysis.executed.ipynb`) and helper Python scripts (`make_tables.py`, `make_table_visuals.py`, `make_analysis_figures.py`, `make_per_algo_figures.py`, `rerun_pigeonhole.py`), plus the four output subfolders described below. |
| `benchmark/data/` | All input text and pattern files used as benchmark inputs (one per case). |
| `benchmark/results/` | Raw measurement CSVs: `all_runs.csv`, `all_accuracy.csv`, `leaderboard.csv`, `theoretical_vs_empirical.csv`, plus per-experiment slices `exp1_*.csv` through `exp8_*.csv`. |
| `benchmark/tables/` | Eight numbered summary tables (`table1_*` … `table8_*`), each as both `.csv` (machine-readable) and `.md` (human-readable). |
| `benchmark/tables/visuals/` | PNG renders of those eight tables for slide reuse. |
| `benchmark/figures/` | All plots used throughout the report and slides. Subfolders: `all/` (cross-algorithm experiment plots), `analysis/` (Pareto, theory-vs-empirical, runtime distributions), and `brute/` `pigeonhole/` `seed/` `sellers/` (per-algorithm detail panels). |
| `presentation/` | Compiled PDF of the mid-semester slide deck. |
| `report/` | The final compiled comprehensive report PDF (built on Overleaf from the LaTeX source). |

---

## How to Run the Benchmark Yourself

```powershell
# 1. Build the unified C++ harness
cd benchmark
g++ -std=c++17 -O2 bench.cpp -o bench.exe

# 2. Smoke-test all four algorithms on a tiny example
Set-Content -Encoding ASCII -NoNewline data/_t.txt `
   "ATCGGTAATCGTACAATCGGTAGGGATCGTAATCGTA"
Set-Content -Encoding ASCII -NoNewline data/_p.txt "ATCGTA"
foreach ($a in 'brute','sellers','pigeonhole','seed') {
    .\bench.exe $a data/_t.txt data/_p.txt 1
}

# 3. Re-execute the master notebook to regenerate every CSV and figure
python -m nbconvert --to notebook --execute comparative_analysis.executed.ipynb `
       --output comparative_analysis.executed.ipynb `
       --ExecutePreprocessor.timeout=900
```

All randomness is seeded (`random.Random(42)`); a clean re-run reproduces every
table and figure in the report bit-for-bit.

**CLI of the unified harness:**

```
bench <algo> <T-file> <P-file> <k>
```

where `<algo>` is one of `brute`, `sellers`, `pigeonhole`, `seed`.

---

## Reading the Report

The final deliverable is the comprehensive PDF report under **[report/](report/)**.
It is structured as:

1. Problem Statement (with a worked DNA example)
2. Algorithm Descriptions (one section per paradigm, with pseudocode)
3. Data Details (eight input families)
4. Experimental Setup
5. Empirical Results (seven experiments)
6. Discussion and Conclusion
7. References + Reproduction recipe

For per-experiment commentary, head straight to **§5** of the report.

---

## License & Acknowledgements

Coursework artifact for **CSE 317, Spring 2026**. Implementations and analysis
are original work by the four team members listed above; algorithmic credit is
given inline in the report's bibliography (Wagner–Fischer 1974, Sellers 1980,
Altschul et al. 1990 BLAST, etc.).
