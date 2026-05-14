# Comparative Benchmark — Approximate String Matching

Unified harness for the four algorithms our group implemented for the
**CSE 317 Spring 2026** project:

| Member   | Algorithm           | Paradigm                | File of origin                               |
|----------|---------------------|-------------------------|----------------------------------------------|
| Khalid   | Recursive Brute Force | Branch-and-Bound      | `../Muskans algo/brute_force.cpp.txt`        |
| Kashish  | Sellers'             | Dynamic Programming    | `../Kashishs algo/sellers_algorithm.cpp`     |
| Muskan   | Pigeonhole Filter    | Divide-and-Conquer     | `../Muskans algo/Pigeonhole Principle.txt`   |
| Abdullah | Seed-and-Extend      | Greedy / Approximation | `../Abdullahs algo/seed_and_extend.cpp`      |

All four algorithms are re-implemented in the single binary `bench.cpp` so
that the comparison is unbiased: same compiler flags, same machine, same
input files, same wall-clock timer. The originals in each member's folder
remain untouched.

## Files

| File                            | Purpose                                            |
|---------------------------------|----------------------------------------------------|
| `bench.cpp`                     | All four algorithms behind one CLI.                |
| `comparative_analysis.ipynb`    | Generates the dataset, runs every experiment, plots. |
| `comparative_analysis.executed.ipynb` | The version with all outputs already populated. |
| `data/`                         | Per-case text/pattern files written by the notebook. |
| `results/`                      | CSVs of every experiment.                          |
| `figures/`                      | PNG plots used in the report.                      |

## Build & run

```powershell
# 1. Build the benchmark binary (only needed once).
g++ -std=c++17 -O2 bench.cpp -o bench.exe

# 2. Smoke test — run each algorithm on a tiny example.
Set-Content -Encoding ASCII -NoNewline -Path data/_t.txt -Value "ATCGGTAATCGTACAATCGGTAGGGATCGTAATCGTA"
Set-Content -Encoding ASCII -NoNewline -Path data/_p.txt -Value "ATCGTA"
foreach ($a in 'brute','sellers','pigeonhole','seed') { .\bench.exe $a data/_t.txt data/_p.txt 1 }

# 3. Run the full notebook (re-generates all CSVs and PNGs).
python -m nbconvert --to notebook --execute comparative_analysis.ipynb `
    --output comparative_analysis.executed.ipynb --ExecutePreprocessor.timeout=900
```

## CLI contract for `bench.exe`

```
bench.exe <brute|sellers|pigeonhole|seed> <text_file> <pattern_file> <k>
```

Stdout is two lines (machine-readable for the notebook):

```
<algo>,<n>,<m>,<k>,<num_matches>,<elapsed_microseconds>
<comma-separated 0-based start positions of the matches>
```

## What the notebook produces

1. **Experiment 1 — input families.** Same `(n, m, k)` across DNA / English /
   binary / repetitive / low-entropy DNA / adversarial texts. Shows how each
   algorithm reacts to the *structure* of the input.
2. **Experiment 2 — scaling with n.** `m` and `k` fixed, `n` swept over two
   decades. Used to fit empirical complexity exponents.
3. **Experiment 3 — scaling with m.** Pattern length swept; `n` and `k` fixed.
4. **Experiment 4 — scaling with k.** Error budget swept; the most diagnostic
   axis (brute force explodes, pigeonhole filter loosens, seed-and-extend's
   recall deteriorates).
5. **Experiment 5 — accuracy.** Precision and recall against Sellers' DP as
   ground truth, using a `±k` windowed evaluation (standard in the
   approximate-matching literature because edit distance does not produce
   unique starting positions).
6. **Theoretical vs empirical.** Power-law fit `T(n) = a·n^b` and table of
   empirical vs textbook exponents.
7. **Leaderboard.** Median / mean / max runtime, plus average precision /
   recall, per algorithm.

## Notes for the report

- Brute force is automatically skipped on cases where `n·3^k` would exceed
  ~5 M operations, to keep the notebook tractable. Skipped runs appear as
  `NaN` in the CSVs and as gaps in the plots.
- The notebook uses **best-of-3** repeats per case to dampen OS jitter.
- All randomness is seeded (`random.Random(42)`) so the dataset is
  reproducible.
- The `±k` window in the accuracy comparison can flag legitimate
  *re-discoveries* of the same true match at slightly different start
  positions as false positives. This is a known artefact of windowed
  evaluation and is acknowledged in the discussion section of the notebook.
