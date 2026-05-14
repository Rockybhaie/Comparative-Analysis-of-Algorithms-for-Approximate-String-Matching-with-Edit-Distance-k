| Algorithm       |   Median runtime (μs) |   Mean runtime (μs) |   Mean precision |   Mean recall |   Cases skipped | Verdict                                    |
|:----------------|----------------------:|--------------------:|-----------------:|--------------:|----------------:|:-------------------------------------------|
| Brute Force     |                     0 |                2086 |            0.902 |         0.94  |              74 | Pedagogical / tiny inputs only             |
| Sellers' DP     |                     0 |                3347 |            1     |         1     |               0 | Reference exact algorithm — predictable    |
| Pigeonhole D&C  |                     0 |               59578 |            1     |         1     |               0 | Fastest exact for small k; slow at large k |
| Seed-and-Extend |                   995 |                7809 |            1     |         0.239 |               0 | Fastest overall; trades recall for speed   |
