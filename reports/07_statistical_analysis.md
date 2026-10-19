# 07 — Statistical analysis (brief v2 §15)

- **Unit:** specimen. Seeds are averaged within a specimen; views are averaged within an asset.
- **Paired comparison per metric:** cluster bootstrap over disease classes (primary), specimen bootstrap (sensitivity), Wilcoxon signed-rank, Holm correction across the listed comparisons, and sign counts.
- **Power caveat:** the pilot has ≤ 15 specimens in ≤ 5 classes.
  - With fewer than 3 classes the cluster CI is degenerate.
  - Wilcoxon cannot reach p < 0.05 with n < 6.
  - The effect estimates and CIs are therefore the result; p-values are reported for completeness and none is claimed significant unless Holm p < 0.05.

| Comparison | Metric | Result | Wilcoxon p | Holm p |
|---|---|---|---|---|
| J vs G | geometry_recall | 0.44 vs 0.07; Δ +0.37 (cluster CI [+0.11, +0.63]; specimen CI [+0.13, +0.61]); a>b in 4/6, n=6 (2 classes) ⚠ <3 classes | 0.125 | 0.875 |
| J vs G | unsupported_rate | 0.06 vs 0.17; Δ -0.10 (cluster CI [-0.20, +0.00]; specimen CI [-0.28, +0.03]); a>b in 1/9, n=9 (3 classes) | 0.375 | 1.0 |
| J vs E | coherence | 0.81 vs 0.31; Δ +0.50 (cluster CI [+0.25, +0.66]; specimen CI [+0.37, +0.61]); a>b in 9/9, n=9 (3 classes) | 0.00390625 | 0.04296875 |
| J vs E | grain_ratio | 1.28 vs 12.94; Δ -11.66 (cluster CI [-14.68, -6.20]; specimen CI [-15.89, -7.94]); a>b in 0/9, n=9 (3 classes) | 0.00390625 | 0.04296875 |
| K vs E | appearance_recall | 0.50 vs 0.29; Δ +0.21 (cluster CI [-0.50, +0.67]; specimen CI [-0.14, +0.57]); a>b in 2/7, n=7 (3 classes) | 0.5 | 1.0 |
| K vs E | grain_ratio | 8.71 vs 16.98; Δ -8.27 (cluster CI [-12.81, -4.00]; specimen CI [-14.15, -4.24]); a>b in 0/15, n=15 (5 classes) | 6.103515625e-05 | 0.00079345703125 |
| K vs E | fuzz_ratio | 1.00 vs 1.90; Δ -0.90 (cluster CI [-1.00, -0.79]; specimen CI [-1.01, -0.78]); a>b in 0/15, n=15 (5 classes) | 6.103515625e-05 | 0.00079345703125 |
| H vs F | geometry_recall | 0.56 vs 0.00; Δ +0.56 (cluster CI [+0.33, +0.78]; specimen CI [+0.22, +0.89]); a>b in 4/6, n=6 (2 classes) ⚠ <3 classes | 0.125 | 0.875 |
| H vs F | appearance_recall | 0.67 vs 0.67; Δ +0.00 (cluster CI [+0.00, +0.00]; specimen CI [+0.00, +0.00]); a>b in 0/3, n=3 (1 classes) ⚠ <3 classes | nan | None |
| H vs F | grain_ratio | 7.09 vs 15.73; Δ -8.64 (cluster CI [-11.86, -2.93]; specimen CI [-11.96, -5.45]); a>b in 0/9, n=9 (3 classes) | 0.00390625 | 0.04296875 |
| F vs E | appearance_recall | 0.50 vs 0.29; Δ +0.21 (cluster CI [-0.50, +0.67]; specimen CI [-0.14, +0.57]); a>b in 2/7, n=7 (3 classes) | 0.5 | 1.0 |
| H vs G | appearance_recall | 0.67 vs 0.00; Δ +0.67 (cluster CI [+0.67, +0.67]; specimen CI [+0.00, +1.00]); a>b in 2/3, n=3 (1 classes) ⚠ <3 classes | 0.5 | 1.0 |
| H vs G | geometry_recall | 0.56 vs 0.07; Δ +0.48 (cluster CI [+0.33, +0.63]; specimen CI [+0.17, +0.80]); a>b in 4/6, n=6 (2 classes) ⚠ <3 classes | 0.125 | 0.875 |
| F vs E | coherence | 0.31 vs 0.31; Δ +0.00 (cluster CI [+0.00, +0.00]; specimen CI [+0.00, +0.00]); a>b in 11/15, n=15 (5 classes) | 0.0301513671875 | 0.2412109375 |
| K vs E | appearance_recall_v1 | 0.00 vs 0.00; Δ +0.00 (cluster CI [+0.00, +0.00]; specimen CI [+0.00, +0.00]); a>b in 0/3, n=3 (1 classes) ⚠ <3 classes | nan | None |
| F vs E | appearance_recall_v1 | 0.00 vs 0.00; Δ +0.00 (cluster CI [+0.00, +0.00]; specimen CI [+0.00, +0.00]); a>b in 0/3, n=3 (1 classes) ⚠ <3 classes | nan | None |

**2×2 ablation (geometry × KB):** `{"n_specimens_complete": 6, "cells": {"D (no geo, no KB)": 0.0, "K (no geo, KB)": 0.16666666666666666, "G (geo, no KB)": 0.037037037037037035, "H (geo, KB)": 0.5277777777777778}, "interaction_mean": 0.32407407407407407, "interaction_per_specimen": [0.0, 0.0, 1.0, 0.3333333333333333, 0.16666666666666663, 0.4444444444444444]}`

**Seed variability (J, KB geometric recall, per specimen):**
- Corn__Smut__0012: [0.0, 0.0, 0.0] (sd 0.00)
- Corn__Smut__0025: [0.0, 0.0, 0.0] (sd 0.00)
- Corn__Smut__0052: [0.0, 0.0, 1.0] (sd 0.47)
- Peach__Leaf_Curl__0006: [0.67, 0.67, 0.67] (sd 0.00)
- Peach__Leaf_Curl__0063: [0.67, 1.0, 0.67] (sd 0.16)
- Peach__Leaf_Curl__0087: [1.0, 0.67, 1.0] (sd 0.16)

**Failure rates:** see `results/statistics/qc_flags.csv` and report 09.
