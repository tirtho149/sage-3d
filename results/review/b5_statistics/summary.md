# B5: statistics -- paired values, leave-one-class-out, and what the design can detect

13 comparisons enter the Holm correction. `exact p floor` = 2/2^(non-tied pairs), the smallest two-sided p
the exact Wilcoxon test can give; `Holm floor` = that floor times this comparison's Holm multiplier.
If the Holm floor exceeds 0.05, no effect size could survive correction with this n.

| comparison | metric | n (nonzero) | classes | mean diff | specimen CI | p | Holm p | exact floor | Holm floor | could survive | sign holds leaving any class out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| J vs G | geometry_recall | 6 (4) | 2 | +0.370 | [+0.13, +0.61] | 0.125 | 0.875 | 0.125 | 0.875 | no | yes |
| J vs G | unsupported_rate | 9 (4) | 3 | -0.105 | [-0.28, +0.03] | 0.375 | 1.000 | 0.125 | 0.500 | no | yes |
| J vs E | coherence | 9 (9) | 3 | +0.498 | [+0.37, +0.61] | 0.004 | 0.043 | 0.004 | 0.043 | yes | yes |
| J vs E | grain_ratio | 9 (9) | 3 | -11.659 | [-15.89, -7.94] | 0.004 | 0.043 | 0.004 | 0.039 | yes | yes |
| K vs E | appearance_recall | 7 (3) | 3 | +0.214 | [-0.14, +0.57] | 0.500 | 1.000 | 0.250 | 0.750 | no | NO |
| K vs E | grain_ratio | 15 (15) | 5 | -8.267 | [-14.15, -4.24] | 0.000 | 0.001 | 0.000 | 0.001 | yes | yes |
| K vs E | fuzz_ratio | 15 (15) | 5 | -0.895 | [-1.01, -0.78] | 0.000 | 0.001 | 0.000 | 0.001 | yes | yes |
| H vs F | geometry_recall | 6 (4) | 2 | +0.556 | [+0.22, +0.89] | 0.125 | 0.875 | 0.125 | 0.750 | no | yes |
| H vs F | appearance_recall | 3 (0) | 1 | +0.000 | [+0.00, +0.00] | - | - | 1.000 | - | no | - |
| H vs F | grain_ratio | 9 (9) | 3 | -8.640 | [-11.96, -5.45] | 0.004 | 0.043 | 0.004 | 0.035 | yes | yes |
| F vs E | appearance_recall | 7 (3) | 3 | +0.214 | [-0.14, +0.57] | 0.500 | 1.000 | 0.250 | 0.500 | no | NO |
| H vs G | appearance_recall | 3 (2) | 1 | +0.667 | [+0.00, +1.00] | 0.500 | 1.000 | 0.500 | 0.500 | no | - |
| H vs G | geometry_recall | 6 (4) | 2 | +0.481 | [+0.17, +0.80] | 0.125 | 0.875 | 0.125 | 0.625 | no | yes |
| F vs E | coherence | 15 (15) | 5 | +0.002 | [+0.00, +0.00] | 0.030 | 0.241 | 0.000 | 0.000 | yes | yes |
| K vs E | appearance_recall_v1 | 3 (0) | 1 | +0.000 | [+0.00, +0.00] | - | - | 1.000 | - | no | - |
| F vs E | appearance_recall_v1 | 3 (0) | 1 | +0.000 | [+0.00, +0.00] | - | - | 1.000 | - | no | - |

Primary biological outcome (fixed in reports/13 before any rating): `f_overall_accept`, Holm over 3 comparisons.
Per-specimen paired values: paired_values.csv.
