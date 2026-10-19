# Expert-review analysis

**Data status: SIMULATED RATINGS - NOT EXPERT DATA. No real expert ratings exist yet (as of 2026-10-14).**

Raters: SIM01, SIM02, SIM03; flagged by catch rule (excluded from the main method-effect analysis, kept for agreement): none.

## 1. Agreement (consistency of the rubric; says nothing about which method is better)

| criterion | Krippendorff alpha | 95% CI (unit bootstrap) | ICC(2,k) | items |
|---|---|---|---|---|
| a_symptom_match (ordinal) | 0.155 | [0.012, 0.286] | 0.401 | 72 |
| b_distribution (ordinal) | 0.286 | [0.128, 0.426] | 0.587 | 72 |
| c_geometry (ordinal) | 0.232 | [0.078, 0.369] | 0.564 | 72 |
| d_surface (ordinal) | 0.262 | [0.091, 0.410] | 0.592 | 72 |
| e_artefacts (ordinal) | 0.091 | [-0.082, 0.239] | 0.392 | 72 |
| f_overall_accept (ordinal) | 0.248 | [0.100, 0.385] | 0.556 | 72 |
| prohibited_feature (nominal) | 0.037 | [-0.081, 0.160] | - | 72 |

Mean pairwise quadratic-weighted kappa per criterion:

- a_symptom_match: 0.180 over 3 pairs
- b_distribution: 0.321 over 3 pairs
- c_geometry: 0.306 over 3 pairs
- d_surface: 0.332 over 3 pairs
- e_artefacts: 0.197 over 3 pairs
- f_overall_accept: 0.312 over 3 pairs

Intra-rater (duplicated items, quadratic kappa / exact agreement / n pairs):

- SIM01: a 0.46/0.56/9; b 0.60/0.33/9; c 0.40/0.33/9; d 0.08/0.44/9; e 0.60/0.44/9; f 0.64/0.67/9
- SIM02: a 0.44/0.33/9; b 0.10/0.22/9; c 0.65/0.67/9; d 0.23/0.33/9; e 0.48/0.78/9; f -0.42/0.00/9
- SIM03: a 0.57/0.44/9; b -0.17/0.11/9; c 0.28/0.33/9; d 0.49/0.33/9; e 0.54/0.56/9; f 0.52/0.56/9

Catch items (pass = a_symptom_match <= 2 or prohibited_feature = yes):

- SIM01: 2/3 passed
- SIM02: 2/3 passed
- SIM03: 3/3 passed

## 2. Method effect (separate question; only interpretable where agreement alpha >= 0.667)

Planned paired comparisons over specimens (family mean per specimen, A minus B):

| criterion | A vs B | n spec | mean diff | 95% CI | p (sign-flip) | p Holm | alpha ok | status |
|---|---|---|---|---|---|---|---|---|
| a_symptom_match | KB-3D vs KB-2D | 9 | -0.235 | [-0.568, 0.087] | 0.281 | nan | no | secondary (exploratory, unadjusted) |
| a_symptom_match | KB-3D vs generic-3D | 9 | 0.562 | [0.290, 0.852] | 0.008 | nan | no | secondary (exploratory, unadjusted) |
| a_symptom_match | KB-2D vs generic-2D | 9 | 0.481 | [0.037, 0.963] | 0.133 | nan | no | secondary (exploratory, unadjusted) |
| b_distribution | KB-3D vs KB-2D | 9 | 0.074 | [-0.407, 0.519] | 0.844 | nan | no | secondary (exploratory, unadjusted) |
| b_distribution | KB-3D vs generic-3D | 9 | 1.037 | [0.747, 1.395] | 0.004 | nan | no | secondary (exploratory, unadjusted) |
| b_distribution | KB-2D vs generic-2D | 9 | 0.259 | [-0.407, 0.926] | 0.547 | nan | no | secondary (exploratory, unadjusted) |
| c_geometry | KB-3D vs KB-2D | 9 | -0.111 | [-0.506, 0.297] | 0.641 | nan | no | secondary (exploratory, unadjusted) |
| c_geometry | KB-3D vs generic-3D | 9 | 0.852 | [0.420, 1.253] | 0.012 | nan | no | secondary (exploratory, unadjusted) |
| c_geometry | KB-2D vs generic-2D | 9 | 0.630 | [0.259, 0.927] | 0.023 | nan | no | secondary (exploratory, unadjusted) |
| d_surface | KB-3D vs KB-2D | 9 | 0.099 | [-0.198, 0.506] | 0.719 | nan | no | secondary (exploratory, unadjusted) |
| d_surface | KB-3D vs generic-3D | 9 | 0.673 | [0.475, 0.908] | 0.004 | nan | no | secondary (exploratory, unadjusted) |
| d_surface | KB-2D vs generic-2D | 9 | 0.519 | [0.111, 0.852] | 0.055 | nan | no | secondary (exploratory, unadjusted) |
| e_artefacts | KB-3D vs KB-2D | 9 | -0.185 | [-0.580, 0.173] | 0.449 | nan | no | secondary (exploratory, unadjusted) |
| e_artefacts | KB-3D vs generic-3D | 9 | 0.444 | [0.123, 0.778] | 0.047 | nan | no | secondary (exploratory, unadjusted) |
| e_artefacts | KB-2D vs generic-2D | 9 | 0.741 | [0.296, 1.185] | 0.016 | nan | no | secondary (exploratory, unadjusted) |
| f_overall_accept | KB-3D vs KB-2D | 9 | 0.049 | [-0.321, 0.432] | 0.867 | 1.000 | no | primary (confirmatory) |
| f_overall_accept | KB-3D vs generic-3D | 9 | 0.346 | [0.105, 0.661] | 0.020 | 0.059 | no | primary (confirmatory) |
| f_overall_accept | KB-2D vs generic-2D | 9 | 0.148 | [-0.185, 0.407] | 0.531 | 1.000 | no | primary (confirmatory) |

Per-method means are in per_method_scores.csv (specimen-clustered bootstrap CIs).

**Data status: SIMULATED RATINGS - NOT EXPERT DATA. No real expert ratings exist yet (as of 2026-10-14).**
