# 03 — Geometry-aware editing method (brief v2 §5, §11)

- **Code:** `src/geometry_editor/deform.py`, `src/metrics/{geom_attributes,calibrate_attributes,kb_grounding,gaussian_diff,view_metrics}.py`
- **Tests:** `tests/test_geometry_ops.py` (5 passing)
- **Configs:** `configs/geometry/operator_defaults.json`, `configs/evaluation/attribute_thresholds.json`

## 1. Why a separate geometry editor
GaussianEditor's objective contains no geometric target (report 02). Arm E moves Gaussians only incoherently: coherence 0.31 vs 0.25 for noise, with a doubled sheet thickness. The simplest method that can produce a measurable, KB-specified shape change is an explicit, regularised displacement field on the Gaussians (brief method **G2**). Mesh-guided deformation (**G3**) was not needed: the SAM 3D splat is dense (58k–561k Gaussians) and locally sheet- or volume-like, so a k-NN graph gives enough local correspondence for ARAP and Laplacian terms.

**Hybrid G4:** geometry edit first, then GaussianEditor appearance editing with geometry frozen. These are arms H (KB) and K (appearance only).

## 2. Formulation
G = {μ_i, R_i, S_i, α_i, c_i}. Edited positions are μ_i' = μ_i + m_i·d_i, where m_i ∈ [0,1] is the editable-region weight (whole organ by default; RBF-local when centres are given). d is obtained by minimising, with Adam (300 iterations, lr = 1e-3·L):

| Term | Definition | Weight |
|---|---|---|
| L_edit | mean_i m_i (d_i·n_i − h_i)² / L² — match the target **normal** offset field h (KB arms). Generic arm G: mean_i m_i ‖d_i − g_i‖² / L² | 1 |
| L_arap | mean over 8-NN edges of ((‖e'_ij‖ − ‖e_ij‖)/‖e_ij‖)² — edge-length preservation | 0.1 |
| L_smooth | mean over 8-NN edges of ‖m_i d_i − m_j d_j‖² / L² — Laplacian smoothness | 0.01 |
| L_anchor | mean_i (1 − m_i)‖d_i‖² / L² — keep non-target parts | 1 |

- **L** is the bbox diagonal. All quantities are unit-free; nothing is a physical length.
- **Normals n_i:** local PCA over 16-NN. Orientation comes from the global PCA normal plus neighbour majority voting (sheets), or outward from the centroid (volumes). Three passes of 32-NN averaging stabilise the leaf margins.
- **Rotations:** each Gaussian gets the minimal rotation from its old to its new normal, the latter re-estimated on the deformed positions with the same neighbourhoods.
- **Scales:** unchanged except for `thicken`. Colours and opacities are never changed by this stage.

## 3. KB operators → measurable target h

| KB term (verbatim, matched by regex) | Operator | Target field | Default (fraction of L) |
|---|---|---|---|
| puckered, blister | `pucker` | zero-mean sum of RBF bumps; Poisson-disk centres ≥ 3σ apart | σ 0.035, amp 0.02 |
| distorted, deformed, malformed | `distort` | two broad RBFs of opposite sign | σ 0.3, amp 0.06 |
| thickened | `thicken` | no displacement; Gaussian scale along n ×2 (only where the normal axis is unambiguous, \|cos\| > 0.8) | factor 2 |
| gall, tumor-like, swollen | `outgrowth` | max of positive RBFs at n sites | σ 0.08, amp 0.08, n 3 |
| crater, depression, sunken | `depress` | max of negative RBFs at n sites | σ 0.012, amp 0.006, n 12 |
| curl, roll, cup · wither, wilt · corky · raised, warty | none (`unsupported_operator`) | recorded, never approximated | — |

**Severity** is not graded from text (one moderate setting per operator). **Placement** of bumps and sites is random (seeded); it is *not* tied to observed lesions in the photo.

## 4. Design history (deviations, logged)
1. **First pucker operator** (σ 0.045 → 0.07 L, farthest-point centres at spacing 2σ): the synthetic-sheet test showed that ~50 overlapping bumps sum to a near-uniform lift, not blisters. It was redesigned to Poisson-disk centres ≥ 3σ apart and a zero-mean field, and all specimen outputs were regenerated.
2. **Margin artefacts** on Peach 0087 (ragged edges from unreliable margin normals) were fixed by normal smoothing and by restricting thickening to well-defined normal axes. This was decided from one visual check on 0087, before any metric was computed.
3. **Detector bugs** found by the tests: voxel-grid padding (half the samples read as 0), and thickness measured along the stale normal.
4. **Pucker detector:** the absolute blister count was replaced by blister density per (0.1 L)². This was decided **after** seeing that J on Peach 0087 had 14–19 blisters, below a count threshold calibrated on a larger synthetic sheet. Scale invariance is verified by a test (half-area sheet), and results are reported under both rules.

## 5. Attribute detectors and calibration
Out-of-plane displacement h_i = d_i·n_i (correspondence via `gaussian_diff.match`) is band-split with normalised Gaussian smoothing S_σ on a voxel grid:

| Attribute | Score |
|---|---|
| distorted | RMS(S_0.12L[h]) / L |
| puckered | blister density (local maxima of S_0.02L[h] − S_0.12L[h] above max(0.3·p99, 0.003 L), per (0.1 L)²) |
| thickened | median ratio of Gaussian extent along each state's own normal |
| outgrowth | p99.5(S_0.03L[h] − S_0.3L[h]) / L |
| depression | −p0.5(S_0.005L[h] − S_0.04L[h]) / L |
| jitter (QC) | RMS(h − S_0.02L[h]) / L |

**Threshold rule** (uniform, synthetic flat sheet only: `src/metrics/calibrate_attributes.py`): threshold_k = midpoint between operator k's own response and the largest off-target response. A detector is **specific** only if its own response exceeds every off-target response.

| Attribute | Own response | Max off-target | Threshold | Specific |
|---|---|---|---|---|
| distorted | 0.0286 | 0.0116 | 0.0201 | yes |
| puckered (density) | 4.75 | 0.36 | 2.55 | yes |
| thickened | 2.00 | 1.00 | 1.50 | yes |
| outgrowth | 0.0556 | 0.0259 | 0.0407 | yes |
| depression | 0.0028 | 0.0097 | 0.0063 | **no** (blister troughs and gall flanks dip more than "point-sized" pits). Excluded from binary recall; raw score reported |

After calibration, the operator × detector matrix is diagonal for the four specific detectors (`results/geometry_calibration.json`).

## 6. Peach 0087 diagnostic (Phase A)
KB stage 0 requires {distorted, puckered, thickened}. Seeds 0, 1, 2. Recall is computed on specific detectors only.

| Arm | distorted | blister density (count) | thickened | outgrowth | jitter | coherence | geometry recall | unsupported |
|---|---|---|---|---|---|---|---|---|
| E (GE generic) | 0.0001 | 0.00 (0) | ×1.04 | 0.001 | 0.0078 | 0.31 | 0 | — |
| J seed 0 | 0.0293 | 3.51 (19) | ×1.87 | 0.028 | 0.0078 | 0.997 | **1.00** | 0 |
| J seed 1 | 0.0224 | 2.52 (14) | ×1.87 | 0.025 | 0.0066 | — | 0.67 | 0 |
| J seed 2 | 0.0257 | 2.93 (16) | ×1.87 | 0.036 | 0.0072 | — | **1.00** | 0 |
| G seed 0 (generic, RMS-matched) | 0.0204 | 0.70 (4) | ×1.00 | 0.046 | 0.0070 | 0.999 | 0.33 | 0.50 (outgrowth) |
| G seed 1 | 0.0184 | 1.43 (8) | ×1.00 | 0.028 | 0.0112 | — | 0 | 0 |
| G seed 2 | 0.0173 | 2.20 (12) | ×1.00 | 0.016 | 0.0161 | — | 0 | 0 |

- J recall: mean **0.89**. Under the original count rule, pucker is never detected, so J recall is 0.67 per seed.
- G recall: mean **0.11**.

**Render space (18 views vs D):**

| Arm | Silhouette change | Depth change | Grain | Edge fuzz |
|---|---|---|---|---|
| J | 0.26–0.28 | 0.11–0.16 | ×1.24 | ×1.37 |
| G | 0.24–0.27 | 0.07–0.09 | ×1.0–1.2 | ×1.0–1.1 |
| E | 0.14 | 0.03 | ×13.6 | ×1.9 |

**Regularity (J seed 0, deform report):**
- realised / target normal RMS = 0.038 / 0.043 L (fidelity r = 0.98);
- edge stretch: median 2.7 %, p95 19 %;
- normal change: median 13°, p95 38°.

**Caveats.**
- J is *built* from the KB operators, so high recall partly holds by construction. What it shows is that the specified geometry survives the regularisers on a real asset, and that it is distinguishable from an equal-magnitude generic warp (G).
- It does **not** show biological correctness. No expert has judged whether these blisters look like *Taphrina* puckering.
- Blister placement is random, not at observed symptoms.

## 7. Limitations
- **Single-sheet leaves:** SAM 3D leaves are one Gaussian sheet (SH degree 0). Upper and lower surfaces cannot carry different colours, which matters for Tomato Leaf Mold, where the mold is on the lower surface (report 06).
- **Pucker at real scale:** "puckered" on a real *Taphrina* leaf may be finer or coarser than σ = 0.035 L. That is not validated.
- **Unsupported terms:** curl and roll have no operator, by design, since their direction is not specified in the cited text.
