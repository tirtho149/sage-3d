# Generator ablation results (Stage 0R, 2026-10-16)

Setting: 20 classes, the same 60 parent assets (59 root photos) for every arm, 1,920 images per arm (32/parent),
T0 frozen DINOv2 ViT-S/14 + logistic regression, strict S0 (no real label used for any choice).
Evaluation: **DEV** (Stage-0 test, 4,202 photos; exploratory). FINAL not yet evaluated.
Paired differences on the same test photos, pooled over 3 matched replicates (edit seed + nuisance draw), group
bootstrap 95% CI. Files: `results/stage1/stage0r/*.json|npz`, `contrasts_contrasts_stage0r_dev.csv`,
`interaction_*.json`; cross-class: `results/stage1/cross/`.

## Arm scores (DEV macro-F1; mean of 3 replicates)
| arm | texture | geometry | DEV | synthetic LOPO |
|---|---|---|---|---|
| PHOTO2D | real photo | – | 0.648 | 0.63 |
| GEN2D | generic IP2P | – | 0.503 | 0.36 |
| KB2D | KB IP2P | – | 0.397 | 0.53 |
| U3D | SAM 3D | – | 0.527 | 0.51 |
| HARM3D | SAM 3D, colour-matched | – | 0.527 | 0.51 |
| TEX3D | photo lifted | – | 0.590 | 0.51 |
| GEN3D | generic edit lifted | – | 0.477 | 0.34 |
| KBAPP3D | KB edit lifted | – | 0.370 | 0.51 |
| GEO3D | SAM 3D | KB operators | 0.510 | 0.51 |
| KB3D | KB edit lifted | KB operators | 0.374 | 0.53 |
| TEXGEO3D (= Stage-0 "KB3D") | photo lifted | KB operators | 0.575 | 0.51 |
| MIX25 / 50 / 75 (KB3D share) | | | 0.394 / 0.394 / 0.389 | |

## Primary contrasts (Holm over P1–P3; P4 = fine-tuning, see CLASSIFIER_FINE_TUNING_STUDY.md)
| | Δ macro-F1 | 95% CI | Holm p |
|---|---|---|---|
| P1 KB3D − U3D | **−0.153** | [−0.171, −0.136] | < 0.001 |
| P2 KB3D − KB2D | −0.023 | [−0.032, −0.012] | < 0.001 |
| P3 MIX50 − KB2D | −0.003 | [−0.010, +0.004] | 0.38 |

## Decomposition (secondary, not multiplicity-controlled)
| effect | contrast | Δ | 95% CI |
|---|---|---|---|
| photo texture on the asset | TEX3D − U3D | +0.062 | [+0.047, +0.077] |
| colour harmonisation | HARM3D − U3D | 0.000 | [−0.007, +0.007] |
| generic edit, 3D | GEN3D − U3D | −0.050 | [−0.067, −0.034] |
| knowledge vs generic, 3D | KBAPP3D − GEN3D | −0.107 | [−0.119, −0.096] |
| knowledge vs generic, 2D | KB2D − GEN2D | −0.106 | [−0.121, −0.091] |
| generic edit, 2D | GEN2D − PHOTO2D | −0.145 | [−0.159, −0.131] |
| KB edit, 2D | KB2D − PHOTO2D | −0.251 | [−0.267, −0.236] |
| residual render gap | TEX3D − PHOTO2D | −0.058 | [−0.071, −0.046] |
| geometry operators, all classes | GEO3D − U3D | −0.017 | [−0.024, −0.011] |

## Symptom modality × generator (4 geometry-operator classes vs 16 appearance classes)
Geometry cohort = peach leaf curl (distort/pucker/thicken), corn smut (outgrowth), mango anthracnose (depress),
citrus greening (thicken). Macro-F1 within each cohort (20-way head).
| contrast | geometry cohort | appearance cohort | interaction (geo − app) |
|---|---|---|---|
| U3D − PHOTO2D | +0.009 [−0.023, +0.038] | −0.149 [−0.165, −0.132] | **+0.158 [+0.123, +0.192]** |
| KB3D − KB2D | +0.042 [+0.009, +0.075] | −0.043 [−0.053, −0.031] | **+0.085 [+0.052, +0.120]** |
| KB3D − KBAPP3D (ops on KB colour) | +0.027 [+0.001, +0.051] | −0.002 | +0.029 [+0.005, +0.055] |
| GEO3D − U3D | −0.018 [−0.044, +0.007] | – | – |
| TEXGEO3D − TEX3D | −0.017 [−0.040, +0.005] | – | – |
- [M] 3D renders are as informative as the photos for shape-symptom classes and much worse for appearance classes.
- [M] KB geometry operators help only together with the KB appearance edit and only in the geometry cohort (+0.027);
  on their own (GEO3D) or with photo texture they are neutral to slightly negative.
- [I/H] Candidate explanation: shape symptoms (curl, galls, sunken lesions) survive the coarse splat, fine lesion
  texture does not. Threats: only 4 classes, 3 of them SAGE-Extend pilot assets (better pose fitting), organ
  differences (corn ear). **Needs replication on the 13 eligible geometry classes before any claim** (compute gate G1/G2).

## Knowledge-only test (cross-class parents, 16 classes; replicate 0)
Parent = a photo of the *sibling* disease of the same crop and organ; label = target class; KB arms apply the target
class's KB prompt (and KB operators in 3D). Within-crop discrimination on DEV (chance 0.5):
| arm | within-crop accuracy [95% CI] |
|---|---|
| XPHOTO2D (relabelled sibling photo) | 0.209 [0.189, 0.230] |
| XGEN2D (generic edit) | 0.261 [0.239, 0.281] |
| **XKB2D (target KB edit)** | **0.438 [0.418, 0.460]** |
| XKBAPP3D | 0.392 [0.377, 0.410] |
| XKB3D | 0.399 [0.383, 0.416] |
| reference: PHOTO2D / KB2D / U3D (own parents) | 0.791 / 0.670 / 0.717 |
- [M] The KB edit moves within-crop discrimination by +0.23 (2D) and +0.19 (3D) relative to the relabelled photo:
  **the source-grounded text carries transferable disease information**. It does not overcome the parent's real
  symptoms (all X arms stay below 0.5), and 3D lifting loses part of it (0.39–0.40 vs 0.44).

## Mixing (fixed 32 images/parent)
KB3D share 0/25/50/75/100 %: 0.397 / 0.394 / 0.394 / 0.389 / 0.374 (pooled). No complementarity: 3D dilutes the
(slightly) more transferable KB2D distribution; the geometry-restricted mixture is pending (needs the geometry
cohort at scale). No mixture was selected on any real test.

## What this means
- [M] With InstructPix2Pix as the 2D editor, "knowledge-guided" edits of already-diseased photos make the training
  data worse; the Stage-0 claim "KB edits help 3D" was produced by the mask bug and is withdrawn.
- [M] The text itself is informative (knowledge-only test) — the bottleneck is the editor's fidelity, not the KB.
- [M/I] Geometry is justified, if anywhere, for shape-symptom classes; this is the one condition where 3D ≥ 2D.
