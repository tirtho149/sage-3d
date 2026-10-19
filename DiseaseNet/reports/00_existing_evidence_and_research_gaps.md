# 00 — Existing evidence from SAGE-3D and the gaps DiseaseNet must close

Sources: SAGE-Extend repository (`~/Desktop/Research/external/SAGE-Extend`), reports 00–17,
deck v5 (Overleaf 6ac80070), review response (`REVIEW_TODO.md`, `reports/17`).

## What SAGE-3D established (measured)

| Finding | Evidence | Consequence for DiseaseNet |
|---|---|---|
| 15 SAM 3D assets, 5 classes, 3 per class; reconstruction runs locally (17–26 min, 11.6 GB) | reports/01, 08 | Stage 0 needs ~45 more assets; asset count, not views, is the scaling variable |
| Pose fix lifts *fitted* input-view IoU 0.63 → 0.92; 3D accuracy unverified; 1/15 topology failure (ring) | reports/01, review A5 | QC must reject ring/closed topologies before rendering |
| KB geometry operators (curl, pucker, galls, two surfaces) follow commands; curl direction holds on 11/11 leaves but magnitude varies 7–43° and fails on 2 Citrus leaves | review B1 | Operator ranges need per-shape QC; Citrus geometry edits are risky |
| Every curl contradicts its photo (0/22 in preserve mode); KB-sized galls do too unless render-checked | review B2/B3 | Generator must use the evidence policy or label edits counterfactual |
| Cross-view consistency comes from sharing one asset (2D edit painted on the asset: dE 2.7 vs floor 2.0) | review B6 | Do not use consistency as evidence of classifier benefit (protocol J5) |
| 3D adds direction (2D warp dE 5–14 vs 3D 41–46) and partly surface identity (two-layer 2D wrong on 0–23% of pixels) | review B4/B7 | Geometry may matter for curl/gall classes; for flat leaves expect little |
| **Minimal downstream test: KB-3D renders added to real few-shot training did not help** (macro-F1 −0.025 / −0.033 / −0.003 at k = 1/4/16) | reports/14 | Starting challenge; the new question is synthetic-supervised training, not augmentation |
| Renders are trivially separable from photos (AUC ≈ 1.00; white background 0.99); renders shift the class boundary (pilot recall −21 points at k = 4) | review A12 | Domain gap is the first obstacle |
| Matched gap decomposition: object rendering at photo pose +0.28, new viewpoints +0.21, background +0.09, recolouring −0.01 | review B8 | Prioritise near-photo views, real-background compositing, render fidelity |
| KB-recall comparisons cannot reach significance with 6 specimens | review B5 | Power must come from classes and independent assets |

## Gaps DiseaseNet must close

1. **Asset diversity.** 3 assets/class is a debugging scale; J2 (fixed images, varying assets)
   is the decisive ablation.
2. **Domain gap before scale.** No evidence yet that any render style transfers; the first
   real-domain result decides whether to scale.
3. **Training regime.** Only frozen linear probes and augmentation were tested; synthetic-
   supervised fine-tuning (T0–T3) is untested.
4. **Coverage.** Generation has only been demonstrated on 5 classes; the audit allows 195.
5. **Provenance.** Real-derived assets require provenance-aware splits (now locked:
   `IID_OOD_SPLIT_PROTOCOL.md`).
6. **Biological validity** of edits is still unmeasured (expert kit ready, raters not recruited).
