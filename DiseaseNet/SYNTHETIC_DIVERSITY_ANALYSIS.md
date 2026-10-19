# Synthetic diversity analysis (Stage 0 data; 2026-10-16)

Script: `scripts/s1_diversity.py` → `results/stage1/diversity/{per_class.csv, summary.json}`,
`figures/stage1/diversity_images_vs_parents.{pdf,png}`. Features: frozen DINOv2 ViT-S/14 CLS (L2-normalised).

## Measured

Per class and arm (Stage-0 data as generated):

| quantity | value (every class) |
|---|---|
| independent root photos | 3 (59 distinct overall: one photo is shared by the two coffee classes) |
| independent 3D assets | 3 |
| accepted geometry edits | 3 in peach leaf curl, corn smut, mango anthracnose, citrus greening; 0 elsewhere |
| accepted appearance edits | **0** (edit bug: every 2D "edit" equals the photo) |
| camera views / 2D variants | 8 (yaw/pitch ≤ 30°) / 8 seeded affine variants |
| rendering conditions | 4 (white, own background, other-class background, + photometric) |
| backgrounds | own plate + plates from up to 8 other parents |
| disease stages | 1 (the KB stage selected per class) |
| technically rejected renders | 0 recorded (renders are not QC'd; parents had 13 visual rejects) |
| biologically ambiguous renders | not assessed — needs expert raters **[BLOCKER]** |
| images per class and arm | 96 |

Embedding statistics, mean over the 20 classes (an equal-size random sample of real DEV photos as reference):

| arm | spread (1 − mean cos) | near-duplicate pairs (cos ≥ 0.95) | clusters (avg-link, d ≤ 0.10) | effective parents | variance share between parents |
|---|---|---|---|---|---|
| PHOTO | 0.406 | 2.3 % | 39.9 | 2.86 | 0.49 |
| KB2D (= photo) | 0.405 | 2.4 % | 38.6 | 2.86 | 0.49 |
| U3D | 0.459 | 1.7 % | 47.8 | 2.90 | 0.46 |
| KB3D (= photo-textured 3D) | 0.464 | 1.7 % | 50.0 | 2.87 | 0.43 |
| real DEV sample | 0.431 | 0.3 % | 69.8 | 70.7 | 1.00 (every photo independent) |

- **[MEASURED]** Nominal diversity looks real: embedding spread of the synthetic sets equals or exceeds the real
  sample's, and near-duplicate rates are low (no class × arm has > 50 % near-identical pairs).
- **[MEASURED]** But half of that spread (0.51–0.57) is **within-parent nuisance variation** (views, backgrounds,
  photometrics); the effective number of independent parents is ~2.9 per class versus ~71 in an equal-size real
  sample. 96 images per class carry the information of 3 photographs.
- **[MEASURED]** Parents used vs available: 3 per class against 37–3,817 independent parent groups in the
  `parent` partition (median ≈ 1,190). Every class can grow to ≥ 12 independent parents without touching evaluation data.

## Inference
Stage 0 multiplied renders, not objects. Any generator comparison at 3 parents/class is dominated by which 3
photographs were drawn; the parent-diversity scaling experiment (fixed images/class, 3 vs 6 vs 12 parents) is
therefore the primary Stage-1 experiment.

## Coverage-expansion strategy (planned)
1. Add independent parents first: 9 more per Stage-0 class (12 total) from `parent`, one per near-duplicate group,
   round-robin over source datasets, same mask QC, no multi-label photos (`select_parents --stage1`, running).
2. Hold images/class fixed (96): 3 × 32, 6 × 16, 12 × 8 (nested parent sets, 3 random parent draws).
3. 2D arms (PHOTO2D, KB2D) first — masks + edits only. 3D needs ≈ 18 GPU-min per new asset (180 assets ≈ 54 GPU-h):
   compute gate G1, requested only if the 2D curve shows that more parents help.
4. Only then multiply views/conditions — Stage-1 viewpoint ablation shows no gain from more or wider views.
