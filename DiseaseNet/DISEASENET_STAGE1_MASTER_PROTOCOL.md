# DiseaseNet Stage 1 — master protocol

Status: **DRAFT v1, 2026-10-16 01:30, written before any Stage-1 result.** Sections marked FROZEN may not change
after the first Stage-1 DEV result; the rest is frozen (v2) before the first FINAL evaluation. Every change is
appended to the change log at the end with a reason.
Brief: `reports/stage1/TASK_stage1_master_prompt.md`. Audit that motivates it: `reports/DN01_stage0_audit.md`.

## 0. Ultimate question
Can source-grounded synthetic plant-disease generation give a training distribution whose discriminative features
transfer to real crop–disease recognition? If knowledge-grounded 3D does not beat 2D, under which measured
conditions (if any) is geometry worth its cost?

## 1. Data and evaluation (FROZEN)
- Real data: locked partitions (`DATA_PROVENANCE_AUDIT.md`). Parents only from `parent`; shots and S1 validation
  only from `adapt` (split 50/50 by near-duplicate group, seed 20261016); FINAL test = `val` partition, cap 300/class,
  photos with >1 label in the cohort removed.
- DEV = the Stage-0 test (4,202). Used for diagnosis, ablations and design choices. Never used for confirmatory claims.
- FINAL: evaluated **once per frozen configuration**, through `src/s1/harness.final_set` (access-logged). No
  configuration, mixture, hyper-parameter or class list may be chosen by FINAL results.
- Unit of resampling: near-duplicate group (real); parent asset (synthetic IID). 2,000 bootstrap resamples.
- Primary metric: real macro-F1 (classes present in the head; closed set). Secondary: top-1, balanced accuracy,
  top-5, per-class recall, ECE (15 bins), crop-level accuracy.

## 2. Protocols (FROZEN)
- **Strict S0**: synthetic training data only; every selection (hyper-parameters, early stopping, mixtures) uses
  synthetic leave-one-parent-out (LOPO) validation only. Pretrained backbones are allowed (generic real images, not
  SAGE) and stated.
- **Adapted S1**: S0 model + k ∈ {0, 1, 4, 8, 16} real shots/class from `adapt`-shots; selection on `adapt`-val.
  Baseline with identical shots and budget: the same backbone without synthetic pre-training (generic-pretrained).
- Classifier ladder: T0 frozen DINOv2 ViT-S/14 + logistic regression (Stage-0 T0, kept); T1 frozen + linear head
  trained with SGD and augmentation; T2 last 1–2 blocks unfrozen; T3 LoRA (rank 8, qkv) ; T4 full fine-tune
  (Stage-0 "T3" is T4 with a fixed schedule; kept). T5 larger ViT only after T0–T4 are complete.

## 3. Generators (arms) (FROZEN)
Same parent assets for every arm; nuisance randomness (2D variant, background plate, photometrics) seeded
independently of the arm, so arms from one parent differ only in texture source and geometry.

| arm | texture | geometry | role |
|---|---|---|---|
| PHOTO2D | real photo | – | few-shot real reference |
| GEN2D | generic IP2P edit | – | generic 2D synthetic |
| KB2D | KB IP2P edit | – | knowledge 2D synthetic |
| U3D | SAM 3D colours | – | unedited 3D |
| TEX3D | photo lifted onto the asset | – | appearance-preserving 3D (texture projection) |
| GEN3D | generic edit lifted | – | generic 3D appearance edit |
| KBAPP3D | KB edit lifted | – | KB appearance-only 3D |
| GEO3D | SAM 3D colours | KB operators | KB geometry-only 3D |
| KB3D | KB edit lifted | KB operators | full KB 3D |
| TEXGEO3D | photo lifted | KB operators | = Stage-0 "KB3D" (bug), kept for continuity |
| PLATE2D | background only (organ inpainted) | – | shortcut control |
| MIX | KB2D and KB3D at 100/75/50/25/0 % (fixed total) | | complementarity |

Geometry operators exist for few classes (Stage-0: 4/20; eligible pool: 13/87). For classes without one, GEO3D = U3D
and KB3D = KBAPP3D by construction; geometry effects are estimated **only** in the geometry cohort.

## 4. Primary comparisons (FROZEN; Holm-adjusted over these four)
P1 KB3D vs U3D · P2 KB3D vs KB2D · P3 (KB2D+KB3D 50/50) vs KB2D · P4 T2 (last-block FT) vs T0, on KB3D and KB2D.
Primary protocol: strict S0, DEV for development, FINAL for the confirmatory run. Effect = paired macro-F1
difference (same test images), group bootstrap 95% CI and two-sided bootstrap p, per seed and pooled over seeds.

## 5. Secondary / diagnostic analyses (not multiplicity-controlled; reported as exploratory)
- Decomposition of 3D effects: TEX3D−U3D (texture), KBAPP3D−GEN3D (knowledge in 3D appearance), KB2D−GEN2D
  (knowledge in 2D), GEO3D−U3D and KB3D−KBAPP3D (geometry; geometry cohort only).
- Background: R0 (plain), R1 (source-like own background), R2 (natural agricultural plates of other classes),
  R5 (R2 + photometric randomisation) at fixed image count; PLATE2D shortcut.
- Viewpoint: input view only, NARROW (≤ 8°), V8 (Stage-0, ≤ 30°), WIDE (to 180°) at fixed count.
- Feature preservation: input-view render vs photo (DINOv2 cosine, LPIPS, high-frequency energy ratio inside the
  mask, symptom-region visibility); class separability of synthetic features. Perceptual similarity is not
  treated as biological correctness.
- Harmonisation: SAM 3D colours vs photo projection vs colour transfer (render matched to the parent photo's
  Lab statistics inside the mask) vs KB2D; record whether symptoms survive.
- Parent diversity at fixed images/class: {3 parents × 32, 6 × 16, 12 × 8} (2D arms first; 3D after the compute gate).
- Modality cohorts: geometry-sensitive (implemented KB operator) vs appearance-sensitive classes; modality × generator.
- Open set: 19 reserved classes (+ classes beyond the head in nested cohorts); MSP, energy, AUROC, FPR@95TPR,
  risk–coverage. Photos sharing groups with known classes excluded.

## 6. Gates (compute)
- G0 (now): Stage 0R on existing 60 assets (no new reconstructions).
- G1 new SAM 3D reconstructions for parent-diversity scaling (≈ 18 min/asset on one GPU): requested from the user
  with a measured estimate, only if Stage 0R shows a 3D arm within the noise floor of 2D on DEV **or** a
  geometry-cohort effect worth testing. Otherwise 3D scaling stops and the 2D/photo diversity curve is reported.
- G2 50-class expansion, G3 87-class (maximum eligible; the brief's 100/250 tiers are not reachable — only
  87 classes satisfy the evaluation rules, `CLASS_ELIGIBILITY.csv`).

## 7. Reporting rules
Each report separates measured / pilot / inference / hypothesis / planned / blocker. Negative results are kept.
Every number is recomputed from saved per-image predictions by a script in the repo.

## Change log
- v1 2026-10-16 01:30 — initial draft (before any Stage-1 result).
