# Data provenance audit (2026-10-16)

[MEASURED] unless marked otherwise. Sources: `data/real_splits/{LOCK.json, real_splits.parquet, stage0_test.csv}`,
`data/synthetic/stage0/{parents.csv, manifest_*.csv, verified_inputs.json}`, `src/generation/registry.py`.

## Real splits (locked 2026-10-15, seed 20261015)

| partition | images (unique sha1) | role in Stage 1 |
|---|---|---|
| parent | 155,651 | source of parent photos for generation (only) |
| adapt | 59,005 | S1 few-shot shots + S1 real validation (split by group) |
| val | 58,092 | **Stage-1 FINAL test** (never used so far) |
| test | 116,946 | Stage-0 test cache (4,202) = DEV; the rest unused |
| openset | 24,523 | 19 held-out classes, open-set evaluation only |
| excluded_openset_overlap | 11,972 | near-duplicates of open-set images in known classes; never used |

- Unit of splitting = near-duplicate group (pHash Hamming ≤ 6, union-find; 235,142 groups). Among known-class
  partitions every group lies in exactly one partition. 433 groups span an open-set partition and a known one
  (222 openset/excluded, 152 openset/test, 59 openset/val): known-vs-open-set near duplicates. 16 Stage-0 test
  photos are in such groups → they must be excluded from open-set (unknown-class) analyses.
- Label conflicts: 1,840 photos carry >1 class label in SAGE; 13 within the Stage-0 classes; 1 in the Stage-0 test.
  Stage-1 extractions drop photos with >1 label inside the evaluated cohort (`scripts/s1_extract_real.py`).
- Why a new final test: after removing groups already used in the Stage-0 test, 9 of the 20 Stage-0 classes have
  **zero** unused test-partition photos (citrus canker, greening, coffee ×2, corn smut, grape downy mildew,
  peach leaf curl, potato TSWV; coffee brown eye spot has 1). The `val` partition is untouched and covers all 20
  (8–2,744 photos/class); it becomes FINAL, capped at 300/class, group bootstrap.

## Parents and synthetic data

- 60 Stage-0 assets: 45 new (3 per class for 15 classes; SAM ViT-H centred-box masks, automatic QC + visual QC,
  13 visual rejects logged) + 15 SAGE-Extend pilot assets (read-only). **59 distinct root photos**: sha1 338c753b…
  is labelled both `coffee::berry_blotch` and `coffee::brown_eye_spot` in SAGE and is a parent of both.
- Every root photo is in the `parent` partition (all 74 matching SAGE rows, pilot included); no parent's group
  occurs in the Stage-0 test.
- Every synthetic image row records arm, class, asset, root sha1, view, condition and background asset. The R2/R5
  background plates come from parents of *other* classes (class-independent); R1 uses the parent's own background.
- Model checkpoints: none saved in Stage 0 (fixed schedule, last epoch). Per-image predictions are saved for all runs.
- **Generation bug** (see `reports/DN01_stage0_audit.md`): Stage-0 2D edits are identical to the photos. Stage-0
  files are kept unchanged as the audited record; Stage 1 writes to `data/synthetic/stage1/`.

## Provenance guarantees in Stage 1

- `src/provenance/leakage.Guard` verifies the lock hash before any real extraction.
- FINAL-test reads only through `src/s1/harness.final_set`, which appends to `results/stage1/FINAL_ACCESS_LOG.csv`.
- Class lists for every cohort come from `CLASS_ELIGIBILITY.csv`, built from audit + partition counts only
  (no model results).
