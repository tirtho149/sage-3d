# DN01 — Stage 0 audit (2026-10-16)

Scope: everything Stage 0 produced (results/E0, manifests, splits, parents, generated images, logs). Labels used
below: **[MEASURED]** recomputed from files in this audit; **[OBSERVED]** visual/pilot observation;
**[INFERENCE]** follows from measured facts; **[BLOCKER]** cannot be resolved locally.
Companion files: `STAGE0_VERIFICATION.md` (numbers), `DATA_PROVENANCE_AUDIT.md` (splits/provenance),
`SYNTHETIC_DIVERSITY_ANALYSIS.md` (diversity).

## 1. Headline: a pipeline bug changes what Stage 0 measured

**[MEASURED]** `src/generation/edit2d.py` passed a boolean mask to SAGE-Extend's `ip2p_core.edit`, which computes
`mask > 127`. For a boolean array that is all-False, so the composite weight inside the organ was 0 and every
"edited" image was written back as the **unedited photo**. Evidence:

| check | result |
|---|---|
| `kb_s0.png` vs `generic_s0.png`, all 60 assets | byte-identical, 60/60 |
| CIEDE76 photo -> KB edit, inside organ mask, 45 new assets | 0.000 (max 0.000) |
| same, outside mask | 0.000 |
| same code with a uint8 mask (fixed), first 30 edits | mean RGB change in mask 12-58 |

`results/stage1/audit/edit_magnitude.csv`. SAGE-Extend's own callers pass uint8 masks, so its 2D edits (E2/E4/E5 slides) are unaffected.

**[INFERENCE] What each Stage-0 arm actually was:**

| Stage-0 label | intended | actually |
|---|---|---|
| PHOTO | 3 real parents/class + 2D variants | as intended |
| KB2D | KB InstructPix2Pix edit | **the real photo** (different variant/background seeds than PHOTO) |
| GEN2D | generic edit | **the real photo** (again different seeds) |
| U3D | unedited SAM 3D render | as intended |
| KB3D | KB geometry + KB colour lifted | **KB geometry (4 classes) + the source photo's colours re-projected onto the visible Gaussians** |

**[MEASURED]** Re-rendering with the Stage-1 generator reproduces Stage-0 "KB3D" pixel-for-pixel as
`TEXGEO3D` (photo lift + geometry) — mean |diff| 0.0 on R0/R1 renders.

Consequences for the reported Stage-0 contrasts (the numbers are correct; their meaning is not what the deck says):

| contrast | T0 | T3 (3 seeds) | what it really compares |
|---|---|---|---|
| KB3D − U3D | +0.057 | +0.052 | photo-texture re-projection (+ geometry in 4 classes) vs SAM 3D colours. **Not knowledge.** |
| KB3D − PHOTO | −0.071 | −0.066 | photo-textured 3D renders vs real photos |
| KB2D − KB3D | +0.082 | +0.061 | real photos vs photo-textured 3D renders |
| KB2D − PHOTO | +0.010 | −0.005 | real photos vs real photos (augmentation seeds only) |
| GEN2D − KB2D | −0.003 | +0.000 | real photos vs real photos |

The last two rows are an **empirical noise floor** for the 5-arm design: identical training photos with different
2D variant/background draws move macro-F1 by up to ±0.010 (T0) and ±0.017 (T3 single seed).
The Stage-0 deck statements "KB edits help 3D renders" and "KB vs generic 2D prompt is a wash" are unsupported and
are being corrected (Stage 0R reruns all arms with fixed edits).

## 2. Verification checklist (brief §1)

| # | item | result |
|---|---|---|
| 1 | reported metrics | **[MEASURED]** all 10 arm scores and all 10 contrasts recomputed exactly from saved predictions (`STAGE0_VERIFICATION.md`) |
| 2 | classes | 20 classes, 10 crops (`configs/datasets/stage0_classes.json`, 3 logged amendments) |
| 3 | independent parents | **59 root photos, not 60**: one photo (sha1 338c753b…) is labelled both `coffee::berry_blotch` and `coffee::brown_eye_spot` in SAGE and was used as a parent of both classes |
| 4 | parent-derived counts | 32 images per parent per arm (8 views/variants × 4 conditions) → 1,920 per arm; 96 per class |
| 5 | labels / mapping | class index = sorted class_id; test labels in `t0_predictions.npz` match `stage0_test.csv` row-for-row (asserted) |
| 6 | training steps | T0: lbfgs logistic regression to convergence (max 3000 it). T3: batch 64, drop_last → 30 steps/epoch × 15 = **450 steps** (real), 300 (fold 0) |
| 7 | checkpoint selection | none: fixed schedule, last epoch evaluated, **no checkpoints saved** (`checkpoints/` empty) → models cannot be re-evaluated, only predictions exist |
| 8 | real-test group independence | 4,202 photos in 3,763 near-duplicate groups; no test group contains a parent photo; all parents are in the `parent` partition; 16 test photos share a group with open-set-class images (affects open-set only) |
| 9 | test influence on choices | class amendments 05:52, test drawn 06:10 → no influence on class selection. After results: only the decision to add seeds 1–2 (no selection). From now on the Stage-0 test is treated as **exploratory (DEV)**; Stage 1 final claims use the untouched `val` partition |
| 10 | diagnostic symptoms in generated data | **[OBSERVED]** contact sheet (`figures/stage1/contact_sheet_r0.jpg`): 2D arms = photo (bug); U3D renders keep large lesions (late blight, leaf spots, curl) but blur fine texture (rust pustules, mildew, mottling) and shift colour towards yellow; photo-lifted 3D restores input-view texture. No expert check exists **[BLOCKER: raters]** |

Other facts: no Slurm logs (everything ran locally on beast0, logs in `logs/stage0_*.log`); synthetic-IID folds:
T0 3 parent folds, T3 fold 0 only; 1,840 SAGE photos (sha1) carry more than one class label overall, 13 within the
Stage-0 classes, 1 of them in the Stage-0 test (`coffee::berry_blotch`).

## 3. Corrective actions taken

1. `edit2d.py` fixed (uint8 mask) with a guard that raises if the organ is unchanged; fixed edits written to
   `data/synthetic/stage1/edits2d/` (seeds 0–2). Stage-0 files untouched.
2. New generator `src/generation/s1_arms.py` separating texture source (SAM 3D / photo / generic edit / KB edit)
   from geometry (KB operators): arms U3D, TEX3D, GEN3D, KBAPP3D, GEO3D, KB3D, TEXGEO3D and PHOTO2D, GEN2D, KB2D.
3. Stage-1 evaluation protocol: DEV = Stage-0 test; FINAL = `val` partition, guarded and access-logged
   (`src/s1/harness.py`, `results/stage1/FINAL_ACCESS_LOG.csv`).
