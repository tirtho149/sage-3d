# 09 — Failure analysis and taxonomy (brief v2 §20, Fig. 9)

Every entry below is an observed failure with a stored artifact. Full root-cause records are in `00_full_failure_audit.md`. Figure: `figures/paper/fig9_failures.png`.

## A. Technical failures (pipeline correctness)

| # | Failure | Class | Detection | Artifact | Status |
|---|---|---|---|---|---|
| A1 | Arm F crashed at config load (colon in KB prompt parsed as a YAML mapping) | prompt serialisation | rc=1 in 5.7 s | `results/pilot/Peach__*/log.txt` | FIXED, TESTED live (Tomato 0015 F, `prompt_delivered: true`) |
| A2 | GE returned the input unchanged (+inf opacity → NaN anchor loss → every fp16 step skipped) | numerical | bit-identical output | `results/ge_diagnosis/Peach__Leaf_Curl__0006_E_diff.json` | FIXED, TESTED live (0006 re-run changes parameters) |
| A3 | GE crashed at CUDA init (inherited `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`, unsupported by torch 2.0.1) | environment | rc=3 in 6 s | `results/pilot/runner_gpu1.jsonl` 17:12 | FIXED (env stripped in runner and `run_ge.sh`) |
| A4 | Pose refinement A3 diverged (IoU → 0.00–0.23) | optimisation | IoU(A3) < IoU(A2) | `results/alignment/{Peach__Leaf_Curl__0063,Peach__Leaf_Curl__0006,Tomato__Leaf_Mold__0092}` | HANDLED (A3\* acceptance rule) |
| A5 | First pucker operator produced a uniform lift, not blisters | operator design | synthetic-sheet test | `tests/test_geometry_ops.py` | FIXED, regenerated |
| A6 | Attribute detector read half the samples as zero (voxel-grid padding) | metric bug | synthetic-sheet test | — | FIXED |
| A7 | Disk exhaustion risk (GE checkpoints ~115 MB/run; uncompressed render arrays ~18 MB/arm) | operations | `df` | — | FIXED (pruning; compressed alpha/depth) |

## B. Reconstruction failures (SAM 3D)

| Specimen | Failure | Evidence |
|---|---|---|
| Peach 0006 | Strongly curled leaf reconstructed as a **closed ring**. A topology error that pose correction cannot fix | A0 IoU 0.43; best 0.50 (A2); labelled `reconstruction_failure` and kept in all tables |
| Peach 0063 | Borderline. The threshold for this label was chosen after seeing the data (report 01) | A3\* IoU 0.67 |
| Systematic | Over-scaled objects at the predicted pose (13/15 too large; Tomato +150 %) | report 01 |
| Representation | Leaves are **single Gaussian sheets with view-independent colour (SH degree 0)**: upper and lower surfaces cannot differ | inspection of every leaf asset; affects Tomato Leaf Mold (lower-surface mold) and Squash Downy Mildew |

## C. Editing failures (appearance vs geometry)

| Arm | Failure | Evidence |
|---|---|---|
| E/F (GaussianEditor) | **Appearance edit plus incoherent geometric jitter**, not deformation | coherence 0.31–0.32 (noise ≈ 0.25); sheet ×1.8–2.3 thicker; grain ×13.6, fuzz ×1.9 on 0087; KB geometric recall 0 |
| E/F | View-to-view texture inconsistency of the same surface point | same-point ΔE 7.4 on 0087 vs a 2.1 floor |
| J (KB geometry) | Blister placement is random, not at observed lesion sites. On 0087 it *adds* puckering to a leaf that already visibly puckers | report 03 §6 |
| G (generic geometry) | Can produce attributes the KB does not call for (outgrowth on 0087 seed 0) | `unsupported_geometry:outgrowth` in QC |

## D. Baseline-validity failures (fixed so comparisons are fair)

| Baseline | Problem | Fix |
|---|---|---|
| B/C (2D) | Edits composited inside the original mask, so they *cannot* change shape | Added arm I (dilated mask, image guidance 1.2) |
| Bmv/Cmv (per-view 2D) | Run without a mask: IP2P invented whole new scenes | Post-hoc masking, provably identical to a masked run (`mask_perview.py`) |
| Cross-view metric | Mean-colour consistency rated per-view 2D as consistent as 3D | Replaced by same-point correspondence ΔE |

## E. Biological-plausibility failures (observed; not expert-validated)

| Case | Observation |
|---|---|
| Arm I on Peach 0087 | Given freedom to change shape, the 2D editor replaced the **real, visible puckering** with a larger, flat, smooth leaf with painted veins. That is the opposite of the KB's "puckered" |
| Generic 2D (B) and wrong-stage KB (C-NG) | Yellow/orange "autumn" recolouring. "Reddish or purple" appears only when the prompt contains it |
| KB stage mixing (interim F/C prompts) | Combined early (distorted, reddish) and later (powdery grey) symptoms. Replaced by stage-consistent prompts |

## F. Cannot be evaluated here (blocked, not failed)
- **External validity of novel views and hidden structure:** there is no second real view of any specimen (SAGE-only decision).
- **Biological validity:** no expert raters.
- **Downstream utility (§18):** not run (report 10 explains why and what it would cost).
