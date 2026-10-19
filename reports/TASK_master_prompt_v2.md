# Task brief v2: full pipeline repair, geometry-aware knowledge editing, controlled evaluation

Received 2026-10-12, 16:35 CDT. The brief is recorded below, followed by how it maps to this project's earlier decisions. Execution status lives in `RESEARCH_PROGRESS.md`; the final summary is in `reports/NEXT_EXPERIMENT_RESULTS.md`.

## Adaptation to this project (binding decisions made earlier by the user)

| Brief says | What we do | Why |
|---|---|---|
| Nova / Slurm jobs, job IDs | All runs go on the local machine (beast0, 2× TITAN X 12 GB, Maxwell sm_52). There are no Slurm IDs; each run is identified by a JSONL run log with PID, exit code, runtime and output checks. | User: "do everything here". |
| Search public multi-view / 3D plant datasets | **Not done (BLOCKED by user decision).** We use SAGE only, so every novel view is *synthetic, unverified*. Track 1 (healthy-plant reconstruction accuracy) therefore has no data. | User: "no multi view / u have to use … sage somehow" (twice rejected searches). |
| `testtrial.pdf` (interim report) | This is the Overleaf project `6acd4ee4b048478b8c02c78f` (title "testtrial"), generated from our own outputs. Its numbers are verified against `results/pilot/**/e1.json`. | — |
| Expert review | **BLOCKED**: no raters are available. Every biological-validity claim is labelled "unvalidated". | — |
| Phase C categories (Black Knot, Squash Downy Mildew, Grape Leafroll) | These were deferred or excluded in `phase1_protocol/01_case_selection.md`: too few images, a label-noise flag, and canopy scenes respectively. They are re-checked only once the Phase A/B gates pass. | Evidence in `01_case_selection.md`. |
| Repository layout | Adapted to the existing tree: no imports are broken and new modules sit alongside the old ones. | — |

## Brief (verbatim structure; condensed wording where marked)

**Role:** autonomous research-engineering team. Make concrete code and experiment changes, then test, execute and document them. Do not kill other users' jobs, delete original data, overwrite irreplaceable outputs or launch unusually expensive runs without approval.

### Part I: Understand the current failure
- Audit the interim report, protocol, scripts, manifests, KB, SAM 3D outputs, GaussianEditor outputs, prompts, 2D edits, renders, logs, environments and metrics.
- Reported E1 IoU values, to be verified:
  - Citrus 0012 0.73, Citrus 0016 0.76
  - Corn 0012 0.63, Corn 0025 0.61, Corn 0052 0.72
  - Peach 0006 0.43, Peach 0063 0.58, Peach 0087 0.84
  - Tomato 0015 0.40, Tomato 0052 0.55, Tomato 0092 0.47
- Errors may come from the camera, the silhouette, the mask or the mesh→Gaussian step, not only from SAM 3D.
- Known failures:
  - F did not run (colon in the prompt).
  - E on Peach: 0006 no change; 0063/0087 darker and grainy with fuzzy edges, no geometry change.
  - 2D edits only recolour.
  - Novel views are unverifiable.
- Deliverable: `reports/00_full_failure_audit.md`. Each failure gets: code location, root cause, evidence, reproduction, modification, verification test, status (CONFIRMED / FIXED / TESTED / UNRESOLVED / BLOCKED) and residual risk.

### Part II: Repair
- **§2 Arm F:**
  - Use structured, serialized prompts instead of raw text in the command line.
  - Keep prompt generation separate from execution.
  - Add preflight validation and unit tests (colons, quotes, newlines, braces, non-ASCII).
  - Verify the prompt the model actually receives.
  - Keep all intermediates and re-run only failed work.
  - Judge success by outputs, exit codes and logs; exit code 0 alone is not success.
  - Acceptance: loads, initialises, edits a real specimen, saves the asset, saves before/after renders, exports metrics, records config and logs.
- **§3 SAM 3D / camera alignment:**
  - Audit preprocessing (resolution, EXIF, resize, crop, mask, alpha, normalisation, origin), the prediction (pose, scale, R, t, conversion, completeness) and the camera (f, principal point, FOV, projection, extrinsics, conventions, normalisation).
  - Study A0–A5 (existing; camera convention; scale + translation; camera optimisation with frozen geometry; geometry-preserving refinement; joint refinement only if needed).
  - Metrics: IoU, boundary F, centroid error, area error, landmark reprojection, bounding-box error, projection consistency, failure categories.
  - Per-specimen artifacts: original, mask, original render, corrected render, overlay, camera, metrics, log.
  - Report all 11; no silent exclusion.

### Part III: Geometric editing
- **§4 GaussianEditor diagnosis:** find which parameters are trainable and receive gradients. Then compare parameters before and after editing, matching Gaussians correctly despite densify/prune:
  - displacement (mean, median, p95);
  - changes in rotation, scale, opacity and colour;
  - count, added and deleted;
  - per region.
- Visuals: displacement heatmap, geometry-only and appearance-only comparisons, edited vs stable regions, silhouettes. If only colour changed, call it appearance editing.
- **§5 Geometry-aware editing:**
  - Formulation: μ' = μ + m·Δμ with regularisers (target deformation, ARAP / rigidity, Laplacian, anchor, silhouette, multi-view, biology). Each must have a measurable target.
  - Methods to compare:
    - G1: GaussianEditor, tuned;
    - G2: explicit position optimisation;
    - G3: mesh-guided deformation transferred to the Gaussians;
    - G4: geometry, then appearance refinement.
  - Prefer the simplest method that works. Check whether the representation has the topology and correspondence needed, and document how geometry is transferred to the Gaussians.

### Part IV: KB grounding
- **§6:** structured symptom specs with:
  - crop, disease, organ and stage;
  - visible symptoms, plus KB-supported additions;
  - geometry and appearance attributes;
  - localisation, severity and confidence;
  - citation, prohibited modifications and uncertainty.
- Add stage-consistency rules (no mixing of stages; abstain or keep multiple hypotheses when ambiguous).
- Generate separate prompts: geometry-only, appearance-only, joint, generic, KB-no-geometry, mismatched.
- Save the exact KB sentences behind every edit.

### Part V: Baselines
- Keep A, B, C, C-NG, C-MM, D, E, F. Add:
  - G: geometry 3D, no KB;
  - H: KB + geometry 3D;
  - I: a 2D editor that can change shape, given the same KB;
  - J: KB geometry-only 3D;
  - K: KB appearance-only 3D.
- Label Bmv/Cmv as independent per-view 2D edits.

### Part VI: Revised pilot
- **Phase A:** Peach 0087, all arms D–K, seed-controlled. Diagnose:
  - whether F runs;
  - whether E changes geometry;
  - whether G deforms;
  - whether H adds grounded change;
  - whether the KB alters deformation;
  - whether 2D shape editing matches the result;
  - whether views are consistent;
  - whether non-target regions stay stable.
  - Gate: do not scale if geometry editing fails.
- **Phase B:** all 11 cases. Record which diseases need geometry and which do not.
- **Phase C:** further categories, balanced selection.

### Part VII: Metrics
- **§11 Geometry:** displacement, normalised deformation, curvature change, silhouette difference, local edit precision, surface regularity, preservation. Nothing is physically calibrated.
- **§12 Appearance:** target change, non-target preservation, similarity, colour, texture, quality, artifacts.
- **§13 KB:** symptom precision and recall, unsupported-attribute rate, organ localisation, stage consistency, expert validity. A VLM score is supplementary only.
- **§14 Multi-view:** lesion and deformation stability, identity, correspondence, visibility, continuity. Internal consistency is not external validity.
- **§15 Statistics:** the specimen is the unit. Paired comparisons, specimen-level CIs, cluster bootstrap, effect sizes, multiplicity correction, subgroups, seed variability, failure rates.

### Part VIII: Hypothesis tests
- **§16:** 2×2 ablation of geometry × KB, with the interaction effect.
- **§17:** separate what is observable from what is occluded, ambiguous, KB-supported or synthetic. Hidden structure counts as a plausible synthetic hypothesis unless a reference exists. (Dataset discovery is BLOCKED; see above.)

### Part IX: Downstream (§18)
- Classifier conditions:
  - real only;
  - real + generic 2D;
  - real + KB 2D;
  - real + generic 3D;
  - real + KB 3D;
  - real + geometry-KB 3D.
- Keep the synthetic count and training budget matched; no test leakage.
- Metrics: top-1, macro-F1, balanced accuracy, per-class recall, confusion matrix, CIs, cost.

### Part X: Compute (§19)
- Audit the environment. Run smoke → diagnostic → expanded.
- Track GPU-h, peak memory, wall clock, iterations, cost per success, failures, retries and storage.
- Deliverables: `reports/gpu_compute_audit.md` and `reports/gpu_scaling_estimate.md`.

### Part XI: Automated QC (§20)
- Flag:
  - missing or invalid outputs: empty, missing, NaN/Inf, invalid parameters, inconsistent render sizes, incomplete optimisation;
  - geometry failures: negligible motion, exploding deformation, disappearance, silhouette collapse;
  - image defects: wrong mask, texture corruption, grain, fuzzy edges;
  - camera misalignment;
  - unsupported symptoms.
- Thresholds come from pilot measurements. Technical QC is not biological validation.

### Part XII: Figures (§21)
1. Pipeline
2. Reconstruction debugging
3. Geometry editing D/E/F/G/H
4. KB ablation
5. Displacement
6. Biological symptoms
7. Cross-view consistency
8. Downstream
9. Failures
10. Added value of 3D

Only real outputs, programmatic generation, vector charts, high-res renders, provenance, synthetic labels. Inspect the final PDF.

### Part XIII: Layout
Configs, src modules, experiments, scripts, results, figures, and `reports/00`–`10` plus `NEXT_EXPERIMENT_RESULTS.md`.

### Part XIV: Order and gates
1. Repairs. Gate: editing completes and reconstruction is characterised.
2. Geometry. Gate: reproducible deformation without catastrophic damage.
3. KB. Gate: every edit is traceable.
4. Controlled pilot. Gate: evaluation reproducible.
5. Scale-up.

### Part XV: Decision (§24)
- Outcome A: verified reconstruction.
- Outcome B: KB-controlled 3D simulation.
- Outcome C: downstream augmentation.
- Outcome D: no established advantage.

The outcome must follow from the evidence.

### Part XVI: Completion
- Report: code changes, files, fixed and unresolved bugs, commands, run IDs, completed vs failed runs, verified assets, all metrics, comparisons, statistics, panels, downstream results, GPU use, failure taxonomy, PDF, editable figure sources, next direction.
- Final table with columns: Experiment, Baseline, Proposed, Metric, Result, Evidence, Status. No estimated values.
- Keep `RESEARCH_PROGRESS.md` up to date after every stage.
