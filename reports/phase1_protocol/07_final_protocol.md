# 07 — Consolidated Experimental Protocol (v0.1, 2026-10-12)

**Knowledge-grounded 3D disease assets from single SAGE images: what, if anything, does 3D add?**

This document merges two sources:

- The diagnosis-guided protocol PDF found on Nova ("Diagnosis Guided Optimization of SAM 3D Crop Disease Twins", 13 pp.).
- The master prompt (Phases 0–15).

It is constrained by four decisions made on 2026-10-12:

1. SAGE is the only data source.
2. No multi-view or public-3D data is used.
3. Everything runs locally on beast0 (2× GTX TITAN X 12 GB).
4. GaussianEditor (buaacyw) is the 3D editor.

Status words used throughout: **exists / implemented / tested / blocked**.

---

## 1. Central question and what can and cannot be claimed

> Does a source-grounded (SAGE-KB) 3D disease asset, built from one SAGE image with SAM 3D and edited with GaussianEditor, provide a measurable advantage over (i) the original image and (ii) KB-guided 2D editing, and is any advantage specific to symptoms that are geometric or occluded?

The data contain no specimen-matched second view. So every structure a 3D asset shows outside the input view is **synthetic, source-supported at best, and never "recovered"**. The protocol therefore claims only what single-view data can test.

| Hypothesis | Status in this protocol | Test |
|---|---|---|
| H1 Reconstruction reliability | **Partial.** Input-view fidelity only. | E1 |
| H2 KB-grounded editing beats generic editing | **In scope** | E2 |
| H3 Multi-view consistency (3D vs independent 2D edits) | **In scope as consistency only**, never as accuracy | E4 |
| H4 Hidden-trait recovery | **Out of scope.** Needs specimen-matched views SAGE lacks. Reported as "plausible-unverified" rates only. | E3 (reduced) |
| H5 Downstream utility on real SAGE test images | **In scope. Primary confirmatory outcome.** | E5 |

The PDF's primary confirmatory claim is adopted as the H5 test: *under equal observation and compute budgets, KB-guided 3D references improve diagnosis of untouched real images more than static 2D references, raw 3D, random 3D edits, and KB-guided 2D edits.*

## 2. Cases (from `01_case_selection.md`)

| Case | Class | Question it answers |
|---|---|---|
| A | Peach Leaf Curl | Geometry-dominant symptom (pucker/curl) |
| B | Tomato Leaf Mold | Occlusion-dominant symptom (lower-surface mold never seen in input) |
| C | Corn Smut | Volumetric outgrowth (galls) |
| D | Citrus Canker | Surface relief (raised/crater lesions) |
| Z | Mango Anthracnose | **Negative control**: flat, appearance-only. 3D should add nothing. |

**Predicted pattern if 3D matters:** benefit(A, B, C, D) > benefit(Z) ≈ 0. A uniform benefit across A–D and Z would indicate the gain comes from more or different images, not from geometry.

## 3. Data partitions and leakage control (PDF §5, adapted)

- **Unit:** one deduplicated SAGE image (SHA-1 of image bytes). SAGE has no specimen IDs. Near-duplicates are clustered with a perceptual hash (pHash, Hamming ≤ 6), and a whole cluster stays in one partition.
- **Split before any processing:** D_build 45% / D_opt 25% / D_val 15% / D_test 15%, stratified by class and grouped by pHash cluster. For classes with <100 images, use grouped 5-fold CV inside non-test data. Each split is frozen with a hash file.
- **Source holdout:** where a class has ≥2 SAGE `source` datasets, one source is reserved as an external test domain.
- **Prohibited:** test images in reconstruction, masks, prompt design, reward, selection, or early stopping. This is enforced by `src/evaluation/leakage.py`, which checks every asset's provenance hash against the test manifest.

## 4. Experimental arms (master §7 merged with PDF §13)

All arms receive the same input image and the same SAGE KB record for that class. "KB" always means the validated, provenance-carrying instruction from `src/kb/prompt_builder.py`.

| Arm | Representation | Knowledge | Editor | Purpose | PDF ID |
|---|---|---|---|---|---|
| A | Original real 2D image | — | — | Observed-information baseline | B2 |
| B | 2D | Generic text | InstructPix2Pix | Non-3D generative baseline | — |
| C | 2D | **SAGE KB** | InstructPix2Pix | Is knowledge alone enough? | — |
| D | SAM 3D, unedited | — | — | Reconstruction alone | B3 |
| E | SAM 3D | Generic text | GaussianEditor (edit-n2n) | 3D editing without knowledge | — |
| F | SAM 3D | **SAGE KB** | GaussianEditor (edit-n2n) | **Proposed method** | B6 / B8-lite |
| R | SAM 3D | Random valid edit | GaussianEditor | Controls for "any extra variation" | B4 |
| K | Text only | SAGE KB | — | Knowledge-only diagnosis baseline (E5 only) | B1 |
| N | Image only | — | — | Foundation-model baseline (E5 only) | B0 |

**Why B/C use InstructPix2Pix:** GaussianEditor's `edit-n2n` mode is itself guided by InstructPix2Pix. Using the same 2D editor for the 2D arms isolates the effect of the 3D representation. Same prompt, same guidance scales, same seeds.

**Deferred until a GO at the pilot gate** (PDF §9–10, kept but not run now): CMA-ES diagnosis-guided optimisation (B8), PPO/SAC policy (B9), the VLM3D-style critic (B7), and GaussCtrl as a second 3D editor. These are expensive on 12 GB Maxwell hardware. They only make sense if F already beats C.

**Matching:** identical input image, mask, KB record, prompt text, seeds {0,1,2} for the pilot ({0..4} main), output resolution 512², camera set, and per-asset GPU-time cap. Reconstruction seed and editing seed are recorded separately.

## 5. Experiments

### E1 — Reconstruction fidelity (input view only)
- Render arm D at the **input camera** recovered by SAM 3D's layout/pose output.
- **Metrics:** mask IoU with the SAM mask; LPIPS and SSIM inside the mask; colour ΔE in CIELAB inside the mask.
- **Stability:** across seeds, report Chamfer distance between the seed-wise splat centres (normalised by bounding-box diagonal). This measures **reproducibility, not accuracy**.
- **Per-class failure taxonomy:** flattened organ, merged background, missing organ parts, colour shift, floaters.
- **Not claimed:** accuracy of unseen views or of real geometry.

### E2 — KB-guided vs generic editing (H2)
- **Arms:** D, E, F (3D) and A, B, C (2D).
- **Automatic metrics:**
  - Symptom correspondence: the CLIP or VLM score of the rendered or edited image against each KB clause, minus its score against matched clauses from the **mismatched-KB negative control** (another disease of the same crop).
  - Preservation outside the target organ: LPIPS between before and after, inside versus outside the organ mask.
  - Unsupported-symptom rate: share of VLM-detected symptoms with no supporting KB span.
  - Failure rate.
- **Human:** blinded expert ratings (organ validity, symptom validity, severity plausibility, "would mislead diagnosis"), with two raters and weighted κ. **Blocked: needs raters.** A VLM judge is never used as the only evidence of biological accuracy.

### E3 (reduced) — What does the 3D asset add beyond the input view?
For each specimen, list the KB-described properties of its class and tag each one:
- **Observed in input:** confirmed by a rater looking at the input image.
- **Represented only in 3D asset:** e.g. the lower-surface mold in Case B, or the pucker depth in Case A.
- **Not represented anywhere.**

All "represented only in 3D" properties are reported as **plausible-unverified**, never as recovered. Reported per case:
- the fraction of KB properties that the 3D asset represents but the 2D image cannot;
- the fraction the 2D KB edit (arm C) can represent;
- the unsupported-addition rate.

This answers the advisor's question ("what does 3D add?") honestly. It measures **representational capacity**, not truth.

### E4 — Multi-view consistency (H3, consistency ≠ accuracy)
- **Shared cameras:** 8 fixed orbit views plus top and bottom. For 2D arms, the 2D edit is applied independently to each SAM 3D render of arm D (same prompt, same seed). This is the standard "independent per-view 2D editing" baseline.
- **Metrics:**
  - Cross-view lesion persistence: back-project lesion masks via known cameras and take the IoU in the asset's UV/Gaussian-ID space.
  - Severity variance across views.
  - Feature-consistency score: CLIP embedding variance.
- **Mandatory figure:** one asset that is perfectly consistent but biologically wrong (Fig. 5).

### E5 — Downstream diagnosis on real SAGE test images (H5, primary)
- **Task:** SAGE-style reference-based diagnosis. A frozen local VLM (Qwen2.5-VL-3B-Instruct, the largest that fits 12 GB in fp16) receives a real D_test query image plus a reference set per candidate class. It answers by constrained multiple choice over the crop's classes.
- **Conditions:** N, K, A (real 2D references from D_build), B, C, D, E, R, F, and A+F (complementarity, PDF B10).
- **Equal budgets:** same number of reference images per candidate (k ∈ {1, 4, 8}); same Qwen calls.
- **Candidates:** all SAGE classes of the crop. For Mango, all 4 classes; for the others, the full crop class list, capped at 10 with confusers chosen by SAGE confusion statistics before test.
- **Primary outcome:** macro-F1 and paired accuracy, **F vs C** (does 3D beat KB-2D?) and **F vs A** (does it beat real references?). Holm-corrected co-primaries.
- **Secondary outcomes:** gain in A–D vs Z (geometry specificity); log-loss; ECE.

## 6. Ablations (master §13 ∩ PDF §14, trimmed to what this hardware can run)

| Axis | Levels | When |
|---|---|---|
| Knowledge | none · generic · KB · KB-no-geometry · KB-no-organ · **KB-mismatched (negative control)** | Pilot: none/generic/KB/mismatched. Main: all. |
| Representation | 2D · raw 3D · 3D appearance-only edit · full 3D edit | Pilot |
| Views (E5) | 1 · 4 · 8 references | Main |
| Data | case A–D vs Z; source dataset; image quality (auto_ok) | Main |
| Editor | GaussianEditor · none | Pilot (GaussCtrl deferred) |

The full factorial design is not run. Primary comparisons are fixed above, before data.

## 7. Statistics (PDF §16, adopted)

- **Unit:** an independent real test image, clustered by pHash cluster and source. Rendered views are never counted as independent samples.
- **Paired correctness:** exact McNemar test; cluster-bootstrap 95% CI on the accuracy difference (10,000 resamples).
- **Macro-F1:** paired stratified cluster bootstrap.
- **Multiple methods:** Cochran's Q, then Holm-corrected pairwise McNemar.
- **Continuous E1/E2/E4 metrics:** Wilcoxon signed-rank on specimen-level means, with rank-biserial effect size. Hierarchical bootstrap over specimens and seeds.
- **Power:** discordant-pair rate is estimated from the pilot. Target ≥ 80% power for a 5-pp paired accuracy gain after Holm, inflated by the design effect 1+(m−1)ICC.
- **"Supported"** requires all three: corrected p < 0.05, a CI that excludes 0, and an effect ≥ 5 pp. All runs are reported, including failures. Compute cost (GPU-seconds, peak VRAM) is reported per arm.

## 8. Go / Pivot / No-go (thresholds set after the pilot, not before)

| Outcome | Criterion (directional; numeric thresholds frozen after pilot variance is known) |
|---|---|
| **GO** (knowledge-grounded 3D) | F > C and F > A on E5 with supported effect, and the effect is larger on A–D than on Z |
| **PIVOT** (3D as simulation/inspection) | F ≈ C on E5 but E3/E4 show representational and consistency advantages. Publish as KB-grounded 3D simulation with limited claims. |
| **PIVOT** (2D KB editing) | C > B and C ≥ F. Knowledge helps, 3D does not. |
| **NO-GO** | SAM 3D input-view fidelity fails on most specimens, GaussianEditor edits do not localise, or the unsupported-symptom rate dominates |

PDF decision gates 1–5 are retained verbatim as secondary gates.

## 9. Execution modes

1. **Protocol validation:** mock outputs explicitly marked `synthetic_fixture: true`. Exercises loading, pairing, metrics, statistics, and figures.
2. **Minimal pilot:** 15 specimens (3 × 5 classes), 3 seeds, arms A–F (+R if budget allows), E1, E2-auto, E3-reduced, E4. E5 runs as a feasibility check on ≤100 test images only.
3. **Main:** only after the pilot gate; resumable; per-specimen outputs.

## 10. Compute reality (measured, see `04_environment.md`)

- SAM 3D on a 12 GB TITAN X needs stage-wise CPU offloading (`src/reconstruction/offload.py`). Official requirement: 32 GB.
- gsplat needed a pre-Volta patch (`patches_gsplat_preVolta.diff`).
- Per-asset runtime is recorded in each `run_meta.json`.
- Main-study size is set from the measured pilot throughput, not assumed.

## 11. Deliverable status

| Deliverable | Exists | Implemented | Tested | Real data | Blocked by |
|---|---|---|---|---|---|
| Repository audit (00) | yes | — | — | yes | — |
| Case shortlist (01, candidate_cases.csv) | yes | — | — | yes | — |
| Dataset protocol + manifests | partial | specimen_manifest.csv | yes | yes | split/pHash script pending |
| KB schema/validator/retriever/prompt builder | yes | `src/kb/*` | smoke-tested on 4 classes | yes | — |
| Baseline table | yes (§4) | runner pending | — | — | — |
| SAM 3D local | yes | offload wrapper | **running** | yes | — |
| GaussianEditor local | env building | — | — | — | — |
| 2D editor (IP2P) | — | — | — | — | — |
| E1–E5 specs | yes | — | — | — | — |
| Statistics plan | yes (§7) | pending | — | — | — |
| Figures | — | — | — | — | — |
| Expert validation | — | — | — | — | **raters needed** |
