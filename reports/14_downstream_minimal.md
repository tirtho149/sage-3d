# 14 — Minimal downstream real-data falsification test (item 9)

Code: `src/downstream/` (`minimal_probe.py` is the entry point). Outputs: `results/downstream_minimal/`.
Execution log with timestamps and exact commands: `results/downstream_minimal/run_log.txt`.

## 1. Pre-registration (written 2026-10-14, before any feature was extracted or any probe was fitted)

**Question.** Does adding synthetic images of the pilot disease, derived from KB-guided 3D editing, improve
few-shot within-crop disease discrimination on *independent real* SAGE photos, beyond (a) real images only and
(b) the same number of *unedited* 3D renders of the same specimens? The test is built to *falsify* a benefit
cheaply. It cannot establish a benefit for the full augmentation study (see §6).

**Task.** For each crop with a pilot disease, a closed-set classifier over the pilot disease and the 3 largest
other SAGE classes of the same crop (classes ranked by total image count; Citrus has only 2 other classes with
≥ 30 images, so it gets 3 classes). SAGE contains no "healthy" classes for these crops, so all classes are diseases.

| Crop | Pilot class | Other classes |
|---|---|---|
| Peach | Leaf Curl | Bacterial Leaf Spot, Brown Rot, Peach Scab |
| Corn | Smut | Common Rust, Northern Leaf Blight, Gray Leaf Spot |
| Tomato | Leaf Mold | Tomato Leaf Curl Virus, Septoria Leaf Blotch, Bacterial Leaf Spot |
| Citrus | Canker | Greening Disease, Penicillium Fungi |
| Mango | Anthracnose | Bacterial Canker, Sooty Mould, Die Back |

**Splits (reused, not rebuilt).** `scripts/make_splits.py` split each crop by SHA-1 dedup + 64-bit pHash
near-duplicate clusters (Hamming ≤ 6), whole clusters assigned to build/opt/val/test (hash in
`data/metadata/splits.sha256`). All 15 pilot source photos are in `build`.
- *Evaluation set* = `test` ∪ `val` images of the selected classes (both cluster-disjoint from `build`),
  capped at 200 per class (random, seed 0).
- *Real training pool* = `build` images of the selected classes, excluding every member of the 15 pilot
  source clusters, capped at 300 per class (seed 0).
- *Background pool* (compositing only) = `opt` images of the selected classes of the crop.
- Leakage checks reported: pilot source SHA-1 in eval set (must be 0); pHash Hamming ≤ 6 and ≤ 10 between the
  15 `data/inputs/<sid>/image.png` and the eval set; between training pool and eval set; between pilot 2D
  edits and the eval set.

**Few-shot regime.** k ∈ {1, 4, 16} real training images per class; 50 random draws per k (seeded). The same
real draw is shared by every condition (paired design).

**Encoder and probe.** Frozen DINOv2 ViT-B/14 (local torch.hub checkpoint `dinov2_vitb14_pretrain.pth`;
no download), CLS token, Resize(256) + CenterCrop(224), ImageNet normalisation, L2-normalised features,
extracted once on GPU 0 and cached. Probe = scikit-learn `LogisticRegression(C=1.0, class_weight="balanced",
max_iter=3000)`. Class-balanced weighting is fixed so that added synthetic images change the *shape* of the
pilot class, not its prior weight.

**Conditions.** Synthetic images exist only for the pilot class (asymmetry: every synthetic condition adds
N = 18 images per crop to the pilot class only — 6 per specimen × 3 specimens — and nothing to other classes).

| Code | Added to pilot class (6 per specimen) | Background treatment |
|---|---|---|
| R | nothing (real only) | — |
| S | the 3 pilot *source photos*, each × 6 fixed classical augmentations (identity, h-flip, rot +15°, rot −15°, 0.7 centre crop, colour jitter) | natural (photo) |
| D | unedited 3D renders, views {0,3,6,9,12,15} | composited (below) |
| KB2D | KB 2D edits: `phaseA/C_stage/seed{0,1,2}` + `pilot/C/seed{0,1,2}` | natural (edits of the photo) |
| GEN3D | generic 3D: G (RMS-matched random geometry field) seed0 views {0,3,…,15}; Tomato/Citrus have no G, so E (GaussianEditor, generic text) seed0 | composited |
| KB3D | KB 3D: H (KB geometry J then KB appearance K) seed0 views {0,3,…,15}; Tomato/Citrus have no H, so K (KB appearance GaussianEditor) seed0 | composited |
| J (exploratory) | KB geometry only, seed0 views (Peach/Corn/Mango only) | composited |
| F (exploratory) | GaussianEditor with interim KB text, seed0 views (all crops) | composited |

D is the control for "rendered-looking images of these specimens labelled as the disease": KB3D differs from D
only by the KB edit. S is the control for "the specimen photo itself, augmented in 2D".

**Compositing (domain gap).** Renders are on white backgrounds; real photos have natural backgrounds. Every
render in D/GEN3D/KB3D/J/F is alpha-composited (render alpha) onto a background made from an `opt` image of the
same crop (class chosen uniformly among the crop's selected classes), Gaussian-blurred (radius 8 px) to
suppress lesion content, on a 256×256 canvas with the object bbox scaled to 70 % of the canvas side. The
background image and placement are fixed per (specimen, view slot), so D, GEN3D, KB3D, J and F of the same
slot share the same background (paired). Sensitivity analysis (secondary): the same renders on plain white.

**Metrics (on the real evaluation set).** Macro-F1 (primary), balanced accuracy, pilot-class recall, and FPR on
other classes (fraction of non-pilot eval images predicted as the pilot class).

**Statistics.** For each crop, k and condition c: Δ = metric(c) − metric(reference), averaged over the 50 draws;
pooled effect = unweighted mean over the 5 crops. 95 % CIs by a two-level paired bootstrap (2000 reps;
resample draws, and eval images stratified by class, independently per crop, shared across conditions).
Two-sided bootstrap p = 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0)). Holm correction within each family.

**Decision rule (primary endpoint = pooled macro-F1).**
- Decision family (Holm, 6 tests): {KB3D − R, KB3D − D} × k ∈ {1, 4, 16}.
- **Supported** iff for at least one k both KB3D − R and KB3D − D are > 0 with Holm-adjusted p < 0.05.
- **Not supported (falsified at this scale)** iff not supported AND for every k the upper 95 % CI bound of
  KB3D − D is < +0.02 macro-F1 (any benefit over unedited renders is smaller than 2 points or absent).
- **Inconclusive** otherwise.
- Secondary family (Holm, 15 tests): {S, D, KB2D, GEN3D, KB3D} − R × k ∈ {1, 4, 16}; also KB3D − KB2D and KB3D − GEN3D
  (descriptive). J and F are exploratory. Per-crop effects and the white-background sensitivity are descriptive
  and cannot change the decision.

<!-- RESULTS BELOW ARE APPENDED AFTER THE RUN -->

## 2. Execution

The run followed §1. The only change was to the file: Holm family size was corrected from "18" to "15" before any
data were touched; `results/downstream_minimal/preregistration_frozen.md` is the frozen copy. Commands and
timestamps are in `results/downstream_minimal/run_log.txt`. Wall time was about 20 min: 2 min extraction,
4 min of features on GPU 0 (TITAN X), 13 min of CPU probes + bootstrap. Total probe fits: 5 crops × 3 k × 50 draws × 13 conditions.
The temporary image cache (272 MB, scratch) was deleted after features were cached (`features.npz`, 12.7 MB).

## 3. Data counts and leakage checks

| Crop | Class | Eval (test ∪ val) | Train pool (build, pilot clusters removed) |
|---|---|---|---|
| Peach | **Leaf Curl** | 66 | 97 |
| | Bacterial Leaf Spot / Brown Rot / Peach Scab | 200 / 65 / 26 | 300 / 98 / 40 |
| Corn | **Smut** | 149 | 221 |
| | Common Rust / Northern Leaf Blight / Gray Leaf Spot | 200 / 200 / 200 | 300 / 300 / 300 |
| Tomato | **Leaf Mold** | 200 | 300 |
| | TYLCV / Septoria Leaf Blotch / Bacterial Leaf Spot | 200 / 200 / 200 | 300 / 300 / 300 |
| Citrus | **Canker** | 163 | 243 |
| | Greening Disease / Penicillium Fungi | 43 / 24 | 66 / 38 |
| Mango | **Anthracnose** | 200 | 300 |
| | Bacterial Canker / Sooty Mould / Die Back | 200 / 200 / 200 | 300 / 300 / 300 |

Eval set total: 2,936 real photos. Synthetic images: 18 per condition per crop (J: Peach/Corn/Mango only).

Leakage (`results/downstream_minimal/leakage.json`), all 5 crops:
- Pilot source SHA-1 in the eval set: **0**. In the training pool: **0** (pilot clusters were removed).
- pHash Hamming between the 15 `data/inputs/<sid>/image.png` and the eval set: **0** pairs ≤ 6. There is
  **1** pair ≤ 10 (Citrus, Hamming 8, above the split's near-duplicate threshold of 6); the other crops have a
  minimum of 12–18. In the training pool, 7 Citrus images are within Hamming ≤ 10 of a pilot input (none ≤ 6).
- Training pool vs eval set: **0** pairs ≤ 6 (by construction of the cluster split; minimum 8). Pairs ≤ 10:
  11 (Peach), 11 (Corn), 56 (Tomato), 27 (Citrus), 146 (Mango). These pairs are borderline
  near-duplicates that the Hamming ≤ 6 split rule allows. They affect every condition in the same way,
  because the real draws are shared.
- KB 2D edits vs eval set: 0 pairs ≤ 6. 5 pairs ≤ 10, all in Mango.

## 4. Results

Pooled = unweighted mean over the 5 crops. Each crop's value is the mean over 50 paired draws. CIs come from the
two-level bootstrap (draws × eval images, 2000 reps). Figure: `results/downstream_minimal/paired_differences.{png,pdf}`.
Full tables: `contrasts.csv`, `levels_pooled.csv`, `levels_per_crop.csv`, `per_draw_metrics.csv`, `summary.json`.

### 4.1 Absolute pooled macro-F1 (95 % CI)

| Condition | k = 1 | k = 4 | k = 16 |
|---|---|---|---|
| R (real only) | 0.559 [0.546, 0.571] | 0.713 [0.701, 0.724] | 0.806 [0.794, 0.818] |
| S (source photos + 2D aug) | 0.563 [0.548, 0.576] | **0.744** [0.731, 0.756] | 0.807 [0.794, 0.820] |
| D (unedited renders) | 0.534 [0.521, 0.547] | 0.674 [0.661, 0.686] | 0.801 [0.789, 0.814] |
| KB2D (C_stage + C) | 0.548 [0.534, 0.559] | 0.719 [0.705, 0.731] | 0.811 [0.799, 0.824] |
| GEN3D (G / E) | 0.549 [0.535, 0.561] | 0.685 [0.672, 0.697] | 0.799 [0.787, 0.812] |
| KB3D (H / K) | 0.534 [0.520, 0.546] | 0.679 [0.666, 0.692] | 0.803 [0.791, 0.816] |

### 4.2 Decision family: pooled macro-F1 differences (Holm over 6 tests)

| Contrast | k | Δ | 95 % CI | p | p (Holm) | Peach / Corn / Tomato / Citrus / Mango |
|---|---|---|---|---|---|---|
| KB3D − R | 1 | −0.025 | [−0.039, −0.013] | <0.001 | <0.001 | +0.026 / +0.006 / −0.060 / +0.012 / −0.110 |
| KB3D − R | 4 | −0.033 | [−0.045, −0.023] | <0.001 | <0.001 | −0.017 / +0.006 / −0.090 / +0.030 / −0.095 |
| KB3D − R | 16 | −0.003 | [−0.011, +0.005] | 0.50 | 1.00 | −0.001 / +0.006 / −0.030 / +0.018 / −0.007 |
| KB3D − D | 1 | −0.000 | [−0.006, +0.006] | 0.92 | 1.00 | +0.004 / +0.007 / +0.016 / +0.011 / −0.040 |
| KB3D − D | 4 | +0.005 | [+0.000, +0.011] | 0.049 | 0.20 | −0.003 / +0.003 / +0.059 / −0.027 / −0.006 |
| KB3D − D | 16 | +0.002 | [−0.003, +0.007] | 0.39 | 1.00 | −0.000 / +0.000 / +0.037 / −0.032 / +0.004 |

(p < 0.001 means no bootstrap replicate crossed 0 in 2000.)

### 4.3 Secondary family: each condition − R (pooled macro-F1, Holm over 15)

| Condition − R | k = 1 | k = 4 | k = 16 |
|---|---|---|---|
| S | +0.004 [−0.009, +0.016] | **+0.031 [+0.024, +0.038]**, Holm p < 0.001 | +0.001 [−0.004, +0.006] |
| D | −0.025 [−0.038, −0.013]* | −0.039 [−0.050, −0.028]* | −0.005 [−0.014, +0.005] |
| KB2D | −0.011 [−0.024, +0.001] | +0.006 [−0.003, +0.015] | +0.005 [−0.001, +0.012] |
| GEN3D | −0.010 [−0.024, +0.002] | −0.028 [−0.038, −0.018]* | −0.007 [−0.014, −0.000] |
| KB3D | −0.025 [−0.039, −0.013]* | −0.033 [−0.045, −0.023]* | −0.003 [−0.011, +0.005] |

\* Holm p < 0.05 (harm).

### 4.4 Pilot recall and false-positive rate (pooled Δ vs R)

| Condition | recall k=1 / 4 / 16 | FPR k=1 / 4 / 16 |
|---|---|---|
| S | +0.33 / +0.04 / −0.03 | +0.21 / −0.00 / −0.01 |
| D | +0.07 / −0.24 / −0.10 | +0.13 / −0.03 / −0.01 |
| KB2D | +0.29 / −0.06 / −0.04 | +0.21 / −0.01 / −0.01 |
| GEN3D | +0.15 / −0.20 / −0.08 | +0.14 / −0.03 / −0.01 |
| KB3D | +0.16 / −0.21 / −0.07 | +0.18 / −0.02 / −0.01 |

At k = 1, every synthetic set moves the decision boundary toward the pilot class. Recall rises, but FPR rises
by about as much, so macro-F1 does not improve. At k ≥ 4 the 18 synthetic images outnumber the 4–16 real
pilot images. Under class-balanced weighting they pull the pilot class toward the rendered domain, and real
pilot photos are then missed (recall −0.20 to −0.24 for all render conditions). The KB edit changes this
very little: KB3D − D on recall is +0.09 / +0.03 / +0.03. The large Tomato effect (+0.15 to +0.37) is offset by
Citrus (−0.04).

### 4.5 Descriptive and sensitivity results (macro-F1, pooled)

- KB3D − KB2D: −0.014, −0.039, −0.008 at k = 1 / 4 / 16 (all CIs < 0). The 3D KB images were worse than the 2D KB edits.
- KB3D − GEN3D: −0.015, −0.005, +0.004. The KB edit is not better than an equally large generic 3D edit.
- D − S: −0.029, −0.070, −0.006. Rendering the specimen costs accuracy relative to augmenting its photo in 2D.
- J − D (Peach/Corn/Mango, exploratory): −0.001, +0.002, +0.000. F − D: −0.000, −0.007, −0.005.
- White-background sensitivity: KB3D − D on white was −0.008, +0.010, +0.000. The upper CI bound at k = 4 is
  +0.016, still below +0.02. Compositing therefore does not change the conclusion.

## 5. Decision (pre-registered rule applied mechanically)

**NOT SUPPORTED (falsified at this scale).** KB3D does not beat real-only at any k. It is significantly worse at
k = 1 and k = 4 and equal at k = 16. It does not beat the unedited-render control D after Holm correction.
The upper 95 % CI bound of KB3D − D is +0.006, +0.011 and +0.007 at k = 1, 4 and 16, all below the
pre-registered +0.02 margin. Any downstream benefit from the KB edit, over rendering the same specimen
without an edit, is therefore smaller than 2 macro-F1 points in this setting.

The only condition that helped was **S**, the specimen's own photo with classical 2D augmentation (+0.031 at
k = 4). Rendering the same specimens through the 3D pipeline (D) lowered accuracy instead. This points to a
rendering/domain gap, not to missing disease knowledge, as the first bottleneck for 3D augmentation.

## 6. What this test can and cannot say

- **It can say:** with 3 specimens per class, 18 synthetic images, and a frozen DINOv2 + linear probe, KB-guided
  3D renders add no measurable real-photo discrimination benefit over unedited renders or over real data. It
  rules out a benefit larger than about 2 macro-F1 points over D in this regime.
- **It cannot say:** whether 250 assets, fine-tuned (not frozen) encoders, many more synthetic images per class,
  synthetic images for *all* classes, or better photo-realistic rendering would help. Those are the questions
  of the gated 250-GPU-h study. A null result from 3 specimens per class is weak evidence about them, though it
  gives no positive signal that would justify the cost.
- **Asymmetry:** synthetic images exist only for the pilot class. Any condition therefore partly acts as a
  class-prior/boundary shift (see §4.4). D is the control for this effect, and KB3D ≈ D.
- **Domain gap:** renders are single SAM 3D objects composited onto blurred real backgrounds. Real SAGE photos
  differ in lighting, focus and clutter. Two background treatments (blurred real, white) gave the same answer.
- **Source confounds:** SAGE classes come from different source datasets (e.g. Peach Bacterial Leaf Spot is mostly
  lab datasets). Real-only probes can exploit source style. This applies to all conditions equally and does not
  bias the paired contrasts. It does limit absolute accuracies.
- **Arm substitution:** G/H/J are unavailable for Tomato and Citrus, so GEN3D/KB3D use E/K there (pre-registered).
  KB3D thus mixes two edit types across crops, and the per-crop estimates are given for that reason. Tomato is
  the only crop where KB3D − D is positive at every k (+0.016, +0.059, +0.037); Citrus is negative at k ≥ 4.
  A single-crop result is descriptive and gives no claim.
- **No expert validation** of the synthetic images was done. This test measures only downstream utility.
