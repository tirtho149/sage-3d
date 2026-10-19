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
