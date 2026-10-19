# Domain-gap diagnosis (Stage 1, 2026-10-16)

All numbers are **DEV** (the Stage-0 test, exploratory), T0 frozen DINOv2 ViT-S/14 + logistic regression, 20 classes,
the 60 Stage-0 parent assets, fixed images per parent within each comparison. Mean ± SD over 3 matched replicates
unless stated. **[M]** measured, **[I]** inference, **[H]** hypothesis, **[B]** blocker.

## A. Background shift (8 images/parent, same 8 views, one condition)
| arm | R0 white | R1 own (source-like) | R2 natural plates (other classes) | R5 R2 + photometric |
|---|---|---|---|---|
| PHOTO2D | 0.607 ± 0.005 | **0.696 ± 0.001** | 0.598 ± 0.011 | 0.546 ± 0.012 |
| TEX3D | 0.538 | **0.655** | 0.528 ± 0.006 | 0.499 ± 0.013 |
| HARM3D | 0.446 | **0.565** | 0.445 ± 0.015 | 0.485 ± 0.021 |
| U3D | 0.435 | **0.562** | 0.433 ± 0.015 | 0.477 ± 0.019 |
| background only (PLATE2D, organ inpainted) | – | 0.129 ± 0.009 | – | – |

- [M] The parent's own background is best for every arm (+0.09 to +0.13 over white); class-independent natural
  backgrounds are no better than white; photometric randomisation helps 3D slightly (+0.04) and hurts photos.
- [M] Shortcut: backgrounds alone predict the real class at 0.129 (chance 0.05). The SAGE **source dataset** alone,
  learned from the parent partition, predicts DEV classes at macro-F1 0.238. DEV photos from a source that also
  supplied a parent of their class are recognised better (+0.04–0.06 class-macro recall, all arms alike).
- [I] Part of the "own background" gain is source/background matching, which the real test rewards. Synthetic-to-real
  claims must report a source-held-out view (planned: FINAL split by source match).

## B. Viewpoint shift (8 views × 4 conditions per parent)
| arm | input view only | NARROW ≤ 8° | V8 ≤ 30° (Stage 0) | WIDE ≤ 180° |
|---|---|---|---|---|
| U3D | 0.536 | 0.523 | 0.525 | 0.538 |
| TEX3D | 0.594 | 0.596 | 0.587 | **0.558** |
(replicate 0)
- [M] More or wider views add nothing for real recognition; with photo texture, wide/back views hurt (−0.036), because
  surfaces unseen in the photo keep SAM 3D colours. SAGE test photos are close-ups from the photographer's side.
- The SAGE-Extend 18-view rig was not re-rendered here: WIDE already covers its angular range.

## C. Disease-feature preservation (input view vs the parent photo, both on white; median over 60 assets)
| arm | DINOv2 cosine | fine-texture energy kept (HF ratio) | colour error ΔE | object IoU |
|---|---|---|---|---|
| U3D / GEO3D | 0.69 | 0.22 | 10.7 | 0.86 |
| HARM3D | 0.71 | 0.28 | 1.4 | 0.86 |
| TEX3D / TEXGEO3D | 0.84–0.85 | 0.35 | 3.6 | 0.85–0.86 |
- [M] SAM 3D assets keep ~¼ of the photo's high-frequency texture; even projecting the photo's own pixels onto the
  Gaussians keeps only ~⅓ (the splat is too coarse). Silhouette is preserved (IoU 0.86).
- [M] Class-discriminative separation (synthetic leave-one-parent-out macro-F1): PHOTO2D 0.62–0.65 vs U3D 0.51 vs
  TEX3D 0.50–0.52 — renders are also less separable among themselves.
- [B] Symptom-region visibility and lesion localisation need lesion masks or expert annotation (none in SAGE).
- Perceptual similarity is not biological correctness; it is used here only to locate where information is lost.

## D. Appearance harmonisation
| change | Δ macro-F1 (pooled, 3 reps) |
|---|---|
| photo texture projection: TEX3D − U3D | **+0.062** [+0.047, +0.077] |
| global Lab colour transfer: HARM3D − U3D | 0.000 [−0.007, +0.007] |
| generic IP2P edit lifted: GEN3D − U3D | −0.050 [−0.067, −0.034] |
| KB IP2P edit lifted: KBAPP3D − U3D | −0.157 |
| residual render gap: TEX3D − PHOTO2D | −0.058 [−0.071, −0.046] |
- [M] What helps is **lesion-level texture from the real photo**, not colour balance: harmonisation fixes the colour
  error (ΔE 10.7 → 1.4) and changes nothing. IP2P edits remove or distort real symptoms (generic edits often "heal"
  the leaf; KB edits recolour the whole organ — contact sheet `figures/stage1/edits_fixed_s0.jpg`).
- GaussianEditor outputs exist only for the 15 SAGE-Extend pilot assets and only E/F/K arms; not used here (would cover 5 classes).
- Harmonisation never used real test images or labels (parameters from each parent photo only).

## Summary of the gap
1. Texture resolution of the 3D asset (largest, measured: ¾ of fine texture lost; +0.062 recovered by projection).
2. Source/background style (+0.09–0.13 when the parent's own background is kept; partly a shortcut).
3. Synthetic edits that move images away from real symptoms (KB −0.25, generic −0.15 vs photos in 2D).
4. Not the colour shift, not the number of views.
Exception [M, small]: for the 4 geometry-symptom classes, 3D renders match photos (U3D − PHOTO2D +0.009) while they
lose 0.149 on appearance classes (interaction +0.158 [0.123, 0.192]) — see `GENERATOR_ABLATION_RESULTS.md`.
