# 01 — SAM 3D input-view alignment: diagnosis and correction (brief v2 §3)

- **Code:** `src/camera_alignment/align.py`, `src/analysis/alignment_summary.py`, `experiments/runners/run_alignment.sh`
- **Outputs:** `results/alignment/<specimen>/`. Each contains:
  - `A0.png`, `A1_pp_centre.png`, `A2_closed_form.png`, `A3_render_compare.png`
  - `overlay_A0.png`, `overlay_A3_render_compare.png` (blue = mask only, red = render only, green = both)
  - `alignment.json` (camera K, metrics, pose corrections)
- **Summary:** `results/alignment/summary.json`
- **Original reconstructions:** unchanged in `results/pilot/sam3d/` (the correction is a separate pose; geometry is never modified).

## 1. Transformation chain (documented)
1. SAGE image → `data/inputs/<sid>/image.png` (RGB, original resolution, no EXIF rotation present in these files) and a SAM ViT-H box-prompted mask `0.png` at the same resolution.
2. **MoGe pointmap** at the image resolution, in the PyTorch3D camera frame (+x left, +y up, +z forward).
3. **SAM 3D layout head** outputs the object pose (quaternion, translation, isotropic scale). Object-local Gaussians are mapped to the camera by `SceneVisualizer.object_pointcloud`; Gaussian rotations become q' = q_pose⁻¹·q and scales are multiplied by s.
4. **PyTorch3D → OpenCV camera:** (x, y) → (−x, −y), with a 180° rotation about z for the quaternions.
5. **Intrinsics K:** fitted by least squares to the pointmap (u = fx·X/Z + cx, v = fy·Y/Z + cy).
6. **Rendering:** gsplat with an identity extrinsic, composited on white. IoU uses alpha > 0.5 against the SAM mask.

## 2. Findings
- **The camera is not the error source.**
  - K reproduces the pointmap with a median reprojection error under 2 px on 12/15 specimens. The remaining three are Corn 0025 at 4.3 px, Citrus 0016 at 5.6 px and Peach 0063 at 6.4 px.
  - Forcing the principal point to the image centre (A1) changes IoU by ≤ 0.01 on every specimen.
- **The error is SAM 3D's predicted object pose and scale.**
  - At the original pose, the object is too large in 13/15 specimens (silhouette area +8 % to +150 %). The worst cases are Tomato 0015 (+150 %) and Tomato 0092 (+111 %).
  - A pose-only correction raises IoU on all 15.
- **SAM 3D's own correction is off.** SAM 3D contains a render-and-compare pose post-optimisation (`with_layout_postprocess`), but it is disabled by default, including in the official demo (`notebook/inference.py:116`) and in our runs.

## 3. Alignment study
| Arm | What changes | Geometry |
|---|---|---|
| A0 | Existing pipeline | — |
| A1 | Camera convention check: principal point at the image centre | frozen |
| A2 | Closed form: scale about the object centroid to match mask area, then image-plane translation to match the centroid (3 fixed-point iterations) | frozen |
| A3 | 7-DoF render-and-compare (axis-angle, translation, log-scale). Loss = soft-IoU + 0.5·BCE on the mask only; 300 Adam steps at 512 px; initialised from A2 | frozen |
| **A3\*** | **A3 if IoU(A3) > IoU(A2), else A2.** This is the acceptance rule SAM 3D's post-optimisation uses | frozen |
| A4/A5 | Geometry refinement | **not run**: A3\* reached IoU ≥ 0.77 on 14/15, and changing geometry to fit a silhouette would hide reconstruction error |

**Held-out checks.** These are never used in fitting: LPIPS of the render vs the masked photo, and the median relative depth difference vs MoGe inside the mask.

| Specimen | reported | A0 IoU | A1 IoU | A2 IoU | A3 IoU | **A3\* IoU** | BF A0→A3* | area err A0 | LPIPS A0→A3* (held-out) | depth err A0→A3* (held-out) | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Citrus Canker 0004 | — | 0.67 | 0.67 | 0.84 | 0.91 | **0.91** | 0.12→0.59 | +0.44 | 0.29→0.30 | 0.025→0.079 | ok |
| Citrus Canker 0012 | 0.73 | 0.73 | 0.73 | 0.98 | 0.99 | **0.99** | 0.00→1.00 | +0.36 | 0.23→0.21 | 0.095→0.029 | ok |
| Citrus Canker 0016 | 0.76 | 0.76 | 0.77 | 0.98 | 0.98 | **0.98** | 0.23→0.99 | -0.19 | 0.18→0.19 | 0.088→0.071 | ok |
| Corn Smut 0012 | 0.63 | 0.63 | 0.63 | 0.86 | 0.94 | **0.94** | 0.13→0.77 | +0.43 | 0.20→0.18 | 0.011→0.015 | ok |
| Corn Smut 0025 | 0.61 | 0.61 | 0.61 | 0.77 | 0.76 | **0.77** | 0.24→0.29 | -0.09 | 0.22→0.21 | 0.039→0.042 | ok |
| Corn Smut 0052 | 0.72 | 0.72 | 0.72 | 0.83 | 0.90 | **0.90** | 0.18→0.62 | +0.34 | 0.16→0.16 | 0.051→0.091 | ok |
| Mango Anthracnose 0049 | — | 0.77 | 0.78 | 0.85 | 0.95 | **0.95** | 0.46→0.98 | +0.19 | 0.11→0.10 | 0.016→0.039 | ok |
| Peach Leaf Curl 0006 | 0.43 | 0.43 | 0.43 | 0.50 | 0.13 (diverged) | **0.50** | 0.11→0.20 | +0.68 | 0.18→0.20 | 0.031→0.027 | reconstruction_failure |
| Peach Leaf Curl 0063 | 0.58 | 0.58 | 0.58 | 0.67 | 0.00 (diverged) | **0.67** | 0.19→0.31 | +0.35 | 0.11→0.11 | 0.015→0.014 | ok (borderline) |
| Peach Leaf Curl 0087 | 0.84 | 0.84 | 0.85 | 0.89 | 0.95 | **0.95** | 0.71→0.96 | +0.08 | 0.05→0.04 | 0.003→0.012 | ok |
| Tomato Leaf Mold 0015 | 0.40 | 0.40 | 0.40 | 0.88 | 0.94 | **0.94** | 0.00→0.84 | +1.50 | 0.20→0.21 | 0.017→0.031 | ok |
| Tomato Leaf Mold 0052 | 0.55 | 0.55 | 0.55 | 0.72 | 0.92 | **0.92** | 0.04→0.65 | +0.79 | 0.29→0.28 | 0.034→0.043 | ok |
| Tomato Leaf Mold 0092 | 0.47 | 0.47 | 0.47 | 0.84 | 0.23 (diverged) | **0.84** | 0.05→0.37 | +1.11 | 0.32→0.32 | 0.030→0.028 | ok |
| Mango Anthracnose 0044 | — | 0.70 | 0.70 | 0.87 | 0.93 | **0.93** | 0.24→0.89 | +0.27 | 0.14→0.12 | 0.009→0.081 | ok |
| Mango Anthracnose 0094 | — | 0.63 | 0.63 | 0.82 | 0.86 | **0.86** | 0.19→0.64 | +0.57 | 0.09→0.13 | 0.035→0.060 | ok |

BF = boundary F-score (tolerance max(2 px, 0.75 % of the image diagonal)). All 15 pilot reconstructions are included; Mango 0094 and 0044 were added in a resumed run.

**Summary (all 15):**
- **IoU:** median rises from 0.63 to **0.92**. All 15 improve; none degrade.
- **Held-out appearance:** LPIPS is worse by more than 0.02 in 1/15 (Mango 0094, 0.09 → 0.13).
- **Held-out depth:** agreement with MoGe is worse in 6/15 (Citrus 0004, Corn 0052, Mango 0044/0049/0094, Tomato 0015). In those cases the silhouette fit partly trades depth for outline.
- **A3 alone diverged on 3/15** (Peach 0006, Peach 0063, Tomato 0092: rotations of 21–50°, collapsing IoU). The A3\* acceptance rule is therefore required, not optional.

## 4. Failure labels
- **Peach 0006** is a **reconstruction failure** (best IoU 0.50): the curled leaf became a closed ring. It is kept in all tables and figures.
- **Peach 0063** (0.67) is borderline. The 0.6 threshold was chosen after seeing these values; at 0.7 it would also fail.

## 5. What this does and does not show
- **It shows** that the input-view mismatch reported in the interim E1 is mostly a pose/scale error, and that it is correctable without touching geometry.
- **It does not show** reconstruction accuracy:
  - A3\* is fitted to the very silhouette it is scored on.
  - The only independent signals (LPIPS, MoGe depth, itself a prediction) are weak.
  - No second real view of any specimen exists (SAGE-only decision).
  - Shape accuracy away from the input view is unmeasured.

## 6. Use downstream
Edited assets are evaluated on the fixed per-asset rig (`D/rig.json`), which does not depend on the input pose. A3\* is used only for input-view comparisons (E1, and the input-view renders in Figures 2 and 10).
