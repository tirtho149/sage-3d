# 02 — What GaussianEditor actually optimises (brief v2 §4)

- **Code inspected:** buaacyw/GaussianEditor @9249f84
  - `configs/edit-n2n.yaml`
  - `threestudio/systems/GassuianEditor.py`, `threestudio/systems/GassuianEditorEdit.py`
  - `gaussiansplatting/scene/gaussian_model.py`, `gaussiansplatting/utils/general_utils.py`
- **Tools written:** `src/metrics/gaussian_diff.py`, `src/metrics/geom_attributes.py`, `src/metrics/view_metrics.py`
- **Outputs:** `results/ge_diagnosis/`, `results/phaseA/Peach__Leaf_Curl__0087/E/seed0/`
- **Figure:** `figures/paper/fig5_displacement_Peach__Leaf_Curl__0087.png`

## 1. Mechanism (from code)
| Component | edit-n2n as run (collaborator hyperparameters) |
|---|---|
| Trainable groups | xyz (lr = position_lr·spatial_scale·`gs_lr_scaler`=5, exponential decay to ·`gs_final_lr_scaler`=5), f_dc (feature lr·`color_lr_scaler`=5), opacity (·2), scaling (·2), rotation (·2). **All geometry parameters are trainable.** |
| Edit signal | `L = 10·L1(render, E_v) + 10·LPIPS(render, E_v)`, where E_v = InstructPix2Pix(render_v, original_v, prompt), recomputed for each view every `per_editing_step`=10 steps (guidance 7.5, image guidance 1.5) |
| Regularisers | Anchor losses to the initial state, weighted per "generation": λ_geo=50 on xyz and rotation, λ_scale=50, λ_opacity=50, λ_color=0. Original Gaussians (generation 0) carry weight `anchor_weight_init_g0`=0.05 |
| Densification | From step 100, every 100 steps, ≤1 % of points per step (`max_densify_percent`). **Pruning** removes rows with sigmoid(opacity) < 0.005 and keeps the survivors in order (verified exactly on Peach 0006) |
| Edit region | `local_edit=False` and no `seg_prompt`, so the whole asset is editable. LangSAM is not used |
| Precision | `16-mixed` (fp16 autocast with GradScaler) |

**Consequence.** Geometry can only change if the IP2P targets demand a different shape. IP2P recolours and, in our 2D tests, never changed a leaf's outline unless the mask let it (report 05). Its per-view outputs are also mutually inconsistent. The optimiser therefore fits an *inconsistent recolouring* target. In practice that means moving and re-orienting Gaussians a little in incoherent directions, thickening the sheet, and growing soft edges. Nothing in the objective produces a coherent deformation such as puckering.

## 2. Measured parameter changes (arm E, generic prompt, 200 steps)
Correspondence is prune-aware index matching: rows with initial sigmoid(opacity) < 0.005 are removed, then the remaining arrays are aligned block-wise. On 0006 this reproduces the GE output bit-for-bit; on 0063/0087, 99.9 % of matches are consecutive rows.

| | Peach 0006 | Peach 0063 | Peach 0087 |
|---|---|---|---|
| Gaussians in → out (matched / added / deleted) | 343,520 → 342,696 (342,696 / 0 / 824) | 182,880 → 182,657 (182,592 / 65 / 288) | 145,536 → 145,537 (145,536 / 1 / 0) |
| Displacement median / p95 (% of bbox diagonal) | **0 / 0** | 1.13 / 2.14 | 1.23 / 2.31 |
| Displacement coherence (k=16; i.i.d. = 0.25, smooth = 1) | — | 0.32 | 0.31 |
| Rotation change median / p95 (°) | 0 / 0 | 7.6 / 33 | 12.2 / 48 |
| \|Δ log-scale\| median | 0 | 0.027 | 0.033 |
| \|Δ opacity\| median / p95 | 0 / 0 | 0.0002 / 0.036 | 0.0003 / 0.031 |
| \|Δ RGB\| median | 0 | 0.18 | 0.15 |
| Sheet thickness (positions, local PCA) ratio | — | ×1.84 | ×2.27 |
| Classification | **no change (bug F-GE-2)** | geometric jitter (incoherent) | geometric jitter (incoherent) |

**KB-attribute detectors (0087, `geom_attributes.py`).** Out-of-plane displacement h = d·n is split by spatial scale.

| Arm | distortion (low band) | blister band | blisters | jitter (high band) | normal thickening |
|---|---|---|---|---|---|
| **E** | 0.0001 | 0.0007 | **0** | 0.0078 | ×1.04 |

**Render-space QC (0087, 18 rig views, `view_metrics.py`).** Grain ×**13.6**, edge fuzz ×**1.9**, ΔE 13.5, silhouette change 0.14.

The silhouette change comes from fuzzy edges and jitter, not from a shape change. In Figure 5 the displacement heatmap is uniform speckle and the geometry-only render is near-identical to D.

## 3. Verdict
- **E is appearance editing plus incoherent geometric noise**, not geometry editing.
- The "grainy, fuzzy" look in the interim report is explained quantitatively:
  - incoherent displacement of about 1.2 % of the leaf size, which is about 9× the spacing between neighbouring Gaussians;
  - a doubled sheet thickness;
  - 8–12° random re-orientations.
- **The 0006 "no visible change" is a separate bug (F-GE-2):** non-finite opacity leads to a NaN loss, so every fp16 step is skipped. It is fixed by sanitising the splat.

## 4. Can geometry be enabled inside GaussianEditor?
Not by tuning alone. Raising position learning rates or lowering λ_anchor_geo only amplifies the jitter, because the supervision has no consistent shape target. Making GE deform would require a geometric target, i.e. a different edit signal. We therefore:
1. keep GE as the appearance editor, with geometry frozen when it follows a geometry edit (arms K and H: all non-colour learning rates 0, densification and pruning off; `FROZEN_GEOMETRY` in `ge_config.py`);
2. add an explicit geometry editor driven by measurable KB operators (G2, report 03).

This matches the brief's preference: extend rather than replace, and use the simplest method that changes shape coherently.

## 5. Other findings
- `loss_anchor_color` is NaN in every run: the mean over the empty `_features_rest` with SH degree 0. It is benign, since λ = 0 and the gradients are finite, but it makes the logged total loss NaN. Loss curves therefore cannot certify a run; `verify.json` and parameter diffs do.
- GE writes two 57 MB checkpoints per run; the runner now deletes them after extracting `last.ply`.
