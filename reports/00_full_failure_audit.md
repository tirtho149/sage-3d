# 00 — Full failure audit (brief v2, Part I)

Factual record:
- **Interim report:** the Overleaf project "testtrial" (= `testtrial.pdf`), generated from `results/pilot/**`.
- **Machine:** local beast0, 2× TITAN X (Maxwell). No Nova/Slurm (user decision), so run IDs are PIDs plus JSONL logs.
- **Status labels:** CONFIRMED (cause shown with evidence) · FIXED (code changed) · TESTED (fix verified on a real artifact) · UNRESOLVED · BLOCKED.

## Verification of the reported E1 numbers
All 11 reported IoUs are reproduced exactly by the A0 render (`results/alignment/*/alignment.json` → `metrics.A0.iou`). The values are 0.73, 0.76, 0.63, 0.61, 0.72, 0.43, 0.58, 0.84, 0.40, 0.55, 0.47. Citrus 0004 (0.67) and Mango 0049 (0.77) finished after the report.

---

### F-GE-1 — Arm F never ran (KB prompt containing ":")
| Field | Content |
|---|---|
| Code | Original `src/gaussian_editing/run_ge.sh` (kept as `run_ge.sh.v1_cli_quoting.bak`): `system.prompt_processor.prompt="$3"` passed as a CLI dotlist |
| Root cause | **CONFIRMED.** threestudio → `OmegaConf.from_cli` parses `a: b` as a YAML mapping. `tag: ${rmspace:${system.prompt_processor.prompt},_}` then calls `.replace` on a dict, so it fails before model load. All KB prompts start `make this … show …:` |
| Evidence | `results/pilot/Peach__Leaf_Curl__0087/log.txt`: `InterpolationResolutionError … Missing key replace`. `runner_gpu0.jsonl`: GE_F rc=1 after 5.6–5.7 s on 0006/0063/0087 |
| Reproduction | `tests/test_ge_config.py::test_old_cli_path_breaks` |
| Modification | Prompts no longer pass through the CLI. `src/gaussian_editing/ge_config.py` writes a full per-run YAML (base config + collaborator hyperparameters + prompt/seed/steps/paths). `check` reloads it exactly as threestudio does and requires the prompt to round-trip byte-identically. Prompts are NFC- and whitespace-normalised. `${` is rejected, because threestudio's `OmegaConf.resolve()` turns escaped `\${` back into a live interpolation (found by the tests). `verify` compares the prompt in GE's own `configs/parsed.yaml` with the intended prompt and requires `save/last.ply`. `run_ge.sh` v2 exits non-zero unless both hold. The CLIP token count uses the IP2P tokenizer that GE uses |
| Verification | `tests/test_ge_config.py`: 10 passed (colon, quotes, newline/tab, braces/brackets/%, YAML specials, Unicode en-dash/accents/½, real KB prompt, empty → rejected, `${` → rejected, old path breaks). **Live run, Tomato 0015 arm F** (`results/pilot/runner_gpu0.jsonl`, 17:51): rc 0, 2636.6 s, 200 steps. `verify.json`: `prompt_delivered: true`; the parsed prompt is byte-identical to the KB text, including ":", ";", "," and "1/4"; `last_ply: true` |
| Status | **FIXED; TESTED (unit + live run)** |
| Residual risk | A prompt with `${` cannot be delivered (rejected loudly). The 77-token CLIP truncation is checked only when the tokenizer is cached locally (it is) |

### F-GE-2 — Arm E on Peach 0006 returned the input unchanged
| Field | Content |
|---|---|
| Code | SAM 3D export (`splat.ply`) → GaussianEditor `gaussian_model.anchor_loss` → Lightning `precision: 16-mixed` (configs/edit-n2n.yaml:77) |
| Root cause | **CONFIRMED.** SAM 3D writes opacity logit `+inf` where opacity == 1.0: 3 Gaussians in Peach 0006; 112 / 2 / 607 in Corn 0012 / 0025 / 0052. GE's opacity anchor loss (inf − inf) becomes NaN, so gradients are non-finite. The fp16 GradScaler skips every optimizer step for all 200 steps. Pruning runs outside the optimizer, so only low-opacity rows were removed |
| Evidence | `results/ge_diagnosis/Peach__Leaf_Curl__0006_E_diff.json`: every surviving Gaussian is bit-identical. The output equals the input minus its 824 rows with sigmoid(opacity) < 0.005 (checked row by row). GE `metrics.csv`: `loss_anchor_opacity = nan` for 0006 only |
| Modification | `src/gaussian_editing/sanitize_splat.py` clamps non-finite opacity logits to ±15.9424 (= logit(1−2⁻²³), the value SAM 3D already uses for saturated Gaussians; render-identical). Any other non-finite field raises an error. The output goes to `splat_ge.ply`; the original is kept. `pilot_runner.py` now always edits `splat_ge.ply`. The invalid output was moved to `Peach__Leaf_Curl__0006/E_INVALID_nan_opacity_noop/` |
| Verification | Sanitised counts are logged in `results/pilot/sam3d/*/seed0/splat_ge.ply.json`. **Live re-run, 0006 arm E** (GPU 1, 17:5x): `loss_anchor_opacity` is finite (0 → 4e-4; it was NaN). Parameters now change: median displacement 1.11 % of the diagonal, 864 pruned and 135 added (`results/ge_diagnosis/Peach__Leaf_Curl__0006_E_rerun_diff.json`). The outcome is the same incoherent-jitter class as 0063/0087. `verify.json`: prompt delivered, asset saved |
| Status | **FIXED; TESTED (live run)** |
| Residual risk | Corn E/F would have silently no-op'd without this fix. A future SAM 3D export with NaN would now fail loudly instead |

### F-GE-3 — Arm E "edits" are grainy/fuzzy, with no shape change
| Field | Content |
|---|---|
| Code | `threestudio/systems/GassuianEditorEdit.py::training_step`: loss = λ_l1·L1 + λ_p·LPIPS against InstructPix2Pix-edited training views (re-edited every 10 steps), plus anchor losses. All Gaussian parameters are trainable |
| Root cause | **CONFIRMED.** The only edit signal is per-view IP2P images. IP2P recolours and does not change shape (see F-2D-1), so nothing pushes coherent geometry. Each view's IP2P edit is independent, so the targets disagree across views. The optimizer fits these inconsistent targets by jittering Gaussians and thickening the sheet |
| Evidence | Peach 0063 / 0087 (`results/ge_diagnosis/*_E_diff.json`, prune-aware index matching with 99.9 % consecutive matches): median displacement 1.1 / 1.2 % of bbox diagonal. Displacement is incoherent: k-NN coherence 0.32 / 0.31, vs 0.25 for i.i.d. noise and ≈1 for a smooth deformation. Median rotation change 8–12°. Sheet thickness ×1.8 / ×2.3. Render-space QC on 0087 (`results/phaseA/Peach__Leaf_Curl__0087/E/seed0/view_metrics.json`): grain ×13.6, edge fuzz ×1.9, ΔE 13.5. KB attribute detectors: distortion 0.000, blister band 0.0003, thickening ×1.01 |
| Modification | None to GE: this is its mechanism, not a bug. Geometry is handled by a separate geometry editor (G2, `src/geometry_editor/deform.py`, report 03). Appearance-only 3D editing (arm K) runs GE with geometry frozen (position/scale/rotation lr ×1e-8, densification off) |
| Status | **CONFIRMED; addressed by design change (arms G/H/J/K)** |
| Residual risk | GE colour edits remain multi-view-inconsistent at the texture level |

### F-E1-1 — Low and variable E1 IoU (0.40–0.84)
| Field | Content |
|---|---|
| Code | `src/rendering/input_view.py` (render at the input camera) and the SAM 3D layout head (pose: rotation, translation, scale) |
| Root cause | **CONFIRMED: object pose/scale error from SAM 3D's layout prediction, not the camera.** (i) The fitted K reproduces MoGe's pointmap to a median of 0.27–1.65 px on 10/13 specimens (4.3–6.4 px on Corn 0025, Citrus 0016, Peach 0063). (ii) Moving the principal point to the image centre (A1) changes IoU by ≤ 0.01. (iii) A pose-only correction raises IoU on every specimen. Tomato 0015: the object was ~1.7× too large (A3 log-scale −0.55). SAM 3D ships a pose post-optimisation (`with_layout_postprocess`) that is **off by default** in its own demo (`notebook/inference.py:116`) and in our runs |
| Evidence | `results/alignment/*/alignment.json`, `reports/01_camera_alignment_fixes.md` |
| Modification | `src/camera_alignment/align.py`: A2 (closed-form scale about the centroid + image-plane translation) and A3 (7-DoF render-and-compare with geometry frozen, mask loss only). Corrected pose = A3 if it beats A2 on IoU, else A2 (A3*), the same acceptance rule SAM 3D's post-optimisation uses. RGB/LPIPS and MoGe depth agreement are held-out checks (never optimised) |
| Status | **FIXED (A2/A3*); TESTED on all 13 reconstructions.** A3 alone diverged on 3/13 (Peach 0006, Peach 0063, Tomato 0092) |
| Residual risk | Silhouette agreement after fitting is **not** reconstruction accuracy. A3 sometimes worsens held-out depth agreement (e.g. Corn 0052 0.05 → 0.09), i.e. it partly trades depth for silhouette. There is no independent view to evaluate on |

### F-E1-2 — Peach 0006 is a reconstruction failure
**CONFIRMED.** The curled leaf is reconstructed as a closed ring. Even the best correction gives IoU 0.50 (A2), and A3 diverged to 0.13. The specimen is labelled `reconstruction_failure` and kept in every table and figure; it is not dropped.

### F-2D-1 — 2D edits only recolour
| Field | Content |
|---|---|
| Code | `src/editing2d/ip2p_core.py::edit`: the output is composited **inside the original organ mask** |
| Root cause | **CONFIRMED (structural).** The compositing step makes any silhouette change impossible; IP2P also tends to recolour. So the 2D baselines B/C could not represent shape by construction, which makes a 2D-vs-3D shape comparison unfair |
| Modification | Arm I: same KB prompt, composite mask dilated by 15 % of the organ diameter, image guidance 1.2 (`dilate`, `igs` job fields). Stage-consistent C (`C_stage`) added for a like-for-like prompt |
| Status | **FIXED (baseline added); results pending (Phase A, CPU)** |

### F-2D-2 — Per-view 2D baseline (Bmv/Cmv) was run unmasked
| Field | Content |
|---|---|
| Code | `experiments/runners/pilot_runner.py` (Bmv/Cmv jobs: `mask: None`) |
| Root cause | **CONFIRMED.** Without a mask, IP2P repainted the white background of each render into whole new leaf scenes, so the baseline was unfairly bad. Evidence: `results/pilot/*/Bmv/*.png` |
| Modification | `src/editing2d/mask_perview.py` composites each existing output inside the D render's alpha (> 0.5), giving `{Bmv,Cmv}_masked/`. `ip2p_core.edit` runs IP2P on the full image and then composites inside the mask, so this equals a masked run with the same seed (no re-run needed). Metrics and Figure 7 use the masked versions |
| Verification | The correspondence-based cross-view metric samples object points only, so it is unchanged (Peach 0087: Bmv 21.7, Cmv 32.9). The figure now shows object-only edits |
| Status | **FIXED; TESTED** |

### F-EVAL-1 — Mean-colour cross-view consistency was uninformative
**CONFIRMED, replaced.** The first metric (spread of per-view mean Lab shift) scored per-view 2D edits (0.79/0.72) about as consistent as the GE 3D edit (0.71), because it ignores point correspondence. It was replaced by `src/metrics/crossview.py`: projected surface points visible in adjacent views (depth-tested), with the median ΔE of the same point. On Peach 0087: floor (D) 2.1, Bmv 21.7, Cmv 32.9, E 7.4, J 2.0, G 2.0.

### F-KB-1 — KB prompts mix disease stages
**CONFIRMED.** Example, Peach: "distorted (thickened and puckered) … reddish or purple cast; later … powdery gray". **FIXED:** `src/kb_grounding/structured.py` → `data/kb/structured_specs.json`. It uses stage segmentation by temporal markers, one stage per edit, verbatim phrases only (asserted), and hedged phrases marked `uncertain` and excluded. Other-stage features are listed as `prohibited`. Each operator and attribute keeps its source phrase. Findings:
- Citrus-fruit crater depression is hedged ("may manifest"), so it is not edited.
- Tomato Leaf Mold has no geometric claim, so geometry arms abstain.
- "Curl" in "Peach Leaf Curl" occurs only in the disease name, not in the quote, so no curl operator is used.

### F-GE-4 — `loss_anchor_color` is NaN in every GE run
**CONFIRMED, benign, not fixed.** It is the mean over the empty `_features_rest` tensor (SH degree 0). λ_anchor_color = 0, so gradients are 0 (finite) and steps proceed, as 0063/0087 show. The logged total loss is NaN, though, so loss curves cannot be used as a success signal. Run success is checked by `verify` and parameter diffs instead.

### F-GEO-1 — First pucker operator produced a uniform lift, not blisters
**CONFIRMED and FIXED** by `tests/test_geometry_ops.py`. About 50 overlapping RBFs at spacing 2σ sum to a near-constant offset. It now uses Poisson-disk centres (≥ 3σ apart) and a zero-mean field. The same tests found a voxel-grid padding bug in the attribute detector (half the samples read as 0) and a thickness measure taken along the stale normal. Both are fixed, and all 4 operator tests pass. Specimen outputs made before the fix were deleted and regenerated.

### F-OPS-1 — Disk exhaustion risk
**FIXED.** GE kept 2×57 MB checkpoints plus a duplicate PLY per run; the runner now prunes them after copying `last.ply`.

### F-VAL-1 — Novel views cannot be verified
**BLOCKED** (user decision: SAGE only, no multi-view data). Every novel view is labelled *synthetic, unverified*. Internal multi-view consistency is reported; external validity is not claimed.

### F-VAL-2 — Biological validity
**BLOCKED.** No expert raters. Technical QC is reported and kept separate from biological validity.
