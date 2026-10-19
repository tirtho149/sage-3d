# 00 — Repository Audit: SAGE-Extend

Audited 2026-10-12. Sources examined:

- GitHub `ushashi149/SAGE-Extend` @ `371d96b` (single commit, "Initial commit: SAGE-Extend pipeline for 3D diseased leaf generation"), cloned to `~/Desktop/Research/external/SAGE-Extend`.
- The live Nova working directory `/work/mech-ai-scratch/UshB/sage-extend` (listed via Globus; logs copied to `~/Desktop/Research/data/sage_extend_nova/`).
- The protocol PDF found in that directory (`att.Ld0V7v…pdf`, 13 pp., "Diagnosis Guided Optimization of SAM 3D Crop Disease Twins").

Every statement below is verified against the files or logs. Nothing was executed except reading files and diffing PLY contents.

---

## 1. Headline findings

| # | Finding | Evidence |
|---|---|---|
| F1 | **There is exactly one specimen, not 100–150.** The entire pipeline has been run on one 468×466 screenshot of a single mango leaf. | `make_leaf_mask.py` reads `Screenshot 2026-10-04 at 11.09.07 PM.png`; `leaf_input/` holds one image + one mask; no other inputs exist in the repo or in the Nova directory. |
| F2 | **SAM 3D works (one run).** SAM 3D Objects produced a Gaussian splat and a mesh on an A100 with seed 42. | `leaf_sam3d_16730616.log`: "Inference complete … leaf_splat.ply … leaf_mesh.obj … Done." Earlier V100 attempt (`16724119`) hit the time limit while loading weights. |
| F3 | **GaussianEditor has never run.** Its conda environment failed to install, and its input views were never generated. | `ge_env_install.log`: tiny-cuda-nn build → `metadata-generation-failed` → `CondaEnvException: Pip failed`. `colmap_gen_16732022.err`: `ModuleNotFoundError: No module named 'sam3d_objects.pipeline.gaussian'`. `leaf_colmap/images/` on Nova is empty. No `leaf_edited/` dir or `ge_leaf_edit_*.log` exists. |
| F4 | **The "anthracnose" edit is hand-written NumPy, not GaussianEditor and not knowledge-driven.** `edit_anthracnose.py` hard-codes 16 ellipses (position, size, angle, darkness), recolors the DC spherical-harmonic term to a brown/black ramp, and shifts those Gaussians −z by up to 4 mm-equivalent units. | Diff vs. original splat: 5,277 / 37,120 Gaussians changed; fields changed = `z, f_dc_0..2, scale_2` only. |
| F5 | **The existing figure mislabels the hand edit as GaussianEditor output.** `pipeline_comparison.png` panel 4 is titled "Gaussian Editor Output (Anthracnose Synthesis)". It is the output of `edit_anthracnose.py` rendered by its own orthographic NumPy rasterizer. | `make_comparison.py` loads `Modified/leaf_anthracnose.png` for that panel. (The committed PNG has a different layout than `make_comparison.py` produces, so it was finalized elsewhere, but the panel content is the same file.) This must be corrected before the figure is shown to anyone. |
| F6 | **No knowledge base, retrieval, or prompt generation exists.** The only text instruction anywhere is a hard-coded generic string in `submit_ge_edit.sh`: `"add brown circular fungal disease lesion spots to the leaf"`. | grep of all files. |
| F7 | **No evaluation code, metrics, held-out views, or ground truth exist.** | No such files. The only "ground truth" label in the repo refers to the input image itself (`make_comparison.py`: "Ground Truth (SAGE Input)"). |
| F8 | **The input leaf already has real anthracnose-like lesions; the edit ignores them.** The 16 synthetic lesions are placed at fixed coordinates unrelated to the real lesions visible in the input. | Visual inspection of `leaf_input/image.png` vs. `Modified/leaf_anthracnose.png`; lesion centres in `edit_anthracnose.py` are constants. |
| F9 | **The reconstructed leaf is close to planar.** Splat extents: x ∈ [−0.139, 0.136], y ∈ [−0.503, 0.502], z ∈ [−0.017, 0.027] (z std 0.0087 vs. y std 0.234). This single example contains essentially no 3D structure beyond a slight bend. | Computed from `leaf_output/leaf_splat.ply`. |
| F10 | **A different research protocol already exists.** The PDF on Nova proposes a *diagnosis-guided* design (Qwen VLM critic via vLLM, GaussianEditor + GaussCtrl baseline, CMA-ES then PPO/SAC, evaluated on untouched real SAGE test images). Its question differs from the master prompt's knowledge-grounded-reconstruction framing. | `att.Ld0V7v…pdf`, §1 "Executive decision", 26 sections. |

---

## 2. Component inventory

| Component | Status | Implementation | Notes |
|---|---|---|---|
| Input acquisition | Partial | Manual screenshot | Source image ID, license and SAGE class are not recorded. |
| Segmentation mask | Implemented, tested on 1 image | `make_leaf_mask.py`: HSV threshold + morphology + largest component | Thresholds are tuned to this image's blue-grey background; will not generalize to field images. Mask coverage 16.7%. Not SAM-based. |
| 3D reconstruction | **Implemented, ran once** | `run_leaf_sam3d.py` → `facebook/sam-3d-objects` (HF checkpoints present on Nova: `ss_generator`, `slat_generator`, `slat_decoder_gs`, `slat_decoder_gs_4`, `slat_decoder_mesh`, encoders), `Inference(..., compile=False)`, seed 42 | Output representation: **3D Gaussian splat** (37,120 Gaussians, SH degree 0 — only DC colour stored) **plus a separate mesh** (`.obj`, no texture; `with_texture_baking False`). One seed only. |
| Gaussian editing (GaussianEditor) | **Not working** | `submit_ge_edit.sh` → GaussianEditor `edit-n2n.yaml` (InstructPix2Pix-guided), 2,000 steps | Env install failed (tiny-cuda-nn). Camera views not generated. Never executed. |
| Pseudo-COLMAP view generation | **Broken** | `make_colmap_from_ply.py`: renders 24 orbit views of the splat itself and writes a COLMAP model | Import error (F3). Methodological note: the "training views" would be renders of SAM 3D's own output, so editing would be supervised only by self-renders. |
| Procedural editing | Implemented, ran | `edit_leaf_splat.py` (yellow recolor + 5 opacity holes), `edit_anthracnose.py` (16 lesions) | Hard-coded constants (`N=37120`, lesion coordinates, colours). Not parameterized, not knowledge-driven. |
| Rendering | Partial | `render_leaf.py` / inline renderer in `edit_anthracnose.py`: CPU orthographic top-down splatting; `render_mesh.py`: matplotlib mesh shading | **Single fixed top-down viewpoint** only. No perspective cameras, no multi-view rendering of edits. The spiky edges visible in the panel-4 render are artifacts of this simplified rasterizer (no depth-correct sorting along the view ray, no anti-aliasing). The proper gsplat renderer in SAM 3D is only called in the broken `make_colmap_from_ply.py`. |
| Knowledge base / retrieval | **Missing** | — | Related KB code exists in other repos (`tirtho149/PlantSwarm` and `tirtho149/PRESAGE` `pathome_kb` with per-crop `final_registry.json`; `tirtho149/GeoPlant` disease graphs). Not connected to this project and not yet audited for provenance. |
| Prompt / edit-instruction builder | **Missing** | — | One hard-coded prompt string. |
| Evaluation / metrics | **Missing** | — | |
| Multi-view or 3D ground truth | **Missing** | — | |
| Figures | Partial | `make_comparison.py` (4-panel PIL composite) | Paths hard-coded; mislabeling (F5). |
| Config / reproducibility | **Missing** | — | All paths are absolute Nova paths under `/work/mech-ai-scratch/UshB/`. No configs, no environment files for this repo (relies on two conda envs in `/work/mech-ai/ushashi/AFM3d/miniconda3`). |

## 3. Input/output specification (as implemented)

```
Screenshot.png (468×466 RGB)
  └─ make_leaf_mask.py ──► leaf_input/image.png, leaf_input/0.png (binary mask)
       └─ run_leaf_sam3d.py (SAM 3D Objects, seed 42, A100) ──► leaf_output/leaf_splat.ply (37,120 GS, DC colour)
                                                              └► leaf_output/leaf_mesh.obj (untextured)
            ├─ edit_leaf_splat.py   ──► Modified/leaf_yellow_holes.ply  (all GS recoloured; 5 holes via opacity=-10)
            ├─ edit_anthracnose.py  ──► Modified/leaf_anthracnose.ply + .png (top-down ortho render)
            ├─ render_leaf.py       ──► Modified/leaf_render(.png|_2x.png)  (renders yellow_holes)
            ├─ render_mesh.py       ──► leaf_output/leaf_mesh.png
            └─ make_colmap_from_ply.py ✗ (import error) ──► leaf_colmap/ (empty)
                 └─ submit_ge_edit.sh ✗ (env not installed; never run)
make_comparison.py ──► pipeline_comparison.png
```

PLY layout (all three files identical header): `x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity scale_0..2 rot_0..3`, float32, little-endian.

## 4. Hardware and dependencies

- **SAM 3D needs Nova.** It ran on an A100 80 GB (Nova `scavenger` partition). A V100 32 GB job timed out while loading weights (2 h limit). This workstation has 2× GTX TITAN X (Maxwell, sm_52, 12 GB). It cannot run SAM 3D's sparse-conv / attention stack and is not a realistic target.
- **SAM 3D env:** `sam3d-objects` conda env (Python 3.11, spconv, warp 1.10, MoGe) on Nova — working.
- **GaussianEditor env:** failed at the `tiny-cuda-nn` build. Typical cause: missing/mismatched `nvcc` or `TCNN_CUDA_ARCHITECTURES` at build time; the exact message is in `ge_env_install.log` lines ~3560–3600. Needs to be built on a GPU node with a matching CUDA toolkit module loaded.
- **Gated model access:** SAM 3D checkpoints are HF-gated (`facebook/sam-3d-objects`); already downloaded on Nova.

## 5. Existing experimental results

None that can be reported as results. The available artefacts are one reconstruction and two procedural edits of it, with no measurements.

## 6. Reusable pieces

- `run_leaf_sam3d.py`: a working SAM 3D call. It can be generalized to a batch runner with a manifest, multiple seeds, and saved metadata.
- The SAM 3D `render_utils.render_frames` + `utils3d` camera code in `make_colmap_from_ply.py`: correct approach for multi-view perspective rendering once the `GaussianModel` import is fixed. The import path needs to be located in the installed `sam3d_objects` package.
- The PLY read/write logic: should become one shared loader instead of three copies with `N=37120` hard-coded.
- `pathome_kb` (PlantSwarm/PRESAGE) and GeoPlant disease graphs: candidate KB sources, pending a provenance audit.
- The protocol PDF: already contains a leakage-control, ablation, and statistics design. The two protocols should be reconciled rather than run in parallel.

## 7. Claims that cannot currently be supported

1. That GaussianEditor (or any learned 3D editor) was used to generate a diseased leaf. It was not (F3, F4, F5).
2. That edits are knowledge-grounded or source-grounded. No KB exists (F6).
3. That the pipeline has been applied to ~100–150 examples. It has been applied to one (F1).
4. Any statement about reconstruction accuracy, biological fidelity, multi-view consistency, or downstream utility. There is no ground truth and no metric (F7).
5. That 3D adds information for this example. The reconstruction is nearly planar (F9), and anthracnose leaf spot is primarily a surface-appearance symptom that a single top view already shows.

## 8. What is needed before any of H1–H5 can be tested

| Need | Blocks | Who/what |
|---|---|---|
| Real **multi-view photographs of the same specimens** (Tier 1), with one held-out-view protocol | H1, H3, H4, E1, E3, E4 | Physical data collection, or an existing public multi-view plant dataset (to be searched in Phase 1). Cannot be substituted with generated views. |
| Specimens whose symptoms are **actually 3D-dependent** (leaf curl/roll, galls, cankers, underside symptoms) | H4, H5, and the advisor's core question | Phase 1 case selection. The current anthracnose leaf is a poor test case for 3D value. |
| A working **learned 3D editor** (GaussianEditor fixed, or an alternative such as GaussCtrl / DGE) | H2, H3, E2–E4 | Rebuild env on a Nova GPU node. |
| A **working 2D editor baseline** with identical knowledge input (arms B, C) | H2, H3, H4 | e.g. InstructPix2Pix / an image-editing diffusion model on Nova. |
| A **structured, provenance-tracked KB** | H2, E2 | Phase 3. Audit `pathome_kb` first. |
| A **multi-view renderer** for the splat | Everything after E1 | Fix `make_colmap_from_ply.py` import. |
| **Expert annotators** for blinded biological review | E2, E3 | Advisor / plant pathology collaborator. |
| A decision on which protocol governs (master prompt vs. existing PDF) | All | PI decision. |

## 9. Known limitations of the current code (for the record)

- Absolute Nova paths everywhere; scripts are not portable.
- `N_VERTS = 37120` hard-coded in three files; any new specimen breaks them.
- Single seed for reconstruction; no record of stochastic variation.
- The HSV mask will fail on field backgrounds (soil, other leaves).
- The custom renderer is top-down orthographic only and is not a faithful 3DGS renderer.
- The "sunken lesion" geometry edit is a fixed −z offset in the splat's local frame, not along the surface normal. It is invisible from the only rendered viewpoint (top-down).

---

## 10. Addendum: the SAGE dataset as data and KB source

Source: <https://sage-dataset.github.io/>, HF `tirtho149/SAGE` (CC-BY-4.0, 48 parquet shards, 20.5 GB). Metadata copied to `~/Desktop/Research/data/sage_meta/` (`sage_metadata.parquet`, `canonical_mapping.json`, `README.md`). Images not yet downloaded.

**Verified counts (2026-10-12).** Built from all 48 image shards with `scripts/sage_manifest.py` → `data/metadata/sage_shard_manifest.parquet`; non-image columns plus parquet null statistics.

| Quantity | Image shards (authoritative) | Standalone `sage_metadata.parquet` |
|---|---|---|
| Rows | 820,502 (0 null images) | 820,504 |
| (crop, disease) classes | 1,251 | 1,251 |
| Rows with `canonical_disease` | **801,807** (= README's "801,807 images") | — |
| Rows with `visual_symptoms` | **631,869** | 422,526 |
| … with symptoms + quote + URL | 631,842 | 421,989 |
| Classes / crops with symptoms | **635 / 325** | 236 / 65 |
| Duplicate (crop, disease, filename) rows | 16,973 | — |

- **Resolution of the 453K vs. 613K question:** the standalone `sage_metadata.parquet` is a **stale, earlier KB fill**. The earlier figures in this audit (452,918 rows / 241 classes) came from it and are superseded. The image shards carry the current KB fill. The README's "613,174" matches neither file and is probably a stale count. The authoritative figure is **631,869 rows with source-grounded symptoms**, of which 631,842 have the full text + quote + URL triple. All 820,502 shard rows have image bytes.
- `plant_organ` is a coarse, disease-level list (e.g. "Foliar, Stem"), not a per-image label.
- Columns: `crop, disease, canonical_disease, plant_organ, visual_symptoms, pathogen, disease_type, symptom_source (URL), symptom_quote, source, filename, raw_label, image`.
- Image sources are aggregated public benchmarks (RadyPlantDis, CDDM, LeafNet, PlantVillage, Soybean_PNAS, Bugwood, …). **Every image is a single, independent view.** No specimen ID, camera ID, or multi-view grouping exists.
- The keyword screen below was run on the stale metadata file. It is re-run on the shard manifest in Phase 1 (`reports/01_case_selection.md`).

**What SAGE provides for this protocol:**

| Need from master prompt | SAGE coverage |
|---|---|
| Phase 3 structured, source-grounded KB (symptom text, organ, pathogen, source URL, verbatim quote) | **Largely provided.** Expert-audited (69–91% agreement per the paper). Lacks the structured *geometric* fields the prompt asks for (deformation type, direction, distribution, allowed/prohibited edits). These would be derived from `visual_symptoms`/`symptom_quote` with provenance kept. |
| Tier 3 public single-view images | **Provided**, at scale. |
| Tier 1 real multi-view of the same specimen | **Not provided.** Still blocks H1, H3, H4 (E1, E3, E4 accuracy claims). |
| Tier 2 real 3D / measured traits | **Not provided.** |
| Downstream real test set (E5) | **Provided.** Real, never-synthesized SAGE test images support Application C (augmentation) and the diagnosis-reference design in the existing protocol PDF. |

**Keyword screen of KB symptom text for 3D-relevant symptoms.** These are regex hits on `visual_symptoms`, not verified case selections. Many hits are incidental mentions; this is input to Phase 1, not a result.

| Symptom class | KB classes hit | Images | Plausible geometry-dominant examples among hits |
|---|---|---|---|
| curl / roll | 13 | 15,457 | Peach Leaf Curl (222), Grape Leafroll (71), Pumpkin Mosaic (5,780) |
| gall / knot | 5 | 14,397 | Cherry/Plum Black Knot (40/19), Corn Smut (505) |
| canker | 15 | 12,939 | Citrus Canker (547), Cucumber Gummy Stem Blight (138), Oak Botryosphaeria canker (33) |
| underside | 33 | 116,240 | Tomato Late Blight, Squash/Pumpkin Downy Mildew, Corn Common Rust |
| twist / distort | 23 | 20,974 | Sugarcane Pokkah Boeng (683), Rose Rosette (229), Cassava Mosaic (843) |
| witches' broom / proliferation | 5 | 5,010 | Rose Rosette, Corn Crazy Top (136) |

**Consequence for the plan:** with SAGE alone, the project can rigorously test **H2** (KB vs. generic editing, judged by blinded experts) and **H5** (downstream utility on real held-out SAGE images). **H1, H3-accuracy, and H4 still require new multi-view captures** of real specimens. Consistency-only H3 can be measured without them but is not accuracy.
