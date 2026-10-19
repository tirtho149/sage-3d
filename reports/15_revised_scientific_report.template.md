# 15 — Revised scientific report: what does KB-guided 3D add beyond 2D?

_SAGE-Extend, 2026-10-14. Continues the experimental record in `SAGE_3D.pdf` (advisor deck of 2026-10-12, 30 pages).
Numbers in the tables are inserted from `results/demos/summary.md`, `results/statistics/comparisons.json` and
`results/downstream_minimal/summary.json` by `src/analysis/assemble_report15.py`; no number is typed by hand._

## 1. The question and the direct answer

**Question (S. Sarkar):** what scientifically meaningful information does 3D add beyond the original 2D image or a
knowledge-guided (KB) 2D edit?

**Answer from the measurements.** KB-guided 3D can *represent* source-grounded facts that a single image cannot
express: **which surface** a symptom is on (Tomato Leaf Mold: lower-surface mold beneath upper-surface spots), **which
direction** an organ deforms (a cited "curl upwards" is reproduced, a reversed control reverses), and **how much
volume** a symptom has and what tissue it replaces (Corn Smut galls as closed bodies on kernel tissue). These
capabilities are measured, with placement or direction controls, from fixed cameras, without contradicting the
observed photo where the edit is meant to be hidden. What is **not** shown: that the represented hidden structure is
*correct* for the specimen (no matched real views exist; none were found publicly), that it is *biologically
realistic* (no expert ratings yet; the package is ready), or that it is *useful* (the minimal downstream test on
independent real photos found no benefit). Hidden symptoms are represented, not recovered.

## 2. What changed since the 2026-10-12 record

| Item (brief v3) | Status | Evidence |
|---|---|---|
| 1. Pending E/F/K/H, regenerate | done; E/F 15/15, K/H 15/15; J/G 9/15 (6 correct abstentions) | results/statistics/*, regenerated 2026-10-14 |
| 2. Directional curl/roll operator + curvature + local preservation | done; tested on synthetic sheets | src/geometry_editor/curl.py, src/metrics/structure3d.py, tests/test_v2_operators.py |
| 3. Separate upper/lower surfaces | done | src/geometry_editor/bilayer.py, src/metrics/surfaces.py |
| 4. Evidence-constrained placement | done (pucker, galls, mold); uncertainty logged | src/geometry_editor/placement.py, data/kb/supplementary_evidence.json |
| 5. Three focused demos | done (Peach, Corn, Tomato; + Mango direction control) | experiments/demos/run_demos.py, results/demos/ |
| 6. 2D vs 3D comparisons | done (KB 2D, generic 3D, KB geometry-aware 3D) | results/demos/summary.md |
| 7. Expert-review package | built; not yet sent | reports/13, results/review_package_v1/ |
| 8. Real multi-view data search + protocol | none suitable; protocol for 30 specimens | reports/11, reports/12 |
| 9. Minimal downstream real-data test | done; NOT SUPPORTED at this scale | reports/14 |
| 10. Three advisor figures | done | figures/slides/demo_{a_directional,b_galls,c_underside}.pdf |
| 11. Structural / external measures | done (see each test) | this report |
| 12. Deck, report, logs, blockers | done | Overleaf 6ac80070; this file; section 9 |

## 3. Phase B completion and regenerated statistics

Unit = specimen; paired Wilcoxon; Holm across the comparison family. Artefact measures now pass Holm with 9-15
specimens; KB recall does not (n <= 9, <= 3 classes).

{{STATS_TABLE}}

The 2x2 knowledge x geometry ablation now has every cell on {{AB_N}} specimens: interaction {{AB_INT}}.

## 4. Methods added in this session

**Anatomy (anatomy.py).** The A3*-corrected input camera is expressed in object coordinates (Umeyama residual
~1e-7 L). The camera-facing side is labelled *adaxial* (an assumption; an unpinned sign propagation keeps {{ORIENT}} of
visible Gaussians camera-facing, i.e. the sheets are consistently orientable). A landmark-Isomap flattening (stress
{{STRESS}}) gives a midrib line and a lateral coordinate; tip vs base is not inferred. Ears: principal axis, radial
normals, kernel / existing-gall tissue from Lab clusters of the photo's colours; if no yellow kernel cluster exists
(b* < 30) the gall operator abstains.

**Directional curl (curl.py).** Isometric bend about the midrib; margins turn by theta toward the adaxial (+) or
abaxial (-) side; direction parsed from the KB phrase only ("upwards", "downward", "rolled under", "cupping");
without a direction the operator abstains. Magnitude is not in any source (defaults: cup 45, curl 90, roll 150 deg;
reported as assumptions). Optimised with fine and coarse-scale ARAP. A unit test on a flat sheet checks isometry
(strain p95 < 2%), direction, and the expected 45 deg chord angle; it caught a smoothing defect that stretched the
sheet by 12%, fixed before the reported runs.

**Two surfaces (bilayer.py).** Each Gaussian becomes a thin upper and a thin lower Gaussian (offset 0.002 L along the
smoothed adaxial normal). Upper colour = the photo's colour; lower base = the same with upper lesions removed
(unknown abaxial colour: SYNTHETIC, unverified); lower-surface symptoms are written to the lower layer only.

**Volumetric galls (gall.py).** Lobed unions of 7-12 spheres attached at kernel surface points; surface Gaussians on
the union boundary; original Gaussians inside are hidden (the gall replaces tissue); colour from the photo's own gall
tissue, with baked Lambertian shading (rendering choice, not a biological claim).

**Evidence-constrained placement (placement.py; data/kb/supplementary_evidence.json).**

| Symptom | Constraint | Evidence | Logged uncertainty |
|---|---|---|---|
| Peach puckering | weight by the photo's reddish/purple tissue | Ohioline: puckering and reddish cast on the same developing leaves | tissue-level co-location is an inference |
| Peach curl | none: abstain | Ohioline page checked: **no** blade curl direction | direction unknown |
| Mango curl (control) | upward | SAGE (Plantix): "Leaves curl upwards" | magnitude assumed |
| Corn galls | on kernel tissue; 1-30 cm | UNL: "develop on the ear replacing kernels"; SAGE size range | ear length 18 cm assumed; count and along-ear position not in any source |
| Tomato mold | lower surface beneath the photo's spots | SAGE (UMN): "on the lower leaf surface below leaf spots" | abaxial colour unknown |

**Measures (structure3d.py, surfaces.py).** Roll angle and transverse curvature about the midrib; tissue strain on
point pairs 0.015-0.025 L apart (8-NN edges are at the noise scale: 0.002 L noise gives 620% 8-NN strain); relief
preservation as the correlation of residuals from local quadrics (identity = 1, removes the intended bend); per-pixel
surface identity in every render; mold/spot colour classes fixed before the edits. **External reference where one
exists:** silhouette IoU at the corrected input camera against the REAL photo's mask (an edit should not contradict
the photo). No measure here is a self-consistency score used as biological validity; placement fractions verify that
a constraint was applied, not that it is biologically right.

## 5. Results

All methods are rendered from the same 18 rig cameras and the same corrected input camera; 3 specimens per disease
(Corn: 2 + 1 abstention), seeds 0-2 averaged within specimen.

### Test 1 — direction (Mango direction control, Peach Leaf Curl)

{{MANGO_TABLE}}

{{PEACH_TABLE}}

Reading. With a cited direction, the operator reproduces it and the reversed control reverses, with tissue strain
comparable to the generic warp. On the Mango leaves the curl lowers input-view IoU: the photo shows a flat leaf, so
the edit is a counterfactual that contradicts the observation; it demonstrates control, not recovery. For Peach Leaf
Curl the source gives no direction, so no curl is claimed; the two hypotheses are shown only as labelled hypotheses.
The observed roll of the Peach reconstructions (from the photos) is reported for D. **Placement on Peach is a weak result:** weighting puckering by the photo's reddish tissue raised the share of blister relief on that tissue by only 0.05-0.12 (enrichment ratio) per specimen over the same operators placed uniformly. The colour map is relative to the greener half of each leaf (a fully red leaf, 0087, is flagged on 15% of its area) and the weights keep a floor of 0.25 for the unseen side, so the constraint is weak by construction; no localisation gain is claimed for Peach.

### Test 2 — volumetric galls (Corn Smut)

{{CORN_TABLE}}

Reading. Volume alone does not separate the methods: the v1 height field adds a similar volume. What separates them
is form and placement: a closed body that overhangs the ear (a height field cannot), centred on kernel tissue as the
source states (random sites: far less), and replacing the kernels it covers. Real galls in the photos (connected
components of gall tissue in the reconstruction) have longest extents {{REAL_GALLS}} cm under the same assumed scale; the KB range is
1-30 cm. Ear 0025 is all gall: no kernel tissue, so the operator abstained.

### Test 3 — upper vs lower surface (Tomato Leaf Mold)

{{TOMATO_TABLE}}

Reading. A photo shows the upper surface only. KB 2D edits, given text that names a lower-surface mold, put
mold-coloured pixels on that upper surface (the wrong side per the KB). Single-sheet 3D assets show the upper spots on
the underside (mirror error), and GaussianEditor paints both sides alike. The two-surface asset keeps the mold on the
lower surface, under the photo's real spots, with the upper surface at its no-symptom baseline, and a reversed
control (mold on the upper layer) reverses the pattern. This shows that the KB statement can be represented in 3D; the
true appearance of these leaves' undersides is unknown.

## 6. Minimal downstream test (reports/14)

{{DOWNSTREAM}}

## 7. Validation readiness

* **Expert review (reports/13; results/review_package_v1).** 84 blinded, shuffled items (72 test, 9 repeats, 3 catch),
  real photo beside each candidate, verbatim KB text, anchored 1-5 criteria (symptom presence, location, shape,
  surface, artefacts, overall) and a prohibited-feature flag; about 57 min per rater; Krippendorff alpha, weighted
  kappa, ICC(2,k) and intra-rater reliability coded and tested (14/14 against textbook values). 2D vs 3D cannot be
  fully blinded. **No ratings exist.**
* **Real multi-view data (reports/11).** About 30 candidate datasets checked: none pairs both surfaces of a diseased
  leaf, measures disease-induced curl in 3D, or gives gall volumes. Healthy 3D plant sets (e.g. PAA-Plant,
  PLANesT-3D) can test the held-out-view code before capture.
* **Acquisition protocol (reports/12; configs/acquisition/specimen_metadata_schema.json).** 30 specimens (8 per disease
  + 2 healthy controls each; minimum 20), 98 views each including an underside ring, flat photographs of both surfaces,
  one SAGE-like view as the only input and all others held out; detectable paired effect d_z = 0.53 at n = 30; about
  1.5 person-hours per specimen. Peach Leaf Curl only in April-May 2027.

## 8. Claims

Supported (measured, with controls): representation of surface, direction (when cited) and volume; evidence-based
placement; multi-view internal consistency; GaussianEditor changes colour, not shape (artefact measures pass Holm).
Not supported: downstream benefit at this scale. Untested: correctness of hidden structure, biological realism,
physical size (no scale in photos). No "recovered", "validated biology" or "proven" claim is made.

## 9. Remaining blockers

| Blocker | Effect | Unblocked by | Owner |
|---|---|---|---|
| No matched real views | hidden surface, direction, volume cannot be scored | capture per reports/12 | lab (needs funding/time decision) |
| No expert raters | biological realism unmeasured | send package v1 to 3 pathologists | advisor / PI |
| No physical scale | gall size in cm rests on an assumed 18 cm ear | ruler in captures | capture team |
| Peach curl direction not in the source | curl abstains for the demo disease | a verbatim source with a direction | literature check |
| Render-to-photo gap | synthetic data lowers real-photo accuracy at small k | rendering/background work before E5 | engineering |
| Reconstruction errors | ring topology (Peach 0006), over-scale | multi-view input | method |
| Workstation disk | 13 GB free after cleanup; /home/scslab holds 585 GB | archive or shared-storage decision | lab admin |

## 10. Execution logs and reproduction

* Demo runs: `results/demos/execution_log.jsonl` (every method, every specimen, timestamps; includes superseded
  entries from runs repeated after the gall-shading fix, the kernel-abstention rule and the curl-target fix, noted in
  `results/demos/execution_notes.txt`), `~/Desktop/Research/logs/demos_*.log`.
* Regeneration: `~/Desktop/Research/logs/regen_1014_1537.log`. Downstream: `results/downstream_minimal/run_log.txt`.
* Cleanup: `results/CLEANUP_LOG.txt`. Code snapshot before this session: `backups/code_pre_v2_20261014.tar.gz`.

```bash
cd ~/Desktop/Research/external/SAGE-Extend; E=~/micromamba/envs/sam3d/bin/python
CUDA_VISIBLE_DEVICES= $E -m pytest -q -p no:cacheprovider tests/test_v2_operators.py tests/test_review_agreement.py
for d in tomato corn mango peach; do CUDA_VISIBLE_DEVICES=1 $E experiments/demos/run_demos.py $d; done
$E src/analysis/demo_summary.py; CUDA_VISIBLE_DEVICES=1 $E src/figures/demo_figs.py; CUDA_VISIBLE_DEVICES= $E src/figures/slides.py
$E src/analysis/assemble_report15.py
```
