# REVIEW_TODO — response to the detailed scientific review of the SAGE-3D deck (v3, 36 slides)

Source: external review pasted 2026-10-15 ("Major scientific revisions needed before
publication"). Each item is one review point turned into a concrete, checkable task.
Status: `[ ]` open · `[~]` in progress · `[x]` done (with evidence path) · `[B]` blocked on
something only a human can provide (specimens, raters). Work items top to bottom.

Guiding rule from the review: **keep operator compliance (engineering) separate from
biological fidelity (needs experts/real data) and practical validity (needs a real task).**
Never promote the first to evidence for the second or third.

---

## Part A — Deck corrections (claims, labels, consistency)

- [x] **A1 Slide 17 vs 20:** make input consistency a *measured condition*, not an
  unconditional claim ("without contradicting the photo"); state the mango curl is a
  counterfactual positive control (fitted-view IoU 0.91 → 0.62).
- [x] **A2 Slide 18:** report gall volume as a normalized quantity (fraction of ear volume /
  ear-length-normalized); label any cm³ value "scale-assumed (18 cm ear, no ruler)".
- [x] **A3 Slide 23:** "no suitable dataset found among the datasets examined" (not "none exist").
- [x] **A4 Slide 26 vs Slide 5:** reconcile "11 of 17 GE runs succeeded" with "E/F complete on
  15/15" (attempt accounting vs final status) on the slide itself.
- [x] **A5 Everywhere:** call 0.92 "fitted input-view IoU (optimised and scored on the same
  mask)", never reconstruction accuracy; show raw → pose-refined → depth agreement separately.
- [x] **A6 Slide 21:** explain that coherence is partly inherent to one shared 3D asset; show
  edit magnitude next to coherence (an unedited object scores perfectly).
- [x] **A7 Slide 14:** three evaluation levels table (operator compliance / biological
  fidelity / practical validity); label recall 0.44 vs 0.07 as compliance with a
  construction-bias warning.
- [x] **A8 Slide 15:** say explicitly that knowledge helps 2D too (not a 3D-specific gain).
- [x] **A9 Slide 4:** pipeline shows three information streams: observed evidence,
  KB-supported inference, synthetic geometry.
- [x] **A10 Slide 27:** each failure category → the method change that addresses it.
- [x] **A11 Structure:** move "three focused tests" (16) right after the pipeline; engineering
  audit (5–6) to backup; twelve questions (29–30) to appendix; merge 31–33 into one
  prioritized execution slide + one backup.
- [x] **A12 Slide 22:** add an error-analysis view of the synthetic-to-real gap.
- [x] **A13 Compact 10-slide meeting deck** (review §12 order); the 36-slide deck becomes backup.

## Part B — Analyses and experiments that can run on beast0 (no new data)

- [x] **B1 Curl geometry, independent of the operator target:** curvature profile at several
  positions along the leaf (not just mean margin angle), surface strain, self-intersection
  rate, midrib/vein preservation; robustness across all specimens with a curl operator.
- [x] **B2 Observed-evidence policy (KB vs image disagreement):** explicit rule
  preserve / counterfactual-flagged / abstain; edits that contradict clearly observed
  morphology are refused unless run in counterfactual mode; report which demos trigger it.
- [x] **B3 Corn smut:** normalized volume; vary gall count and severity; preservation of
  unaffected kernels and ear geometry; matched *generic* volumetric editor baseline (same
  volume budget, no KB/tissue placement); report kernel placement 1.00 as constraint
  compliance, not a finding.
- [x] **B4 Leaf underside:** two-layer 2D baseline (separate upper/lower textures, same KB);
  test that the underside pattern is not a copy of the upper-surface mask; measure
  surface-normal separation / thickness; check upper/lower correspondence under view change.
- [x] **B5 Statistics:** paired per-specimen table for every primary comparison;
  leave-one-disease-class-out; explain bootstrap-CI vs Holm-adjusted test differences;
  pre-declare ONE primary biological outcome (for the expert study).
- [x] **B6 Consistency baseline:** shared-canonical-texture (multi-view-consistent) 2D
  editing baseline; report appearance consistency together with edit magnitude.
- [x] **B7 Stronger 2D shape baseline:** explicit 2D geometric warp baseline that receives
  the same KB constraints (direction, location) and can change the silhouette.
- [x] **B8 Render-to-real domain-gap diagnosis (small pilot, before any 250 GPU-h study):**
  matched comparison of reconstruction-only renders vs KB-edited renders vs controlled 2D
  edits, with background / lighting / texture / pose factors isolated.
- [x] **B9 Method formalisation:** write the editor as the constrained objective
  G* = argmin λ_obs L_obs + λ_KB L_KB + λ_anat L_anat + λ_reg L_reg, map each implemented
  term (anchor, ARAP, placement, KB targets) to it, and state what is NOT yet implemented
  (uncertainty-aware constraints); ablate the terms that exist.
- [x] **B10 Reproducibility audit:** one script regenerates every number on the slides from
  results/; verify counts across slides.

## Part C — Blocked on things only a human can provide

- [B] **C1 (kit READY) Blinded plant-pathologist validation:** review package (84 items, reports/13) is
  ready; needs 2–3 qualified raters. Prepare the send-out kit (instructions, rating form,
  pre-declared primary outcome, agreement analysis script) so it can go out immediately.
- [B] **C2 Specimen-matched multi-view validation:** needs real specimens + held-out views
  (protocol: reports/12). Start with a feasibility set of a few specimens.
- [B] **C3 Measured corn smut galls / real curl references / paired upper-lower photos:** need
  specimens or scans (dataset search was out of scope by the user's decision).

---

## Progress log
(newest on top; one line per completed item with the evidence path)
- 2026-10-15 Deck: new slides "Method: One Evidence-Conditioned Objective", "When the KB and the Photo Disagree", "Is It Really 3D? Fair 2D Baselines and All 12 Leaves"; Answer, E4, E5, Statistics slides updated; all review numbers from experiments/review/review_macros.py (b10 check covers them).
- 2026-10-15 B1: experiments/review/b1_curl_robustness.py -> results/review/b1_curl/summary.md (12 leaves x up/down, seed 0; Peach 0006 ring flagged). Direction follows the command on 11/11 usable leaves (up and reversed); magnitude varies 7-43 deg for one target; midrib preserved (length ratio >= 0.985, shape RMS <= 0.007 L); fails on 2 Citrus leaves (0012, 0016: strain p95 1.2-1.7 vs 0.07-0.41, relief corr 0.06-0.13, positions in commanded direction 60-80%); every curl contradicts its photo (0/22 pass preserve mode, IoU drop 0.07-0.38). Added roll_profile (per-position roll) to metrics/structure3d.roll_metrics (backward compatible).
- 2026-10-15 B4: experiments/review/b4_surface_visibility.py -> results/review/b4_surfaces/summary.md. A two-layer 2D representation (front image + separately edited back image) shows one layer per view, so it must show the wrong surface on the minority-surface pixels: 4-23% of leaf pixels on average across the 12 leaves (Tomato 7-13%, 5-8 of 18 views mixed; flat Mango 0044: 0/18). 3D adds surface correctness only at oblique views and on non-flat leaves. Honest notes: the lower mold FOOTPRINT is the upper spot footprint by construction (KB 'below leaf spots'), only texture differs; thickness 0.004 L is assumed. (First run had a uint8-overflow/background bug, caught by inspecting renders and rerun.)
- 2026-10-15 B9: reports/16_method_formalisation.md (objective G* = argmin l_KB L_edit + l_anat L_arap + l_reg (L_smooth + L_anchor) s.t. L_obs <= tau; each term mapped to code; NOT implemented: uncertainty-aware constraints, differentiable L_obs, species-specific anatomy, learned KB->operator map). Ablation experiments/review/b9_ablation.py -> results/review/b9_ablation/summary.md (Mango 0049, Peach 0087): removing ARAP raises strain p95 3x/4x (0.12->0.35, 0.41->1.75), Peach self-contact 0.006->0.04, relief corr 0.61->0.49; the smoothness term (weight 0.01) has no measurable effect (report as redundant); roll change and IoU set by the KB target, not the regularisers.
- 2026-10-15 C1 prepared (still blocked on raters): results/review_package_v1/sendout/ = SAGE3D_rater_package_v1.zip (88 files, verified no answer key), SENDOUT_KIT.md (rater qualifications, invitation email draft, coordinator checklist, exact analysis command; primary outcome f_overall_accept). Nothing sent: the PI chooses raters.
- 2026-10-15 B10: experiments/review/b10_consistency.py regenerates every macro file (slides.py, demo_figs, edit_magnitude, gap_analysis), diffs them with the deck's generated/ copies, checks every number macro used in the deck is defined, and checks the hand-typed compute table against reports/08. Result: OK, 216 macros identical (only the generation timestamp differed). Also hardened metrics/surfaces.surface_stats against a uint8 overflow (sd + 40) found while debugging B4; demo numbers unchanged until regenerated.
- 2026-10-15 B8: experiments/review/b8_domain_gap.py -> results/review/b8_domain_gap/summary.md (15 specimens, DINOv2 ViT-B/14 as in E5). Median distance to the nearest real same-class photo: render at the photo's own pose + background +0.28 (largest step: object rendering fidelity), white background +0.09, new viewpoints +0.21, replacing colours with the photo's -0.01 (no help). Gap = rendering fidelity + unnatural viewpoints, not colour/texture. Implication for E5: near-input views composited on real backgrounds; improve splat rendering; do not scale arbitrary rig-view augmentation. ('photo' row ~0 because the specimen photo is in the reference pool.)
- 2026-10-15 B7: experiments/review/b7_warp2d.py -> results/review/b7_warp2d/summary.md. Best-case 2D shape baseline = exact projection of the 3D curl into the photo (backward warp, same KB theta/r0). It barely separates 'curl up' from 'curl down' (dE 4.7-13.6 at the input view) while the 3D assets differ by dE 41-46 over 18 views; the 2D warp also contradicts the photo (IoU 0.59-0.76). Direction is a 3D property a single image barely encodes. Mango only (the counterfactual positive control).
- 2026-10-15 B6: experiments/review/b6_shared_texture.py -> results/review/b6_shared_texture/summary.md. Ctex = 2D KB edit (C) lifted onto the shared unedited asset (visible Gaussians take the edited pixel colour): cross-view dE 2.7 (floor 2.0) while recolouring by dE 32 (more than GaussianEditor's 18-20); per-view 2D edits 19.3. The E4 consistency advantage comes from sharing one asset, not from the KB geometry algorithm.
- 2026-10-15 B5: experiments/review/b5_statistics.py -> results/review/b5_statistics/{paired_values.csv, summary.md}. KB-relevant comparisons (geometric recall J/H vs G/F) have 4 non-tied pairs -> exact Wilcoxon floor 0.125, Holm floor 0.625-0.875: cannot survive correction at ANY effect size with this n (the bootstrap CI excluding 0 at n=6 answers a different, anti-conservative question). Only artefact measures (coherence, grain, fuzz; 9-15 specimens) can and do survive. Appearance recall K/F vs E flips sign under leave-one-class-out. Primary biological outcome already fixed in reports/13 (f_overall_accept, Holm over 3).
- 2026-10-15 B3: experiments/review/b3_corn_galls.py -> results/review/b3_corn/summary.{json,md}. Scale-free (volume / L_ear^3). Hidden-side CENTRES alone do not preserve the photo (0012: IoU 0.94 -> 0.77 at 3 galls): KB-sized galls protrude past the silhouette. Render-checked preserve mode Vp (sample hidden-kernel placement, render input view, accept iff IoU drop <= tau, else retry/abstain) = the L_obs term as a hard constraint: 6/6 runs accepted (1-8 tries), the photo forces smaller galls on 0012 (eq. diam/L 0.061 vs 0.094; volume 0.6 vs 6.1 per mille). Severity sweep n=1/3/6: only single galls pass unconstrained. Kernel vs random placement contradict the photo equally (V vs Vr). Synthetic sizes within the range of the real galls visible in the photos (extent/L 0.09-0.14): size plausibility only. Placement helper: placement.gall_centres(hidden_only=True).
- 2026-10-15 B2: src/geometry_editor/evidence_policy.py (preserve / counterfactual / abstain) + experiments/review/b2_evidence_policy.py -> results/review/b2_policy/summary.{json,md}. TAU = 0.046 = largest fitted-IoU drop of 24 appearance-only edits (E/F/K), sensitivity at 2 TAU. Preserve mode: accepts Peach pucker 18/18 and Tomato two-surface 27/27; refuses all 18 Mango curls (counterfactual-only); Peach curl hypotheses = abstain; Corn galls 4/6 KB placements refused.
- 2026-10-15 A13: overleaf meeting.tex = title + 10 slides in review order (question, pipeline, 3 tests, ablations, downstream, failures, validation plan, decision) via \includeonlyframes on labelled frames of the full deck (shared preamble.tex), so numbers never diverge. Full deck reordered to match (ablations+stats before E4; execution plan before go/no-go). PDFs: reports/final/SAGE_3D_deck_v4.pdf (38 pp), SAGE_3D_meeting_v4.pdf (11 pp).
- 2026-10-15 A12: src/downstream/gap_analysis.py (cached features, descriptive) -> results/downstream_minimal/gap_analysis.{json,pdf}; new slide "E5: Why Renders Hurt". Renders 0.46-0.52 cosine from the nearest real photo vs 0.16 real-to-real (2D edits 0.24); real-vs-render AUC ~1.00, white background 0.99 (not the background); at k=4 pilot recall -21 (KB3D) / -24 (unedited) points, FPR other ~-2: renders shift the class boundary. Unedited renders already as far off as KB ones -> gap is reconstruction/rendering. Caveat: source-photo AUC 0.91 reflects only 3 specimens.
- 2026-10-15 A11: deck reordered (tests overview after pipeline; Status + Phase 0 audit + twelve questions + blockers + deliverables -> backup after a divider; next steps + blockers merged into "Execution Plan: Validate Before Generating More" in the review's priority order); numeric slide cross-references replaced by names.
- 2026-10-15 A10: slide 27 cards = failure -> method change (done: pose fix, two-surface leaf, accept-if-better, geometry editor, evidence placement, stage-aware KB; next: ring rejection, held-out depth gate, lesion detector, observed-morphology constraint).
- 2026-10-15 A9: slide 4 legend/callout name the three streams (observed / KB-inferred / synthetic).
- 2026-10-15 A8: slide 15 states KB colour gains are not evidence for 3D; Arm I labelled as IP2P with a KB-constrained 2D warp baseline pending (B7).
- 2026-10-15 A7: slide 14 callout = three evaluation levels; recall 0.44 vs 0.07 labelled operator compliance; stale "curl: none" row fixed (v2 curl operator exists).
- 2026-10-15 A6: src/analysis/edit_magnitude.py -> results/statistics/edit_magnitude.json + macros; slide 21: consistency is partly free for one shared asset; edit size vs D (GE 20.2/18.4 dE, J 4.9 dE but silhouette 0.14, 2D 36-47 in-mask).
- 2026-10-15 A5: "fitted input-view IoU" on slides 3, 11, 30; slide 11 states 3D accuracy and held-out views are not measurable (overleaf sections/act1_question, act3_experiments, act4_decision).
- 2026-10-15 A4: slide 26 compute table refreshed from reports/08 (GE E 16/17, F 15/20, K/H 24/24, IP2P 444/444; was a stale pre-Phase-B 11/17 and 158/158); note explains attempts (slide 26) vs finished specimens (slide 5).
- 2026-10-15 A3: dataset-search wording -> "none suitable among the datasets examined" on slides 3, 20, 23, 30.
- 2026-10-15 A2: scale-free gall volume (V / L^3, per mille) added to src/figures/demo_figs.py (macros CnVolVn 6.1, CnVolJn 4.5); slide 18 leads with it, cm^3 labelled assumed-scale, kernel placement labelled constraint compliance.
- 2026-10-15 A1: slide 17 card retitled "Positive control" with the counterfactual + IoU drop in the card; slide 20 table/callout make photo agreement a measured condition (kept: underside, galls; broken: Mango curl).
