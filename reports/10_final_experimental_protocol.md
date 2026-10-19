# 10 — Decision (brief §24) and next steps

**Outcome A (verified 3D reconstruction): not supported.** No independent reference (second view or 3D measurement) exists for any specimen. Input-view agreement after pose correction is a fit, not an accuracy measure.

**Outcome B (knowledge-controlled 3D simulation): technical preconditions partly met; not established.** Three preconditions hold on the pilot:

- the KB-operator geometry editor gives coherent, regularised, KB-traceable shape change;
- it realises more of the stage's KB geometric attributes than an equal-magnitude generic warp (small n, 2 classes, partly by construction);
- one 3D asset is far more cross-view consistent than independent per-view 2D edits (internal consistency only).

Outcome B additionally requires *biological validation*, which was not possible (no expert raters). The detectors are coarse; the depression detector is non-specific; on volumetric organs the outgrowth response is attenuated.

**Outcome C (downstream augmentation): not tested.** The downstream classifier experiment (§18) was not run. The Stage-5 gate requires a reliable controlled pilot first, and 15 assets cannot power a held-out comparison. At the measured ≈3.4 GPU-h per specimen for the full arm set, 250 assets need about 850 GPU-h locally. With only D, J and H per asset the figure is about 1 GPU-h per asset (≈250 GPU-h, ≈5 days on two TITAN X).

**Classification: Outcome D (no established advantage) for now.** Neither GaussianEditor nor the new pipeline has been shown to produce biologically valid disease geometry, and no advantage on real data has been measured. The evidence supports a narrower technical contribution: diagnosis and repair of the pipeline, plus a controllable, auditable KB→geometry editing method with honest metrics.

**Recommended next experiments, in priority order.**

- **Blinded expert review** (2–3 plant pathologists). Rate stage-matched renders of D/J/H/K/F and 2D C-stage/I per specimen for symptom presence, stage correctness and implausible artefacts. This is the only way to move toward Outcome B.
- **Operator coverage:** a directional *roll* operator. Grape Leafroll's text specifies "margins rolled under", the clearest geometric KB claim in SAGE. Also calibrate outgrowth on synthetic volumes, and lesion-anchored (observed-spot) placement instead of random sites.
- **Two-sided leaves:** SAM 3D leaves are single SH-0 sheets, so lower-surface symptoms (Tomato Leaf Mold, Squash Downy Mildew) cannot be represented. Duplicate the sheet with offset normals before any underside edit.
- **Reference data for external validity:** even 20–30 specimens photographed from several views would turn every "unverified" label into a measurement. (Out of scope by current decision.)
- **Downstream test (C)** only after 1–3: matched synthetic budget, leakage-safe SAGE test split, real-only vs +C-stage 2D vs +J/H 3D renders.



Artifacts: reports 00–09, `NEXT_EXPERIMENT_RESULTS.md`, `reports/final/report.pdf`.
