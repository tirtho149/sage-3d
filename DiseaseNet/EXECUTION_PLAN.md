# DiseaseNet — execution plan (saved 2026-10-15; nothing run yet)

The protocol is `DISEASENET_MASTER_PROTOCOL.md`. This file maps it onto what already exists
on `beast0` and in SAGE-Extend, with measured costs, so the first session can start
directly at Phase 0. All numbers below come from SAGE-Extend artifacts unless marked
**estimate**.

## 1. What already exists (reuse through stable interfaces; do not modify SAGE-Extend)

| Need | Existing piece | Location (under `~/Desktop/Research/external/SAGE-Extend/`) |
|---|---|---|
| SAGE images + KB | 820,502 rows; 631,869 with KB symptoms (635 classes / 325 crops); ~36% exact-duplicate rows; standalone `sage_metadata.parquet` is STALE, use the shards | `~/Desktop/Research/data/raw/SAGE/data` (20 GB); `data/sage_meta/` |
| Leakage-checked real splits | pHash + cluster-disjoint build / test / val / opt split, leakage report | `src/downstream/prepare.py`, `results/downstream_minimal/{image_manifest_real.csv,leakage.json}` |
| Frozen-feature probe + stats | DINOv2 ViT-B/14 features, few-shot probes, paired bootstrap + Holm | `src/downstream/{features,minimal_probe}.py` |
| Reconstruction | SAM 3D with CPU offload on TITAN X (sm_52), pose fix A3* | `src/reconstruction/`, `src/camera_alignment/`, env `sam3d` |
| Editing | geometry editor (curl, pucker, galls, two surfaces), evidence policy, GaussianEditor (env `gedit`), IP2P 2D edits | `src/geometry_editor/`, `src/gaussian_editing/`, `src/editing2d/` |
| Rendering | 18-view rig, input-view render | `src/rendering/rig_render.py`, `camera_alignment/align.py:render_at_input` |
| Gap diagnostics | real-vs-render distance/AUC, matched factor pilot | `src/downstream/gap_analysis.py`, `experiments/review/b8_domain_gap.py` |

Existing assets: **15 reconstructions in 5 classes** (Peach Leaf Curl, Tomato Leaf Mold,
Corn Smut, Citrus Canker, Mango Anthracnose) — 3 per class. Far below even Stage 0.

## 2. Lessons that should shape the design (from the SAGE-3D review work)

- Renders are trivially separable from photos (real-vs-render AUC ≈ 1.00, white background
  still 0.99). Matched pilot: object rendering at the photo's pose +0.28 cosine distance,
  new viewpoints +0.21, background +0.09, recolouring with photo colours −0.01. → Prioritise
  near-photo viewpoints, real-background compositing and splat render fidelity (J4/E5/E6)
  over more edits or more views.
- Adding renders to few-shot real training cut pilot-class recall by ~21 points at k=4:
  renders shift the class boundary. → The strict synthetic-only test (L1) and the mixed
  test (L4) must be read separately.
- Every curl edit contradicted its photo; KB-sized galls do too unless render-checked.
  → Use the evidence policy (preserve mode) in the generator, or label outputs counterfactual.
- With n = 6 specimens the exact test cannot reach significance. → Power comes from
  independent assets and classes, not seeds or views (protocol J2 is the key ablation).

## 3. Measured unit costs on beast0 (2× GTX TITAN X 12 GB, Maxwell: no fast fp16/bf16)

| Step | Measured | Notes |
|---|---|---|
| SAM 3D reconstruction | 1070 s median (17–26 min), peak 11.6 GB | fits one GPU at a time; the other GPU hosts the display |
| Pose alignment | 78 s | |
| Geometry edit (J/G) + renders | ~45 s | cheap |
| GaussianEditor (200 steps) | ~2200 s | expensive; avoid per-image |
| IP2P 2D edit | 57 s GPU / 175 s CPU | |
| DINOv2 ViT-B/14 features | cached for the downstream set | ViT-S/14 needs a download check |
| Disk | **25 GB free** (2026-10-15) | the binding constraint locally |

## 4. Reconstruction budget per stage (**estimate**: 18 min per new asset, 1 GPU)

| Stage | Classes × new assets | New reconstructions | GPU-hours (recon only) | Where |
|---|---|---|---|---|
| 0 debug | 20 × 3 | ~60 (15 exist) | ~14–18 | beast0 (~1 day) |
| 1 pilot | 50 × 5 | ~250 | ~75 | beast0 (~3–4 days) or Nova |
| 2 broad | 100 × 10 | ~1,000 | ~300 | Nova |
| 3 large | 250 × 10–20 | 2,500–5,000 | 750–1,500 | Nova (scavenger) |
| 4 max | ≤ 635 (after audit) | audit first | — | Nova |

Rendering existing assets (8 views × 4 conditions) costs seconds per asset. Storage per
asset must be measured in Phase 0 before committing to Stage 1 on local disk.

## 5. Phase 0 checklist (first session)

1. `DiseaseNet/` skeleton per Part R; `reports/00_existing_evidence_and_research_gaps.md`
   (summarise SAGE-3D reports 00–17).
2. Read InsectNet and WheatNet (actual papers) → `reports/01_literature_and_method_design.md`.
3. Class audit from the SAGE shards (not the stale parquet): canonical `crop::disease`,
   duplicates, per-class counts, KB availability, licences → `metadata/class_ontology.parquet`,
   `CLASS_COVERAGE.csv`, `SAGE_CLASS_AUDIT.md`.
4. Eligibility tiers A–D using the existing operator table (`deform.py` KB terms, curl,
   gall, bilayer) and appearance terms → counts per tier, nothing silently dropped.
5. Locked real splits: start from `src/downstream/prepare.py` (pHash/cluster-disjoint),
   add the provenance graph (source image → asset → edit → render) with executable
   leakage assertions; reserve open-set classes.
6. Choose the 20 Stage-0 classes by the Part M cohort rules (crop diversity, modality,
   frequency, feasibility) before any results exist; record the rationale.
7. Measure: SAM 3D asset size, render throughput, DINOv2 ViT-S/14 availability, fine-tune
   memory for ViT-S at 224 px on 12 GB → `COMPUTE_ESTIMATE.md`.
8. Stop and report to the user before Phase 1 GPU generation.

## 6. Open decisions for the user

- Run Stage 1+ on Nova (data via Globus, already configured) or stay local?
- Disk: free space before Stage 0 (other projects' results were kept, ~3.4 GB).
- Procedural (non-SAGE-derived) assets for a strict synthetic-only claim: in scope?
