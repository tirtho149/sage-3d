# DiseaseNet — RESEARCH_PROGRESS (newest first)



## 2026-10-16 06:00 — Stage 0R + diagnostics (DEV only; FINAL untouched; T0 frozen probes unless stated)
All numbers: results/stage1/*; contrasts pooled over 3 matched replicates (experiments/S1/analyze_probes.py).
- Fixed KB edits HURT: P1 KB3D-U3D -0.153 [-0.171,-0.136]; P2 KB3D-KB2D -0.023; P3 MIX50-KB2D -0.003 n.s. (Holm).
  KB vs generic: -0.106 (2D), -0.107 (3D). Generic edits also hurt (GEN2D-PHOTO2D -0.145). IP2P recolours whole organs.
- Old Stage-0 "KB3D" gain = photo texture: TEX3D-U3D +0.062; colour harmonisation 0.000; render gap TEX3D-PHOTO2D -0.058.
- Knowledge-only test (cross-class, 16 classes, rep 0): within-crop accuracy XPHOTO2D 0.209, XGEN2D 0.261, XKB2D 0.438,
  XKB3D 0.399 -> KB text carries transferable class information (+0.18..0.23) but less than real symptoms.
- MODALITY x GENERATOR interaction (4 geometry classes vs 16 appearance): KB3D-KB2D +0.042 vs -0.043 (interaction +0.085
  [0.052,0.120]); U3D-PHOTO2D +0.009 vs -0.149 (interaction +0.158); KB ops on KB colour +0.027 (geometry cohort).
  Caveat: 4 classes, 3 of them SAGE-Extend pilot assets -> needs the 13 eligible geometry classes.
- Parent diversity (PHOTO2D, 18 classes, 96 img/class): 3 parents 0.686 (draws 0.663-0.732), 6: 0.746, 12: 0.800;
  more renders at 3 parents (8/16/32 per parent): 0.683/0.684/0.686.
- Background (8 img/parent): own background R1 best for every arm (+0.09..0.13 over white); background-only 0.123
  (chance 0.05); source-only rule 0.238. Viewpoint: input/narrow = V8 >= WIDE (TEX3D WIDE -0.036).
- Feature preservation (input view): HF texture kept U3D 22%, HARM3D 28%, TEX3D 35%; DINO cos 0.69/0.71/0.85.
- Few-shot S1 (frozen; adapt shots, w chosen on adapt-val): k=16 PHOTO2D 0.881 ~ U3D 0.878 ~ KB3D 0.876 > REAL+3 0.844
  > REAL 0.798; no synthetic arm beats PHOTO2D at any k; REAL+3 (3 parents, no aug) k=0 0.677 > every synthetic arm.
- Open set (energy): AUROC 0.73-0.85, FPR95 0.61-0.84 for all arms (weak); KB arms worst.
- New parents: 9/class for 18 classes (coffee brown eye spot, mango anthracnose: photos < 256 px); 24 visual rejects
  (scenes, microscopy, overlays) in data/synthetic/stage1/visual_rejects.csv.
Running: fine-tuning study (scripts/s1_ft_study.sh), KB/generic edits for new parents (scripts/s1_diversity_gen.sh).

## 2026-10-16 — STAGE 1 started; Stage-0 audit found a pipeline bug that changes the Stage-0 interpretation

Brief: reports/stage1/TASK_stage1_master_prompt.md.
**BUG (verified):** src/generation/edit2d.py passed a boolean mask to SAGE-Extend ip2p_core.edit, which computes
`mask > 127` (all-False for booleans) -> composite weight 0 -> every Stage-0 "edit" is the unedited photo.
Evidence: all 60 assets have kb_s0.png byte-identical to generic_s0.png; CIEDE76 photo->edit = 0.000 in and outside
the mask for all 45 new assets (results/stage1/audit/edit_magnitude.csv). SAGE-Extend callers pass uint8 -> unaffected.
Consequences for Stage 0 (numbers verified exactly from saved predictions, but their MEANING changes):
- KB2D and GEN2D == PHOTO with different augmentation seeds; their +-0.01 spread = augmentation noise floor.
- KB3D == U3D + source-PHOTO texture re-projected onto visible Gaussians (+ geometry ops in 4 classes):
  KB3D - U3D (+0.057 T0 / +0.052 T3) measures texture restoration (+geometry), NOT knowledge.
- "KB vs generic 2D makes no difference" and "KB edits help 3D" are unsupported. Deck slides must be corrected.
Fix: uint8 mask + guard (raises if the organ is unchanged). Fixed edits -> data/synthetic/stage1/edits2d (seeds 0-2),
scripts/s1_run_edits.sh. Stage-0 files left untouched as the audited record.
Other audit facts: 60 assets but 59 root photos (one photo is labelled both coffee::berry_blotch and
coffee::brown_eye_spot in SAGE; 1,840 sha1 carry >1 class label in SAGE); no checkpoints saved (last-epoch, fixed
schedule; T3 = 450 steps real / 300 steps fold); 16 Stage-0 test photos share near-dup groups with open-set images;
class amendments (05:52) precede test creation (06:10). 9/20 classes have no unused test-partition images ->
Stage-1 FINAL test = the untouched `val` partition; Stage-0 test becomes the exploratory/dev benchmark.

## 2026-10-15 20:10 — Stage 0 pipeline FINISHED (logs/stage0_pipeline.log; results/E0/)
Real locked test (4,202 imgs, 20 classes), macro-F1 [95% CI], seed 0 only:
| arm | T0 frozen probe | T3 full fine-tune |
|---|---|---|
| PHOTO (3 real parents/class) | 0.662 [0.642, 0.679] | 0.611 [0.592, 0.628] |
| KB2D | 0.672 [0.651, 0.689] | 0.604 [0.585, 0.621] |
| GEN2D | 0.669 [0.649, 0.687] | 0.621 [0.602, 0.639] |
| KB3D | 0.591 [0.572, 0.607] | 0.546 [0.528, 0.562] |
| U3D | 0.533 [0.513, 0.550] | 0.478 [0.459, 0.495] |
Reading: 3D renders below photos and 2D edits; KB3D > U3D (+0.06 both protocols); KB2D ~= GEN2D (CIs overlap).
T3 < T0 for every arm (1,920 training images -> fine-tune overfits). IID fold-0 CIs too wide to rank.
Paired group-bootstrap diffs: T0 in t0_summary.json, T3 in results/E0/t3_paired.json (KB3D-U3D +0.057 / +0.068, CIs exclude 0).
Deck: 2 slides in Overleaf 6ac80070 sections/act3c_diseasenet.tex (commit c983c33); regenerate with
`~/Desktop/Research/.venv/bin/python scripts/deck_stage0.py` -> copy figures/deck/stage0_real.pdf to figs/results/, diseasenet_numbers.tex to generated/.
2026-10-16 00:25 T3 seeds 1-2 done (scripts/run_stage0_seeds.sh, logs/stage0_seeds.log). T3 3-seed mean real macro-F1: PHOTO 0.620, KB2D 0.615,
GEN2D 0.615, KB3D 0.554, U3D 0.502 (SD 0.031). Paired (mean; seeds with CI excl. 0): KB3D-U3D +0.052 (2/3, range +0.016..+0.071),
KB3D-PHOTO -0.066 (3/3), KB2D-KB3D +0.061 (3/3), KB2D-PHOTO -0.005 (0/3), GEN2D-KB2D +0.000 (1/3). T0 deterministic, not re-run.
Deck updated (Overleaf 6ac80070). Not yet done: written report.

## 2026-10-15 — Stage 0 approved and running (unattended: scripts/run_stage0.sh)

Done:
- Parent photos: 45 new (15 classes x 3) from the PARENT partition, SAM ViT-H centred-box masks + QC
  (score, coverage, single component, borders, hole-fill, brightness) + documented agent visual QC
  (13 rejects, data/synthetic/stage0/visual_rejects.csv; contact sheets saved). + 15 pilot assets (read-only).
- Class set AMENDED before any result (configs/datasets/stage0_classes.json "amendments"): rice::bacterial_blight
  (KB text describes panicle blight, identical to rice::bacterial_panicle_blight) and the rice pair -> strawberry
  (leaf_scorch, leaf_spot); potato::bacteria_wilt_and_soft_rot (KB leaf wilt vs tuber photos) ->
  potato::tomato_spotted_wilt_virus. Same-crop / seeded-crop rule, scripts/amend_stage0.py.
- KB edit specs (data/synthetic/stage0/kb_specs.json, SAGE-Extend structured KB + documented hedge override for
  "may be observed"): only 4 classes get an implemented geometry operator; coffee pair has near-identical KB text.
- Measured so far: SAM 3D 1052 s/asset; 2D edits ~35 s; T0 PHOTO reference (3 real parents/class) real
  macro-F1 0.662 [0.642, 0.679].
- Code: src/generation/{select_parents,reconstruct,kb_specs,edit2d,render3d,build2d,registry}.py,
  src/training/{features,finetune}.py, experiments/E0_feasibility/run_t0.py, tests/test_synthetic.py,
  scripts/{extract_real,amend_stage0,run_stage0}.sh/py. Locked test cache: 4,202 images (cap 300/class).

Running: reconstruction (GPU 1, ~13 h) -> 9 amended-class assets -> render 3D arms -> 2D arms -> tests -> T0 ->
T3 (real + IID fold 0) on both GPUs. Logs: logs/stage0_*.log. Results: results/E0/.

## 2026-10-15 — Phase 0 complete; STOPPED before Phase 1 GPU generation (as planned)

Done (all from executed code; nothing generated or trained):
- Class audit of all 48 SAGE shards -> 195 eligible classes (135 appearance, 60 geometry), 60 crops,
  428k unique images; 1,057 unsupported with reasons (`reports/02`, `CLASS_COVERAGE.csv`).
- Locked leakage-safe real splits + open-set reserve (19 classes) + provenance guard
  (`IID_OOD_SPLIT_PROTOCOL.md`, `data/real_splits/LOCK.json`, `src/provenance/leakage.py`, tests 7/7).
- Stage-0 20-class set fixed before results (`configs/datasets/stage0_classes.json`).
- Measured compute (`COMPUTE_ESTIMATE.md`): recon ~18 min/asset dominates; render 7 ms/view;
  ViT-S/14 full fine-tune 72 img/s at bs 64 (5.6 GB).
- Literature read and design decisions (`reports/01`); existing evidence and gaps (`reports/00`).

Decisions needed from the user before Phase 1:
1. Approve Stage 0 (~45 new SAM 3D assets, ~14 GPU-h on GPU 1, ~0.6 GB) on beast0.
2. Stage 1+ on Nova or locally (disk: 25 GB free)?
3. Procedural (non-SAGE-derived) assets for a strict synthetic-only claim: in scope?

Resume: read EXECUTION_PLAN.md and this file; run `~/Desktop/Research/.venv/bin/python -m pytest -q tests/`.
