# RESEARCH_PROGRESS — SAGE-Extend 3D (brief v3: advisor requirement "what validated capability does KB-3D add beyond 2D?")

_Last updated: 2026-10-14 (v3 session). The v2 record below is kept unchanged after the divider._

## Review response (2026-10-15) -- START HERE NEXT SESSION
- External review of the v3 deck answered item by item: REVIEW_TODO.md (log, newest first) and reports/17_review_response.md (point-by-point). Method formalisation: reports/16.
- New experiments: experiments/review/b1..b10 -> results/review/*; evidence policy src/geometry_editor/evidence_policy.py; placement.gall_centres(hidden_only=True); roll_profile in structure3d.roll_metrics.
- Deck v5 (Overleaf 6ac80070): main.tex 41 slides, meeting.tex title+10; preamble.tex shared; review numbers from experiments/review/review_macros.py; check all numbers with `python3 experiments/review/b10_consistency.py` (exit 0 = consistent).
- Open, needs people/specimens: C1 expert raters (kit in results/review_package_v1/sendout/), C2 multi-view capture, C3 real galls / curl / paired surfaces.

## v3 summary (2026-10-14)
- **Phase B complete:** E/F on 15/15 (pilot runner, Oct 13), K/H on 15/15, J/G on 9/15 (Citrus and Tomato abstain: no implemented operator; correct per protocol). Regenerated all statistics; J vs E coherence/grain, K vs E grain/fuzz and H vs F grain survive Holm (artefact measures, not biology).
- **v2 editor (src/geometry_editor/):** anatomy.py (input camera in object space, adaxial normals, Isomap leaf frame, ear frame + kernel/gall tissue), curl.py (directional curl/roll/cup from the KB phrase; abstains without a direction), placement.py (evidence-constrained placement), bilayer.py (separate upper/lower surfaces), gall.py (volumetric galls replacing kernels); deform.py: vector targets, external normals, placement weights, coarse ARAP, lambda-override bug fixed. Verbatim external evidence: data/kb/supplementary_evidence.json (incl. negative evidence: no Peach curl direction in the source).
- **Metrics:** src/metrics/structure3d.py (roll angle, transverse curvature, tissue strain at 0.015-0.025 L, quadric-relief preservation, input-view IoU vs the REAL mask), src/metrics/surfaces.py (per-pixel surface identity; mold/spot colour placement).
- **Demos:** experiments/demos/run_demos.py -> results/demos/{peach,mango_curl,corn,tomato}/<specimen>/; summary: results/demos/summary.{json,md}; log: results/demos/execution_log.jsonl. Tests: tests/test_v2_operators.py (4 passed; caught and fixed a curl-target smoothing defect).
- **Agents' deliverables:** reports/11 (multi-view data search: none suitable), 12 (acquisition protocol, 30 specimens), 13 (expert review protocol; results/review_package_v1, 84 items, 14 tests pass), 14 (minimal downstream test: NOT SUPPORTED at this scale).
- **Report:** reports/15_revised_scientific_report.md. Deck: Overleaf 6ac800709ea44535f867ab26 (sections/act3b_what3d.tex + edits).
- **Disk:** results/CLEANUP_LOG.txt (GE per-iteration previews removed, 1.96 GB).

---

# RESEARCH_PROGRESS — SAGE-Extend 3D (brief v2: `reports/TASK_master_prompt_v2.md`)

_Last updated: 2026-10-12 20:00 CDT_

## Current phase
**Stages 1–3 implemented; Stage 4 (controlled pilot) running.**
- **Stage 1:** alignment done on 13/15 (median IoU 0.63 → 0.92, report 01). GE fixes done; the live verification of arm F is pending (Tomato 0015 F, ~17:47).
- **Stage 2 gate (reproducible geometry without catastrophic damage):** PASSED on Peach 0087.
  - J seeds 0–2 give consistent attributes, coherence 0.997, edge-stretch p95 19 %.
  - 5/5 operator tests pass.
- **Stage 3 gate (traceability):** PASSED. Every prompt/operator in `data/kb/structured_specs.json` is a verbatim span, asserted in code.
- **Phase A (0087): COMPLETE** (D, E, F, G, J, K, H, C-stage, I). 2×2 on 0087: D 0, K 0.5, G 0.06, H 1.0, interaction +0.44 (n=1). Interim PDF: `reports/final/report.pdf`.
- **Phase B:** geometry J/G running for all specimens; baseline E/F on GPU 1 (Peach, Tomato, Corn) and on GPU 0 later (Citrus, Mango).

## Completed in this stage (verified)

| Item | Evidence | Status |
|---|---|---|
| **F-GE-1: arm F crash.** Colon in the KB prompt was parsed by the OmegaConf CLI as a mapping. Fixed by structured per-run YAML (`src/gaussian_editing/ge_config.py`), preflight `check`, post-run `verify` against GE's `configs/parsed.yaml`, and `run_ge.sh` v2. | `tests/test_ge_config.py`: 10 passed. Old wrapper kept as `run_ge.sh.v1_cli_quoting.bak`. | FIXED, TESTED (unit). Real-run test pending: Tomato 0015 F on GPU 0. |
| **F-GE-2: GE no-op on Peach 0006.** SAM 3D writes `+inf` opacity logits; GE's anchor loss becomes NaN; with fp16 GradScaler, every step is skipped. | `results/ge_diagnosis/` (0006 output == input minus 824 pruned rows, bit-identical); `metrics.csv` loss_anchor_opacity = nan. Affects Peach 0006 and Corn 0012/0025/0052. | FIXED: `src/gaussian_editing/sanitize_splat.py` → `splat_ge.ply`; runner patched. Invalid 0006 E moved to `E_INVALID_nan_opacity_noop/`. Re-run pending. |
| **Arm E diagnosis (0063, 0087).** Edits move Gaussians by a median of ~1.2% of the bbox diagonal, but incoherently (k-NN coherence 0.31–0.32 vs 0.25 for i.i.d. noise). Leaf sheet is 1.8–2.3× thicker. | `results/ge_diagnosis/*_E_diff.json`; `src/metrics/gaussian_diff.py` (prune-aware index matching, verified on 0006) | CONFIRMED: appearance edit plus geometric jitter, no deformation. |
| **E1 camera check.** K reproduces the MoGe pointmap to 0.33 px (Tomato 0015); a principal-point change has no effect. | `results/alignment/Tomato__Leaf_Mold__0015/alignment.json` | Camera convention OK. |
| **E1 alignment, Tomato 0015.** IoU 0.40 → 0.94 with pose-only render-and-compare. SAM 3D object scale was ~1.7× too large. | same | TESTED (1 specimen) |

## Active jobs (local, beast0)

| What | GPU | PID / log |
|---|---|---|
| SAM 3D seed-0 batch (Mango 0094, 0044 left) | 1 | `run_sam3d_batch.py`, `logs/pilot_sam3d.log` |
| pilot_runner GPU 0 (Tomato 0015: GE E, then F; then exits, since other specimens are locked to GPU 1) | 0 | PID 822469, `results/pilot/runner_gpu0.jsonl` |
| pilot_chain (starts the GPU 1 runner after 15 seed-0 reconstructions, then the seed subset) | 1 | `logs/pilot_chain.log` |
| Alignment study A0–A3 over all reconstructions | 0 | `experiments/runners/run_alignment.sh`, `logs/alignment.log` |
| GE checkpoint pruning loop | – | `logs/prune_ge.log` |

Specimen locks: Peach 0087 = `phaseA` (reserved for the Phase A diagnostic); all others = GPU 1 except Tomato 0015 (GPU 0).

## Key results so far (verified from artifacts)
- **E1 alignment (15/15):** median IoU 0.63 → 0.92 (A3\*); camera fine; SAM 3D pose over-scaled in 13/15; 1 reconstruction failure (Peach 0006 ring).
- **GE (E and F) on Peach:** incoherent jitter (coherence ~0.31), grain ×13–17, KB geometric recall 0. With a verified KB prompt, F also fails to realise "reddish" (Δa\* +1.1).
- **J vs G (KB vs generic geometry, n=6 specimens, 2 classes):** recall 0.44 vs 0.07, Δ +0.37, CI [0.11, 0.63]. Caveats: outgrowth is attenuated on volumes (Corn), ring-failure artefacts, and the blister detector is near threshold on 0063.
- **Cross-view (same-point ΔE):** per-view 2D 17–33 ≫ GE 3D 7–11 ≫ geometry-only ≈ floor 1–4 (4–9 specimens).
- **Phase C (KB level):** Black Knot = outgrowth; Squash Downy Mildew = appearance only; Grape Leafroll "rolled under" has no operator yet.

## Reports written
00 failure audit · 01 camera alignment · 02 GE diagnosis · 03 geometry method · 04 KB grounding · 08 compute audit · 09 failure analysis · gpu_scaling_estimate · TASK brief

## Overnight queue (resumable; claim locks prevent duplicate GE runs)
- GPU 0 (`experiments/phaseA/queue_gpu0.sh`, log `logs/gpu0_queue.log`): Phase B priority K/H (Tomato 0015, Corn 0012, Citrus 0012, Mango 0049), then Citrus/Mango baseline E/F, then the remaining K/H.
- GPU 1: `pilot_runner.py 1 all` (baselines Tomato 0052/0092, Corn ×3), then `queue_gpu1.sh` K/H in reverse order.
- CPU: 2D C-stage/I for all specimens (`logs/phaseB_ip2p_cpu.log`).
- **When they finish, regenerate everything:**
  ```bash
  cd ~/Desktop/Research/external/SAGE-Extend; export CUDA_VISIBLE_DEVICES=1; E=~/micromamba/envs/sam3d/bin/python
  $E src/analysis/phase_summary.py; for s in $(ls results/phaseA | grep __); do $E src/editing2d/mask_perview.py $s; $E src/metrics/crossview.py $s; done
  $E src/quality_control/qc.py; $E src/analysis/metrics2d.py; for f in src/figures/fig*.py; do $E $f; done
  $E src/analysis/make_reports.py; ~/Desktop/Research/.venv/bin/python src/analysis/compute_audit.py
  ~/Desktop/Research/.venv/bin/python src/analysis/make_tables.py; ~/Desktop/Research/.venv/bin/python src/analysis/make_latex.py
  CUDA_VISIBLE_DEVICES= $E src/figures/slides.py   # advisor deck (Overleaf 6ac80070): copy figures/slides/*.{pdf,png} -> figs/results/, numbers.tex -> generated/, push
  ```

## Pending (next)
1. Finish alignment on all 11 (+Citrus 0004, Mango); write `reports/01_camera_alignment_fixes.md`.
2. Confirm F-GE-1 on a real run (Tomato 0015 F): verify.json `prompt_delivered=true`, plus asset and renders.
3. **Stage 2:** geometry editor (G2: explicit deformation field on Gaussians with ARAP/smoothness/anchor; target = measurable curl/pucker). One-specimen pilot on Peach 0087.
4. Structured KB with stages (Stage 3).
5. Reports 00 and 02–10, figures, PDF.

## Known blockers
- No expert raters, so biological validity is unvalidated.
- Multi-view / 3D reference data are out of scope by user decision, so novel views are synthetic and unverified.
- Disk: ~4.7 GB free.

## Exact next commands
```bash
cd ~/Desktop/Research/external/SAGE-Extend
cat ~/Desktop/Research/logs/alignment.log                 # alignment progress
tail -5 results/pilot/runner_gpu0.jsonl                   # GE runs
~/micromamba/envs/gedit/bin/python -m pytest -q -p no:cacheprovider tests/test_ge_config.py
```
