#!/usr/bin/env bash
# Stage-0 orchestration (resumable; every step skips finished work). Logs: logs/stage0_*.log
# usage: nohup bash scripts/run_stage0.sh > logs/stage0_pipeline.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
VENV=~/Desktop/Research/.venv/bin/python
SAM=~/micromamba/envs/sam3d/bin/python
export LIDRA_SKIP_INIT=true CONDA_PREFIX=$HOME/micromamba/envs/sam3d
log() { echo "[$(date '+%F %T')] $*"; }

wait_for() { while pgrep -f "$1" >/dev/null; do sleep 60; done; }

log "1. wait for the running reconstruction batch"; wait_for "src/generation/reconstruct.py"
log "2. reconstruct remaining inputs (amended classes)"
$VENV -m src.provenance.verify_inputs stage0
CUDA_VISIBLE_DEVICES=1 $SAM src/generation/reconstruct.py >> logs/stage0_reconstruct.log 2>&1
grep -c '"status": "ok"' data/synthetic/stage0/assets/*/run_meta.json | grep -c ':1' | xargs -I{} echo "  assets ok: {}"
log "3. wait for 2D edits"; wait_for "src.generation.edit2d"
CUDA_VISIBLE_DEVICES=0 $VENV -m src.generation.edit2d >> logs/stage0_edit2d.log 2>&1      # completes any missing edit
log "4. 3D arms (U3D, KB3D)"; CUDA_VISIBLE_DEVICES=1 $SAM src/generation/render3d.py >> logs/stage0_render3d.log 2>&1
log "5. 2D arms (PHOTO, KB2D, GEN2D)"; wait_for "src/generation/build2d.py"; $SAM src/generation/build2d.py >> logs/stage0_build2d.log 2>&1
log "6. integrity tests"; $VENV -m pytest -q tests/ 2>&1 | tail -3
log "7. T0 frozen probe"; CUDA_VISIBLE_DEVICES=0 $VENV experiments/E0_feasibility/run_t0.py 2>&1 | grep -vE "Warn|xFormers" | tail -8
log "8. T3 full fine-tune (real + IID fold 0), arms split over the two GPUs"
run_arms() { gpu=$1; shift; for arm in "$@"; do for f in real 0; do
  tag=$([ "$f" = real ] && echo real || echo fold0)
  [ -f results/E0/t3_${arm}_${tag}_s0.json ] && continue
  if [ "$f" = real ]; then CUDA_VISIBLE_DEVICES=$gpu $VENV -m src.training.finetune $arm >> logs/stage0_t3.log 2>&1
  else CUDA_VISIBLE_DEVICES=$gpu $VENV -m src.training.finetune $arm --fold 0 >> logs/stage0_t3.log 2>&1; fi
  log "  T3 $arm $tag done (GPU $gpu)"; done; done; }
run_arms 0 PHOTO KB3D GEN2D & run_arms 1 U3D KB2D & wait
log "9. Stage-0 pipeline finished"
