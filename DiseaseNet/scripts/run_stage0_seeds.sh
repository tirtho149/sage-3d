#!/usr/bin/env bash
# Stage 0: extra T3 fine-tune seeds (1, 2) for all 5 arms, real + IID fold 0; same fixed schedule as seed 0.
# T0 (frozen probe + logistic regression, lbfgs) is deterministic given the data, so it is not re-run.
# Resumable: skips finished result files. usage: nohup bash scripts/run_stage0_seeds.sh > logs/stage0_seeds.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
VENV=~/Desktop/Research/.venv/bin/python
log() { echo "[$(date '+%F %T')] $*"; }
run_arms() { gpu=$1; shift; for s in 1 2; do for arm in "$@"; do for f in real 0; do
  tag=$([ "$f" = real ] && echo real || echo fold0)
  [ -f results/E0/t3_${arm}_${tag}_s${s}.json ] && continue
  if [ "$f" = real ]; then CUDA_VISIBLE_DEVICES=$gpu $VENV -m src.training.finetune $arm --seed $s >> logs/stage0_t3_seeds.log 2>&1
  else CUDA_VISIBLE_DEVICES=$gpu $VENV -m src.training.finetune $arm --fold 0 --seed $s >> logs/stage0_t3_seeds.log 2>&1; fi
  log "  T3 $arm $tag seed $s done (GPU $gpu, exit $?)"; done; done; done; }
log "T3 seeds 1-2 start"
run_arms 0 PHOTO KB3D GEN2D & run_arms 1 U3D KB2D & wait
log "T3 seeds 1-2 finished"
