#!/usr/bin/env bash
# Stage 1 fine-tuning study (strict S0). Phase A: hyper-parameter search on synthetic LOPO folds 0 and 1 for KB2D and
# KB3D (replicate 0). Phase B (scripts/s1_ft_select.py): pick per arm x mode the config with the best mean LOPO macro-F1
# and its epoch count, then train on all parents with 3 seeds and evaluate DEV. No DEV / FINAL number is used in Phase A.
# Runs on both GPUs after the 2D edits finish. Log: logs/stage1/ft_study.log
cd "$(dirname "$0")/.."
VENV=~/Desktop/Research/.venv/bin/python
log() { echo "[$(date '+%F %T')] $*"; }
while pgrep -f "src.generation.edit2d" > /dev/null; do sleep 60; done
log "phase A: LOPO search"
CONFIGS=(
 "--mode linear --lr-head 1e-3"
 "--mode linear --lr-head 3e-3"
 "--mode lastk --k 2 --lr-bb 1e-5"
 "--mode lastk --k 2 --lr-bb 3e-5"
 "--mode lastk --k 2 --lr-bb 3e-5 --ls 0.1"
 "--mode lora --r 8 --lr-bb 1e-4"
 "--mode lora --r 8 --lr-bb 5e-4"
 "--mode full --lr-bb 1e-5 --llrd 0.75"
 "--mode full --lr-bb 2e-5 --llrd 0.75"
 "--mode full --lr-bb 2e-5"
)
run_queue() { gpu=$1; shift; for job in "$@"; do
  set -- $job; arm=$1; fold=$2; ci=$3
  tag="${arm}_c${ci}_fold${fold}"
  [ -f results/stage1/ft_search/$tag.json ] && continue
  CUDA_VISIBLE_DEVICES=$gpu $VENV -m src.s1.finetune $arm ${CONFIGS[$ci]} --epochs 15 --val-fold $fold --tag $tag --out results/stage1/ft_search >> logs/stage1/ft_search_gpu$gpu.log 2>&1
  log "  $tag done (GPU $gpu)"; done; }
Q0=(); Q1=()
for ci in "${!CONFIGS[@]}"; do Q0+=("KB3D 0 $ci" "KB3D 1 $ci"); Q1+=("KB2D 0 $ci" "KB2D 1 $ci"); done
run_queue 0 "${Q0[@]}" & run_queue 1 "${Q1[@]}" & wait
log "phase A done"
$VENV scripts/s1_ft_select.py > logs/stage1/ft_select.log 2>&1
log "phase B: selected configs x 3 seeds on all parents"
bash results/stage1/ft_search/phase_b_gpu0.sh & bash results/stage1/ft_search/phase_b_gpu1.sh & wait
log "ft study done"
