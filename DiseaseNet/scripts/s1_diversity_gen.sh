#!/usr/bin/env bash
# Parent-diversity generation: PHOTO2D for all (old + new) parents (CPU), then KB edits (seed 0) for the new parents
# on GPU 1, then KB2D renders. Log: logs/stage1/diversity_gen.log
cd "$(dirname "$0")/.."
export LIDRA_SKIP_INIT=true CONDA_PREFIX=$HOME/micromamba/envs/sam3d
log() { echo "[$(date '+%F %T')] $*"; }
log "PHOTO2D renders"; CUDA_VISIBLE_DEVICES= ~/micromamba/envs/sam3d/bin/python -m src.generation.s1_arms --arms PHOTO2D --views V8 --stage s1all > logs/stage1/div_photo2d.log 2>&1
log "KB edits new parents"; NEW=$(ls data/synthetic/stage1/inputs | paste -sd,)
CUDA_VISIBLE_DEVICES=1 ~/Desktop/Research/.venv/bin/python -m src.generation.edit2d --stage s1all --out data/synthetic/stage1/edits2d --seeds 0 --assets $NEW > logs/stage1/div_edits.log 2>&1
log "KB2D renders"; CUDA_VISIBLE_DEVICES= ~/micromamba/envs/sam3d/bin/python -m src.generation.s1_arms --arms KB2D --views V8 --stage s1all > logs/stage1/div_kb2d.log 2>&1
log "diversity generation done"
