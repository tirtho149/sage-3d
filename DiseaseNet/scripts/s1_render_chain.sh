#!/usr/bin/env bash
# Stage 0R render chain (GPU 1): waits for render_more, renders replicates 1-2 of the no-edit arms; waits for the
# fixed 2D edits; renders replicates 0-2 of the edit arms. Log: logs/stage1/render_chain.log
cd "$(dirname "$0")/.."
export LIDRA_SKIP_INIT=true CONDA_PREFIX=$HOME/micromamba/envs/sam3d
PY=~/micromamba/envs/sam3d/bin/python
log() { echo "[$(date '+%F %T')] $*"; }
while pgrep -f "^bash scripts/s1_render_more.sh" > /dev/null; do sleep 30; done
log "no-edit arms, replicates 1,2"
CUDA_VISIBLE_DEVICES=1 $PY -m src.generation.s1_arms --arms U3D,TEX3D,GEO3D,TEXGEO3D,PHOTO2D,HARM3D,PLATE2D --views V8 --edit-seeds 1,2
log "wait for seed-0 edits (120 files)"
while [ $(ls data/synthetic/stage1/edits2d/*/*_s0.png 2>/dev/null | wc -l) -lt 120 ]; do sleep 60; done
log "edit arms, replicate 0"
CUDA_VISIBLE_DEVICES=1 $PY -m src.generation.s1_arms --arms GEN2D,KB2D,GEN3D,KBAPP3D,KB3D --views V8 --edit-seeds 0
log "wait for all edits"
while pgrep -f "^bash scripts/s1_run_edits.sh" > /dev/null; do sleep 60; done
log "edit arms, replicates 1,2"
CUDA_VISIBLE_DEVICES=1 $PY -m src.generation.s1_arms --arms GEN2D,KB2D,GEN3D,KBAPP3D,KB3D --views V8 --edit-seeds 1,2
log "render chain done"
