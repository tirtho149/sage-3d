#!/usr/bin/env bash
# Stage 0R probe driver: waits for renders, runs T0 probes (DEV + LOPO) for every arm / replicate / mixture,
# renders and probes the cross-class arms after their edits. Log: logs/stage1/probe_driver.log
cd "$(dirname "$0")/.."
VENV=~/Desktop/Research/.venv/bin/python; SAM=~/micromamba/envs/sam3d/bin/python
export LIDRA_SKIP_INIT=true CONDA_PREFIX=$HOME/micromamba/envs/sam3d
log() { echo "[$(date '+%F %T')] $*"; }
ARMS="PHOTO2D GEN2D KB2D U3D HARM3D TEX3D GEN3D KBAPP3D GEO3D KB3D TEXGEO3D"
MIX="MIX0 MIX25 MIX50 MIX75 MIX100"
probe() { CUDA_VISIBLE_DEVICES=0 $VENV experiments/S1/run_probes.py stage0r "$@" 2>&1 | grep -viE "warn|xformers"; }
log "wait for edit-arm replicate 0"; until grep -q "wait for all edits" logs/stage1/render_chain.log; do sleep 60; done
log "probes replicate 0"; probe $ARMS $MIX
log "cross-class: wait for kbx edits"; while pgrep -f "edit2d --out data/synthetic/stage1/edits2d --seeds 0 --cross" > /dev/null; do sleep 60; done
log "render cross arms (GPU 0)"
CUDA_VISIBLE_DEVICES=0 $SAM -m src.generation.s1_arms --arms XPHOTO2D,XGEN2D,XKB2D,XKBAPP3D,XKB3D --views V8 --edit-seeds 0 > logs/stage1/render_cross.log 2>&1
log "cross-class probes (16-class cohort)"
CUDA_VISIBLE_DEVICES=0 $VENV experiments/S1/run_probes.py cross cohort=x16 XPHOTO2D XGEN2D XKB2D XKBAPP3D XKB3D PHOTO2D KB2D GEN2D U3D KBAPP3D KB3D 2>&1 | grep -viE "warn|xformers"
log "wait for render chain"; until grep -q "render chain done" logs/stage1/render_chain.log; do sleep 60; done
log "probes replicates 1,2"; probe $(for r in 1 2; do for a in $ARMS $MIX; do echo -n "$a~$r "; done; done)
log "probe driver done"
