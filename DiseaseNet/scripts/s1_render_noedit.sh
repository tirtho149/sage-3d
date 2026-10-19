#!/usr/bin/env bash
# Stage 0R: arms that need no 2D edit, Stage-0 view set V8 x 4 conditions, all 60 assets. Log: logs/stage1/render_noedit.log
cd "$(dirname "$0")/.."
export LIDRA_SKIP_INIT=true CONDA_PREFIX=$HOME/micromamba/envs/sam3d
CUDA_VISIBLE_DEVICES=1 ~/micromamba/envs/sam3d/bin/python -m src.generation.s1_arms --arms U3D,TEX3D,GEO3D,TEXGEO3D,PHOTO2D --views V8
echo "[$(date '+%F %T')] render_noedit done"
