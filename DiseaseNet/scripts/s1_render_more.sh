#!/usr/bin/env bash
# Stage 1: HARM3D + PLATE2D (V8), and view-set ablation renders (NARROW, WIDE) for U3D, TEX3D. Log: logs/stage1/render_more.log
cd "$(dirname "$0")/.."
export LIDRA_SKIP_INIT=true CONDA_PREFIX=$HOME/micromamba/envs/sam3d
PY=~/micromamba/envs/sam3d/bin/python
CUDA_VISIBLE_DEVICES=1 $PY -m src.generation.s1_arms --arms HARM3D,PLATE2D --views V8
for vs in NARROW WIDE; do CUDA_VISIBLE_DEVICES=1 $PY -m src.generation.s1_arms --arms U3D,TEX3D --views $vs; done
echo "[$(date '+%F %T')] render_more done"
