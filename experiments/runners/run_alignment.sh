#!/bin/bash
# A0-A3 alignment study on every seed-0 SAM 3D reconstruction. Resumable. GPU arg (default 0).
R=$HOME/Desktop/Research/external/SAGE-Extend; E=$HOME/micromamba/envs/sam3d; cd $R
for d in results/pilot/sam3d/*/seed0; do s=$(basename $(dirname $d)); [ -f $d/splat.ply ] || continue
  [ -f results/alignment/$s/alignment.json ] && continue
  CUDA_VISIBLE_DEVICES=${1:-0} CONDA_PREFIX=$E CUDA_HOME=$E LIDRA_SKIP_INIT=true $E/bin/python src/camera_alignment/align.py $d data/inputs/$s results/alignment/$s > results/alignment/$s.log 2>&1 || echo "FAIL $s"
  echo "$(date +%T) $s rc=$?"
done
