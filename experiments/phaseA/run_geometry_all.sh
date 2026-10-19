#!/bin/bash
# Phase B geometry arms (J KB-geometry, G generic RMS-matched) for all pilot specimens; abstains where the KB has no operator.
R=$HOME/Desktop/Research/external/SAGE-Extend; E=$HOME/micromamba/envs/sam3d; cd $R
declare -A KEY=([Peach__Leaf_Curl]=Peach__Leaf_Curl [Tomato__Leaf_Mold]=Tomato__Leaf_Mold [Corn__Smut]=Corn__Smut [Mango__Anthracnose]=Mango__Anthracnose [Citrus__Canker]=Citrus__Canker)
for d in results/pilot/sam3d/*/seed0; do s=$(basename $(dirname $d)); c=${s%__*}; k=${KEY[$c]}; [ "$s" = "Citrus__Canker__0004" ] && k=Citrus__Canker__leaf
  [ -f $d/splat_ge.ply ] || $E/bin/python src/gaussian_editing/sanitize_splat.py $d/splat.ply $d/splat_ge.ply > /dev/null
  echo "$(date +%T) $s ($k)"; CUDA_VISIBLE_DEVICES=${1:-0} $E/bin/python experiments/phaseA/run_geometry.py $s $k 2>&1 | grep -v -i "warn\|pynvml" | tail -1 | cut -c1-200
done
