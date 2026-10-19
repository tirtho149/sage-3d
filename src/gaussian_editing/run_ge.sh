#!/bin/bash
# GaussianEditor edit-n2n for one asset via a generated, validated config (src/gaussian_editing/ge_config.py).
# Hyperparameters = collaborator's submit_ge_edit.sh (A100 run plan), except max_steps (arg).
# usage: run_ge.sh <splat.ply> <views_dir> <prompt> <seed> <max_steps> <exp_root>
#    or: run_ge.sh <config.yaml>          (config made with ge_config.py make)
# Exit codes: 0 = asset saved AND parsed prompt == intended prompt; 2 = preflight failed; 3 = run/verification failed.
set -o pipefail
E=$HOME/micromamba/envs/gedit; GE=$HOME/Desktop/Research/external/GaussianEditor
D=$(cd "$(dirname "$0")" && pwd)
export LD_LIBRARY_PATH=$E/lib CUDA_HOME=$E PATH=$E/bin:$PATH WANDB_MODE=disabled
unset PYTORCH_CUDA_ALLOC_CONF   # torch 2.0.1 rejects expandable_segments (set for SAM 3D elsewhere)
if [ $# -eq 6 ]; then
  mkdir -p "$6"; printf '%s' "$3" > "$6/prompt.txt"
  CFG="$6/ge_config.yaml"
  $E/bin/python $D/ge_config.py make "$CFG" "$6/prompt.txt" "$4" "$5" "$1" "$2" "$6" || exit 2
else
  CFG="$1"; $E/bin/python $D/ge_config.py check "$CFG" || exit 2
fi
cd $GE
$E/bin/python launch.py --config "$CFG" --train --gpu 0; rc=$?
ROOT=$($E/bin/python -c "import json,sys;print(json.load(open(sys.argv[1]))['exp_root'])" "$CFG.meta.json")
TRIAL=$(ls -dt "$ROOT"/edit-n2n/edit@* 2>/dev/null | head -1)
[ -n "$TRIAL" ] || { echo "run_ge: no trial dir (launch rc=$rc)"; exit 3; }
$E/bin/python $D/ge_config.py verify "$TRIAL" "$CFG" | tee "$TRIAL/verify.json" || exit 3
exit 0
