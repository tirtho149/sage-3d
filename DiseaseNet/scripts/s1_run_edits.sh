#!/usr/bin/env bash
# Stage 1: fixed 2D edits (uint8 mask) for all 60 Stage-0 assets, kb + generic; seed 0 first, then seeds 1-2.
# Split over two GPUs by asset parity. Resumable. Logs: logs/stage1/edits_gpu{0,1}.log
cd "$(dirname "$0")/.."
VENV=~/Desktop/Research/.venv/bin/python
L=$($VENV -c "import sys; sys.path.insert(0,'.'); from src.generation.registry import assets; print(' '.join(sorted(assets().asset_id)))")
A0=$(echo $L | tr ' ' '\n' | awk 'NR%2==1' | paste -sd,); A1=$(echo $L | tr ' ' '\n' | awk 'NR%2==0' | paste -sd,)
run() { g=$1; a=$2; for s in 0 1,2; do CUDA_VISIBLE_DEVICES=$g $VENV -m src.generation.edit2d --out data/synthetic/stage1/edits2d --seeds $s --assets $a; done; }
run 0 "$A0" > logs/stage1/edits_gpu0.log 2>&1 & run 1 "$A1" > logs/stage1/edits_gpu1.log 2>&1 & wait
echo "[$(date '+%F %T')] edits done"
