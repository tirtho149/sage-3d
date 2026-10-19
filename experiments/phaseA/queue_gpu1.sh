#!/bin/bash
# GPU1: after its baseline runner exits, help with the remaining K/H runs (reverse order; claims prevent duplicates).
R=$HOME/Desktop/Research/external/SAGE-Extend; PY=$HOME/Desktop/Research/.venv/bin/python; cd $R
while pgrep -f "^/home/tirtho/Desktop/Research/.venv/bin/python experiments/runners/pilot_runner.py 1 all" >/dev/null; do sleep 60; done
echo "$(date) GPU1 worker start"
for x in "Mango__Anthracnose__0044 Mango__Anthracnose" "Mango__Anthracnose__0094 Mango__Anthracnose" "Citrus__Canker__0004 Citrus__Canker__leaf" \
         "Citrus__Canker__0016 Citrus__Canker" "Corn__Smut__0052 Corn__Smut" "Corn__Smut__0025 Corn__Smut" "Tomato__Leaf_Mold__0092 Tomato__Leaf_Mold" \
         "Tomato__Leaf_Mold__0052 Tomato__Leaf_Mold" "Peach__Leaf_Curl__0063 Peach__Leaf_Curl" "Peach__Leaf_Curl__0006 Peach__Leaf_Curl" \
         "Mango__Anthracnose__0049 Mango__Anthracnose" "Citrus__Canker__0012 Citrus__Canker" "Corn__Smut__0012 Corn__Smut" "Tomato__Leaf_Mold__0015 Tomato__Leaf_Mold"; do
  set -- $x; CUDA_VISIBLE_DEVICES=1 $PY experiments/phaseA/run_ge_arms.py $1 $2 K,H 0 200; done
echo "$(date) GPU1 worker done"
