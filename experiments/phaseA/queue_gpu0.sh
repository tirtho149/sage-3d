#!/bin/bash
# GPU0 queue after Phase A: (1) Phase B K/H one specimen per class, (2) baseline runner for Citrus/Mango, (3) remaining K/H.
R=$HOME/Desktop/Research/external/SAGE-Extend; PY=$HOME/Desktop/Research/.venv/bin/python; cd $R
L="Citrus__Canker__0012 Citrus__Canker__0016 Citrus__Canker__0004 Mango__Anthracnose__0049 Mango__Anthracnose__0094 Mango__Anthracnose__0044"
while kill -0 822469 2>/dev/null || pgrep -f "^/home/tirtho/Desktop/Research/.venv/bin/python experiments/phaseA/run_ge_arms.py Peach__Leaf_Curl__0087" >/dev/null || kill -0 936168 2>/dev/null; do sleep 60; done
echo "$(date) phase B priority K/H"
for x in "Tomato__Leaf_Mold__0015 Tomato__Leaf_Mold" "Corn__Smut__0012 Corn__Smut" "Citrus__Canker__0012 Citrus__Canker" "Mango__Anthracnose__0049 Mango__Anthracnose"; do
  set -- $x; CUDA_VISIBLE_DEVICES=0 $PY experiments/phaseA/run_ge_arms.py $1 $2 K,H 0 200; done
for s in $L; do printf 0 > results/pilot/$s/.lock; done
echo "$(date) baseline runner Citrus/Mango"; env -u PYTORCH_CUDA_ALLOC_CONF $PY experiments/runners/pilot_runner.py 0 Citrus,Mango > $HOME/Desktop/Research/logs/pilot_runner_gpu0_b.log 2>&1
echo "$(date) remaining K/H"
for x in "Peach__Leaf_Curl__0006 Peach__Leaf_Curl" "Peach__Leaf_Curl__0063 Peach__Leaf_Curl" "Tomato__Leaf_Mold__0052 Tomato__Leaf_Mold" "Tomato__Leaf_Mold__0092 Tomato__Leaf_Mold" \
         "Corn__Smut__0025 Corn__Smut" "Corn__Smut__0052 Corn__Smut" "Citrus__Canker__0016 Citrus__Canker" "Citrus__Canker__0004 Citrus__Canker__leaf" \
         "Mango__Anthracnose__0094 Mango__Anthracnose" "Mango__Anthracnose__0044 Mango__Anthracnose"; do
  set -- $x; CUDA_VISIBLE_DEVICES=0 $PY experiments/phaseA/run_ge_arms.py $1 $2 K,H 0 200; done
echo "$(date) GPU0 queue done"
