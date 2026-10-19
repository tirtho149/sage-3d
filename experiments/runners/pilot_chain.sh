#!/bin/bash
# Unattended pilot schedule (beast0). Logs: ~/Desktop/Research/logs/pilot_chain.log
R=$HOME/Desktop/Research/external/SAGE-Extend; L=$HOME/Desktop/Research/logs
E=$HOME/micromamba/envs/sam3d; export CONDA_PREFIX=$E CUDA_HOME=$E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
n0() { ls $R/results/pilot/sam3d/*/seed0/run_meta.json 2>/dev/null | wc -l; }
echo "$(date) waiting for 15 seed-0 reconstructions"; until [ "$(n0)" -ge 15 ]; do sleep 120; done
echo "$(date) seed-0 done; stopping full-seed batch"; pkill -f "run_sam3d_batch.py experiments/configs/pilot.yaml"; sleep 20
echo "$(date) starting GPU1 pilot runner"; cd $R && ~/Desktop/Research/.venv/bin/python experiments/runners/pilot_runner.py 1 all > $L/pilot_runner_gpu1.log 2>&1
echo "$(date) GPU1 runner finished; waiting for GPU0 runner"; while pgrep -f "pilot_runner.py 0 all" >/dev/null; do sleep 120; done
echo "$(date) seeds 1-2 stability subset"; CUDA_VISIBLE_DEVICES=1 $E/bin/python src/reconstruction/run_sam3d_batch.py experiments/configs/pilot_seeds_subset.yaml $R/results/pilot/sam3d > $L/pilot_sam3d_seeds.log 2>&1
for d in $R/results/pilot/sam3d/*/seed[12]; do s=$(basename $(dirname $d)); [ -f $d/e1.json ] || LIDRA_SKIP_INIT=true CUDA_VISIBLE_DEVICES=1 $E/bin/python src/rendering/input_view.py $d $R/data/inputs/$s >> $L/pilot_chain.log 2>&1; done
echo "$(date) CHAIN DONE"
