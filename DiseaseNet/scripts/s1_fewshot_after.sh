#!/usr/bin/env bash
# runs the S1 few-shot study once the replicate-0 arms are probed (probe driver passed that step)
cd "$(dirname "$0")/.."
until grep -q "cross-class: wait for kbx edits" logs/stage1/probe_driver.log; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 ~/Desktop/Research/.venv/bin/python experiments/S1/run_fewshot.py PHOTO2D KB2D GEN2D U3D TEX3D KBAPP3D KB3D MIX50 2>&1 | grep -viE "warn|xformers"
echo "[$(date '+%F %T')] fewshot done"
