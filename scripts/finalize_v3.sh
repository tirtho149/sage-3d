#!/bin/bash
# Regenerate every v3 output from result files: demo summary, advisor figures, deck macros, deck figures, report 15.
# usage: bash scripts/finalize_v3.sh <deck_dir>    (deck_dir = local clone of Overleaf 6ac800709ea44535f867ab26)
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); E=~/micromamba/envs/sam3d/bin/python; DECK=${1:?deck dir}
cd "$R"
CUDA_VISIBLE_DEVICES= $E src/analysis/demo_summary.py > /dev/null
CUDA_VISIBLE_DEVICES=${GPU:-1} $E src/figures/demo_figs.py          # figure (a) needs the GPU renderer (input camera)
CUDA_VISIBLE_DEVICES= $E src/figures/slides.py > /dev/null
$E src/analysis/assemble_report15.py
cp figures/slides/{arms_0087.png,crossview.pdf,crossview_strip.png,e1_iou.pdf,e1_panel.png,failures.png,kb2d_strip.png,quant2d.pdf,quant3d.pdf} "$DECK/figs/results/"
cp figures/slides/{demo_a_directional.pdf,demo_b_galls.pdf,demo_c_underside.pdf,demo_d_downstream.pdf} "$DECK/figs/results/"
# the full 4-metric downstream figure stays in results/downstream_minimal; the slide uses demo_d_downstream
cp figures/slides/numbers.tex figures/slides/demo_numbers.tex "$DECK/generated/"
echo "finalize_v3: done $(date)"
