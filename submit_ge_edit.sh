#!/bin/bash
#SBATCH --job-name=ge_leaf_edit
#SBATCH --output=/work/mech-ai-scratch/UshB/sage-extend/ge_leaf_edit_%j.log
#SBATCH --error=/work/mech-ai-scratch/UshB/sage-extend/ge_leaf_edit_%j.err
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --qos=scavenger
#SBATCH --account=mech-ai-scavenger

echo "Started: $(date) | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi | head -12

source /work/mech-ai/ushashi/AFM3d/miniconda3/etc/profile.d/conda.sh
conda activate GaussianEditor

cd /work/mech-ai-scratch/UshB/sage-extend/GaussianEditor

python launch.py \
    --config configs/edit-n2n.yaml \
    --train \
    --gpu 0 \
    trainer.max_steps=2000 \
    system.prompt_processor.prompt="add brown circular fungal disease lesion spots to the leaf" \
    system.seg_prompt="leaf" \
    system.gs_source=/work/mech-ai-scratch/UshB/sage-extend/leaf_output/leaf_splat.ply \
    data.source=/work/mech-ai-scratch/UshB/sage-extend/leaf_colmap \
    system.max_densify_percent=0.01 \
    system.anchor_weight_init_g0=0.05 \
    system.anchor_weight_init=0.1 \
    system.anchor_weight_multiplier=1.3 \
    system.gs_lr_scaler=5 \
    system.gs_final_lr_scaler=5 \
    system.color_lr_scaler=5 \
    system.opacity_lr_scaler=2 \
    system.scaling_lr_scaler=2 \
    system.rotation_lr_scaler=2 \
    system.loss.lambda_anchor_color=0 \
    system.loss.lambda_anchor_geo=50 \
    system.loss.lambda_anchor_scale=50 \
    system.loss.lambda_anchor_opacity=50 \
    system.densify_from_iter=100 \
    system.densify_until_iter=1501 \
    system.densification_interval=100 \
    exp_root_dir=/work/mech-ai-scratch/UshB/sage-extend/leaf_edited

echo "Finished: $(date)"
