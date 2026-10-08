#!/bin/bash
#SBATCH --job-name=leaf_sam3d
#SBATCH --output=/work/mech-ai-scratch/UshB/sage-extend/leaf_sam3d_%j.log
#SBATCH --error=/work/mech-ai-scratch/UshB/sage-extend/leaf_sam3d_%j.err
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --qos=scavenger
#SBATCH --account=mech-ai-scavenger

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi | head -12

source /work/mech-ai/ushashi/AFM3d/miniconda3/etc/profile.d/conda.sh
conda activate sam3d-objects

cd /work/mech-ai-scratch/UshB/sage-extend/sam-3d-objects
python /work/mech-ai-scratch/UshB/sage-extend/run_leaf_sam3d.py

echo "Job finished: $(date)"
