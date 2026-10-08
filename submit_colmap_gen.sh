#!/bin/bash
#SBATCH --job-name=colmap_gen
#SBATCH --output=/work/mech-ai-scratch/UshB/sage-extend/colmap_gen_%j.log
#SBATCH --error=/work/mech-ai-scratch/UshB/sage-extend/colmap_gen_%j.err
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --qos=scavenger
#SBATCH --account=mech-ai-scavenger

echo "Started: $(date) | Node: $SLURMD_NODENAME"
nvidia-smi | head -12

source /work/mech-ai/ushashi/AFM3d/miniconda3/etc/profile.d/conda.sh
conda activate sam3d-objects

python /work/mech-ai-scratch/UshB/sage-extend/make_colmap_from_ply.py

echo "Finished: $(date)"
