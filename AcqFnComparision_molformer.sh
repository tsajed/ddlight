#!/bin/bash
#SBATCH -o logs/DDS_final_dock_%A_%a.out  # DDS_AL_2Mrun_AllTgt_%A_%a.out
#SBATCH --job-name=ddsRecall
#SBATCH --partition=gpu-long
#SBATCH --time=484-24:00:00
#SBATCH --mem=500000M
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dds

# Run the Python script with the seed_idx argument as SLURM_ARRAY_TASK_ID
python -u AcqFnComparision_molformer.py configs/params.yml
