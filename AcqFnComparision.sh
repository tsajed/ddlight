#!/bin/bash
#SBATCH --array=0-7 #11  # Array indices correspond to the 21 experiments (0 to 20)
#SBATCH -o /home/mkpandey/DDS_AL_AcqTest100_100Ktest_cyclic_topK_%A_%a.out
#SBATCH --job-name=Acq100
#SBATCH --partition=gpu-bigmem
#SBATCH --time=24:00:00
#SBATCH --mem=100000M
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Activate the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dds

# Run the Python script with the seed_idx argument as SLURM_ARRAY_TASK_ID
python /groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/Experiments/AcqFnComparision.py $SLURM_ARRAY_TASK_ID
