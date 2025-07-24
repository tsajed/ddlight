#!/bin/bash
#SBATCH --array=0 #159 #-7 #11  # Array indices correspond to the 21 experiments (0 to 20)
#SBATCH -o /home/mkpandey/DDS_AL_mt1r_fixedcutoff_1pct_Regression_%A_%a.out  # DDS_AL_2Mrun_AllTgt_%A_%a.out
#SBATCH --job-name=mproReg
#SBATCH --partition=gpuA
#SBATCH --time=484-24:00:00
#SBATCH --mem=500000M
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Activate the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dds

# Run the Python script with the seed_idx argument as SLURM_ARRAY_TASK_ID
# python /groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/Experiments/Final_Regression/Final_Regression_inf.py /groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Projects/DDS_AL_2M_2_dynamicVAL_mt1r_least_confidence_advanced_molformer_False_True/iteration_49/docking_model_advanced_molformer.pt /groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Projects/DDS_AL_2M_2_dynamicVAL_mt1r_least_confidence_advanced_molformer_False_True
python /groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/Experiments/Final_Regression/Final_Regression_inf.py /groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Projects/DDS_AL_2M_2_dynamicVAL_mpro_bald_advanced_molformer_False_True/iteration_6/docking_model_advanced_molformer.pt /groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Projects/DDS_AL_2M_2_dynamicVAL_mpro_bald_advanced_molformer_False_True