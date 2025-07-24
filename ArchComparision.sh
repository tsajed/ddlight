#!/bin/bash
#SBATCH --array=0-1
#SBATCH -o /home/mkpandey/arch_comp_experiments_%A_%a.out
#SBATCH --job-name=ArchComp
#SBATCH --partition=gpu-long
#SBATCH --time=24-00:00:00
#SBATCH --mem=100000M
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# List of architectures and targets
# architecture_list=('gcn' 'gat' 'gin' 'gine' 'mlp6K' 'mlp3K' 'mlpTx' 'rf' 'molformer' 'enhanced_molformer' 'advanced_molformer')
architecture_list=('gat' 'gin')
target_list=('parp1')
# target_list=('5ht1b' 'parp1' 'fa7' 'jak2' 'braf')

# Calculate total number of experiments
num_architectures=${#architecture_list[@]}
num_targets=${#target_list[@]}
total_experiments=$((num_architectures * num_targets))

# Ensure SLURM_ARRAY_TASK_ID does not exceed total combinations
if [ $SLURM_ARRAY_TASK_ID -ge $total_experiments ]; then
    echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID exceeds total experiments=$total_experiments"
    exit 1
fi

# Compute target and architecture index
target_idx=$((SLURM_ARRAY_TASK_ID / num_architectures))
arch_idx=$((SLURM_ARRAY_TASK_ID % num_architectures))

# Select target and architecture for this task
target=${target_list[$target_idx]}
architecture=${architecture_list[$arch_idx]}

echo "Running experiment for Target: $target and Architecture: $architecture"

# Activate the appropriate Conda environment
source ~/anaconda3/etc/profile.d/conda.sh
if [[ "$architecture" == "molformer" || "$architecture" == "enhanced_molformer" || "$architecture" == "advanced_molformer" ]]; then
    conda activate alex
else
    conda activate dds
fi

# Run the experiment
python /groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/Experiments/ArchComparision.py --target $target --architecture $architecture

# Export library path if required
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mkpandey/anaconda3/lib/
