#!/bin/bash
#SBATCH --array=0  # Adjusted to match the number of valid experiments
#SBATCH -o /home/mkpandey/dataset_comp_experiments_%A_%a.out
#SBATCH --job-name=DatasetComp
#SBATCH --partition=gpu-bigmem
#SBATCH --time=24-00:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# List of architectures, filtered targets, and training sizes
architecture_list=('advanced_molformer') 
target_list=('parp1')
train_size_dict=(
    "parp1:200_000"
)

# Generate valid (target, train_size) pairs
target_train_size_pairs=()
for entry in "${train_size_dict[@]}"; do
    target="${entry%%:*}"
    sizes="${entry#*:}"
    IFS=',' read -ra size_array <<< "$sizes"
    for size in "${size_array[@]}"; do
        target_train_size_pairs+=("$target:$size")
    done
done

num_architectures=${#architecture_list[@]}
num_pairs=${#target_train_size_pairs[@]}
total_experiments=$(( num_architectures * num_pairs ))

# Ensure SLURM_ARRAY_TASK_ID does not exceed total combinations
if [ $SLURM_ARRAY_TASK_ID -ge $total_experiments ]; then
    echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID exceeds total experiments=$total_experiments"
    exit 1
fi

# Compute target-train_size index and architecture index
pair_idx=$((SLURM_ARRAY_TASK_ID / num_architectures))
arch_idx=$((SLURM_ARRAY_TASK_ID % num_architectures))

# Extract target and train size from selected pair
IFS=':' read -r target train_size <<< "${target_train_size_pairs[$pair_idx]}"
architecture=${architecture_list[$arch_idx]}

echo "Running experiment for Target: $target, Architecture: $architecture, Train Size: $train_size"

# Activate the appropriate Conda environment
source ~/anaconda3/etc/profile.d/conda.sh
if [[ "$architecture" == "enhanced_molformer" || "$architecture" == "advanced_molformer" ]]; then
    conda activate alex
else
    conda activate dds
fi

# Run the experiment with the specified parameters
python /groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/Experiments/DatasetSizeComparision.py --target $target --architecture $architecture --train_size $train_size

# Export library path if required
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mkpandey/anaconda3/lib/










# #!/bin/bash
# #SBATCH --array=0-139
# #SBATCH -o /home/mkpandey/dataset_comp_experiments_%A_%a.out
# #SBATCH --job-name=DatasetComp
# #SBATCH --partition=gpu-bigmem
# #SBATCH --time=24-00:00:00
# #SBATCH --mem=8G
# #SBATCH --cpus-per-task=4
# #SBATCH --gres=gpu:1

# # List of architectures, targets, and training sizes
# architecture_list=('mlp3K' 'mlpTx' 'enhanced_molformer' 'advanced_molformer')
# target_list=('5ht1b' 'parp1' 'fa7' 'jak2' 'braf')
# train_size_list=('10_000' '20_000' '30_000' '50_000' '100_000' '200_000' '500_000')

# # Calculate total number of experiments
# num_architectures=${#architecture_list[@]}
# num_targets=${#target_list[@]}
# num_train_sizes=${#train_size_list[@]}
# total_experiments=$(( num_architectures * num_targets * num_train_sizes))

# # Ensure SLURM_ARRAY_TASK_ID does not exceed total combinations
# if [ $SLURM_ARRAY_TASK_ID -ge $total_experiments ]; then
#     echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID exceeds total experiments=$total_experiments"
#     exit 1
# fi

# # Compute target, architecture, and train size index
# target_idx=$((SLURM_ARRAY_TASK_ID / (num_architectures * num_train_sizes)))
# train_size_idx=$(( (SLURM_ARRAY_TASK_ID % (num_architectures * num_train_sizes)) / num_architectures ))
# arch_idx=$((SLURM_ARRAY_TASK_ID % num_architectures))

# # Select target, architecture, and train size for this task
# target=${target_list[$target_idx]}
# architecture=${architecture_list[$arch_idx]}
# train_size=${train_size_list[$train_size_idx]}

# echo "Running experiment for Target: $target, Architecture: $architecture, Train Size: $train_size"

# # Activate the appropriate Conda environment
# source ~/anaconda3/etc/profile.d/conda.sh
# if [[ "$architecture" == "enhanced_molformer" || "$architecture" == "advanced_molformer" ]]; then
#     conda activate alex
# else
#     conda activate dds
# fi

# # Run the experiment with the specified parameters
# python /groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/Experiments/DatasetSizeComparision.py --target $target --architecture $architecture --train_size $train_size

# # Export library path if required
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mkpandey/anaconda3/lib/
