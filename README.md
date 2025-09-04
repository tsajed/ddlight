# Running ddlight Active-Learning Docking Jobs
This README explains how to launch an active-learning experiment for molecular docking using the provided SLURM script and configuration file.

## 1. Set up your environment

### Conda environment
Ensure that a `dds` conda environment is available and activated before running the job.

### Hardware / scheduler requirements
The job script requests one GPU, four CPU cores, and large memory on a SLURM cluster. Adjust the `#SBATCH` directives as needed for your system.

## 2. Configure the experiment
Edit `configs/params.yml` to match your paths and experiment settings.

### Global settings
- `expt_name`: name used for the wandb experiment.
- `code_path`: absolute path to the `ddlight` repository.
- `project_path`: directory where output files and results will be written.
- `project_name`: grouping folder for multiple runs on the same target protein.
- `dock_pgm`: docking backend (`vina` or `glide`).
- `topK`: number of best molecules kept after the final docking run.
- `grid_file`: path to the docking grid file.
- `glide_input_template`: template for Glide docking inputs.
- `dataset_path`: dataset of SMILES and initial docking scores.
- `model_architecture`: model to train (e.g., `advanced_molformer`).
- `target`: target protein identifier (PDB ID).
- `schrodinger_path`: path to the Schrödinger installation (needed for Glide).
- `env_name`: name of the conda environment to activate.
- `loss`: training loss function (`crossentropy` or `focal`).
- `dock_logs_path`: directory to store docking logs.

### Active-learning parameters
- `acquisition_function`: strategy for selecting new molecules (`bald`, `unc`, `entropy`, `random`, `greedy`, `ucb`, etc.).
- `initial_budget`: number of molecules docked in the first iteration.
- `al_budget`: number of molecules added at each active-learning iteration.
- `n_iterations`: total number of active-learning iterations.
- `model_retrain`: retrain the model from scratch at each iteration (`True`) or continue training (`False`).
- `test_rand_samples`: dock a random sample of molecules at each iteration.
- `drop_db`: remove informative inactive molecules to keep the dataset manageable.
- `drop_db_recall`: only drop inactives if recall improves.

### Model hyperparameters
- `num_gpus`: number of GPUs used for model training.
- `num_layers`: depth of the model.
- `n_epochs`: training epochs per iteration.
- `lr`: learning rate.
- `batch_size`: batch size for training.
- `optimizer`: optimization algorithm.
- `random_seed`: seed for reproducibility.
- `patience`: early stopping patience in epochs.

### Thresholds
- `fixed_cutoff`: if `True`, use an absolute docking score cutoff; otherwise use a dynamic percentile threshold.
- `given_cutoff`: docking score threshold used when `fixed_cutoff` is `True`.

## 3. Launch the job
Submit the SLURM script:

```bash
sbatch AcqFnComparision_molformer.sh
```
The script activates the conda environment and runs the main Python program with the configuration file.

## 4. Run the Python program manually (optional)
You can bypass SLURM and run the experiment directly:

```bash
python AcqFnComparision_molformer.py configs/params.yml
```
`AcqFnComparision_molformer.py` expects a single argument pointing to the YAML configuration and uses it to orchestrate data preparation, model training, and docking iterations.
