# DD-Light Code Overview

This document describes the main modules and workflow of the **ddlight** repository.

## Repository Layout

- **`architecture.py`** – Contains PyTorch neural network models used for docking score prediction.  Models include:
  - `DockingModel6K`, `DockingModel3K`, `DockingModelTx` – MLP-based classifiers.
  - `MoLFormer`, `EnhancedMoLFormer`, `AdvancedMoLFormer` – Models based on IBM's MoLFormer architecture.
  - `MyModel` – Graph neural network architecture (requires DGL).

- **`graph_dataset.py`** – Utilities to build datasets for graph or MolFormer models.  Includes feature extraction from SMILES and dataset classes `MyDataset`, `MolFormerDataset` and `HybridMolTxDataset`.

- **`metrics.py`** – Metric helpers for computing F1/AUC/precision/recall and enrichment factor.

- **`database_helper.py`** – Functions to convert molecule IDs to fingerprints when using a database or DataFrame (`molids_2_fps`).

- **`model_train.py`** – Training and inference routines for the above models.  Functions include `train_docking_model`, `train_molformer_model`, and prediction helpers implementing various active‑learning acquisition functions.

- **`gpuvina.py`** – Wrapper around Vina-GPU for docking.  Defines `QuickVina2GPU` which prepares input PDBQT files and runs docking in batches.

- **`glide_dock.py`** – Helper to run Schrodinger Glide docking on a Slurm cluster.  Handles conformer generation, job submission and extraction of docking scores.

- **`ALHelpers_molformer.py`** – Core logic for active learning.  Key routines:
  - `initialize_model` – Instantiates the chosen architecture.
  - `run_first_iteration` – Samples initial training/validation/test sets, docks them (Glide or Vina) and assigns labels based on a cutoff.
  - `run_inference` / `run_inference_scatter` – Use the model to score unseen molecules and select top candidates according to an acquisition function.
  - `run_subsequent_iterations_mul_gpu` – Full active‑learning loop over multiple iterations with periodic retraining and optional database reduction.

- **`ALMolformer_inference_helper.py`** – Helper script executed on multiple GPUs/Slurm array jobs to run inference on splits of the molecule library.

- **`AcqFnComparision_molformer.py`** – Entry point that loads configuration from `configs/params.yml`, performs the first iteration, trains the initial model, and calls the active‑learning loop.

- **`FinalDocking.py`** – After the final iteration, docks the top predicted molecules and returns the best scoring compounds.

- **`run_vina.py`** – Stand‑alone script to dock a batch of SMILES using `QuickVina2GPU` (used when running jobs on Slurm).

- **`Final_Regression/`** – Notebooks and helper scripts for regression analysis and generating figures used in the manuscript.

- **`configs/params.yml`** – YAML configuration controlling experiments (dataset paths, docking program, active learning parameters, model hyperparameters, etc.).

## Typical Workflow

1. **Configure experiment** – edit `configs/params.yml` to point to datasets, define the target receptor, choose docking program (Vina/Glide), model architecture and active learning parameters.
2. **Launch experiment** – run `AcqFnComparision_molformer.py configs/params.yml` (usually via the provided Slurm script `AcqFnComparision_molformer.sh`).
3. **First iteration** (`run_first_iteration`)
   - Randomly sample a subset of molecules from the large library as train/validation/test.
   - Dock these molecules (using Vina or Glide) if ground‑truth scores are not provided.
   - Compute a cutoff based on the top 1 % docking scores and assign binary labels (1 = active).
   - Prepare dataset objects for the chosen model type.
   - Train the model and evaluate on validation/test sets.
4. **Active learning loop** (`run_subsequent_iterations_mul_gpu`)
   - Repeatedly score the remaining library.  Depending on the acquisition function (`greedy`, `ucb`, `unc`, `bald`, etc.), select the most promising molecules.
   - Optionally drop confidently predicted inactives from the database to reduce size.
   - Dock the newly acquired molecules to obtain ground‑truth scores and labels.
   - Update the training data and retrain or fine‑tune the model.
   - Logging and metrics are recorded (optionally via Weights & Biases).
5. **Final docking** (`FinalDocking.get_topK_mols`)
   - After the last iteration, combine all docked molecules and high‑scoring virtual hits.
   - Dock an expanded set of top candidates and write `final_dock_res.csv` containing the best compounds.

## Inputs and Outputs

- **Input dataset** – A pickle/CSV containing at least `smiles`, `zinc_id`, optionally fingerprint indices (`indices`) and precomputed docking scores.
- **Docking grids** – For Vina, coordinates are defined in `gpuvina.py` (`TARGETS`). For Glide, provide `grid_file` and template via config.
- **Outputs** – Each experiment creates a folder `${project_path}/${project_name}` with subfolders `iteration_0`, `iteration_1`, …
  - Docking results (`vina_results` or `docked/`), model checkpoints, and `train_data_class.pkl` with all datasets.
  - After completion, `final_dock_res.csv` lists the top docked molecules.

## Function Highlights

- `molids_2_fps` in `database_helper.py` converts sparse fingerprint indices to dense binary vectors using either a DataFrame or SQL cursor【F:database_helper.py†L85-L137】.
- `train_docking_model` implements training with optional focal loss, early stopping and returns metrics【F:model_train.py†L1-L122】.
- `initialize_model` in `ALHelpers_molformer.py` selects between MLP and MolFormer variants based on config【F:ALHelpers_molformer.py†L202-L247】.
- The active learning loop orchestrated by `run_subsequent_iterations_mul_gpu` handles inference, database reduction, retraining and logging across iterations【F:ALHelpers_molformer.py†L771-L1103】.

For further details consult the inline comments in each file.