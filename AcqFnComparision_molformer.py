import os
import pickle
import numpy as np
import torch
from collections import Counter
from torch.utils.data import DataLoader
import sys
#sys.path.append('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL')
sys.path.append('/groups/cherkasvgrp/tsajed/ddlight')
from gpuvina import get_vina_scores_mul_gpu
from model_train import predict_with_model, train_docking_model, train_molformer_model, predict_with_molformer, train_molformer_model_wandb
from metrics import log_metrics, compute_enrichment
from easydict import EasyDict
import yaml
import time
import argparse
from itertools import product
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import pandas as pd

from ALHelpers_molformer import (load_smiles_dockscore, 
    filter_data_by_dockscore, save_docking_scores, run_first_iteration,
    initialize_model, prepare_datasets, VirtHitsCutoff, run_subsequent_iterations_mul_gpu)
import wandb
wandb.login(key="27f458bdad899ca70cdd866458b86f4f423d0054")


# Utility functions
def load_config(file_path):
    with open(file_path, 'r') as f:
        return EasyDict(yaml.safe_load(f))

def prepare_and_filter_data(it0_data, smiles_2_dockscore_gt):
    """Filter data based on docking scores for train, validation, and test splits."""
    for split in ['train', 'validation', 'test']:
        data = getattr(it0_data, split)
        data.smiles, data.mol_ids, data.features = filter_data_by_dockscore(data, smiles_2_dockscore_gt)

def compute_docking_scores(it0_data, molecule_df, config, smiles_2_dockscore_gt, output_dir):
    """Compute docking scores using Vina across train, validation, and test splits."""
    for split in ['train', 'validation', 'test']:
        data = getattr(it0_data, split)
        data.dock_scores = get_vina_scores_mul_gpu(
            data.smiles, molecule_df, config, num_gpus=config.model_hps.num_gpus,
            output_dir=output_dir, dockscore_gt=smiles_2_dockscore_gt
        )
        save_docking_scores(
            data.dock_scores, f"{output_dir}/it0_{split}_dock_feat_scores.pkl",
            data.mol_ids, data.features, data.smiles
        )

def assign_labels(it0_data, dock_cutoff: float):
    """Assign docking labels based on cutoff threshold."""
    for split in ['train', 'validation', 'test']:
        data = getattr(it0_data, split)
        data.dock_labels = [1 if value < dock_cutoff else 0 for value in data.dock_scores]
        # (data.dock_scores < dock_cutoff).astype(int)
        print(f'Initial {split}_labels distribution:', Counter(data.dock_labels))

def prepare_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=1024):
    """Prepare PyTorch DataLoaders for training, validation, and test datasets."""
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    )

def train_and_evaluate_model(docking_model, train_loader, val_loader, test_loader, it0_data, config, output_path, rank, world_size):
    """Train, save, and evaluate the docking model."""
    if config.global_params.model_architecture in ('mlp3K','mlpTx'):
        docking_model, _, _, val_proba, val_labels = train_docking_model(
            docking_model, train_loader, val_loader, al_iteration=0, num_epochs=config.model_hps.n_epochs,
            lr= config.model_hps.lr, weight_decay=0.01, model_save_path=output_path, config=config, rank=rank
        )
    elif config.global_params.model_architecture in ('molformer','enhanced_molformer','advanced_molformer'):
        docking_model, _, _ , val_proba, val_labels = train_molformer_model(
            docking_model, train_loader, val_loader, num_epochs=config.model_hps.n_epochs,
            lr= config.model_hps.lr, weight_decay=0.01, model_save_path=output_path, config=config, patience=config.model_hps.patience,
            rank=rank, world_size=world_size
        )
    torch.save(docking_model.state_dict(), f"{output_path}/docking_model_{config.global_params.model_architecture}.pt")

    # Save training data
    with open(f"{output_path}/train_data_class.pkl", "wb") as f:
        pickle.dump({0: it0_data}, f)

    # Evaluate model
    if config.global_params.model_architecture in ('mlp3K','mlpTx'):
        pred_val_labels, pred_val_proba, _ = predict_with_model(docking_model, val_loader, acquisition_function='greedy') # greedy to give correct metircs
    elif config.global_params.model_architecture in ('molformer','enhanced_molformer','advanced_molformer'):
        pred_val_labels, pred_val_proba, _ = predict_with_molformer(docking_model, val_loader, acquisition_function='greedy', rank=rank) # greedy to give correct metircs

    val_metrics = log_metrics(it0_data.validation.dock_labels, pred_val_labels, 0)
    if rank ==0:
        print("Iteration 0 Validation Metrics:", val_metrics)

    if config.global_params.model_architecture in ('mlp3K','mlpTx'):
        pred_test_labels, pred_test_proba, _ = predict_with_model(docking_model, test_loader, acquisition_function='greedy') # greedy to give correct metircs
    elif config.global_params.model_architecture in ('molformer','enhanced_molformer','advanced_molformer'):
        pred_test_labels, pred_test_proba, _ = predict_with_molformer(docking_model, test_loader, acquisition_function='greedy') # greedy to give correct metircs
    true_test_labels = (it0_data.test.dock_scores < it0_data.train.cutoff).astype(int)     # np.float64(config.threshold_params.fixed_cutoff)).astype(int)
    test_metrics = log_metrics(true_test_labels, pred_test_labels, 0)
    test_metrics.update({'enrichment_1':compute_enrichment(pred_test_proba,it0_data.test.dock_labels)})
    if rank==0:
        print(f"Iteration 0 Test Metrics: {test_metrics}")

    return val_metrics, test_metrics, val_proba, val_labels


# Main function
def main(rank, world_size):
    parser = argparse.ArgumentParser(description="Acq Fn experiment configurations.")
    parser.add_argument("config_path", type=str)
    # parser.add_argument("seed_idx", type=int,  help="Index for experiment configuration")

    args = parser.parse_args()
    config_path = args.config_path #'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/config/params.yml'
    config = load_config(config_path)

    """Main function to run the first iteration and subsequent active learning loops."""
    base_path = f"{config.global_params.project_path}/{config.global_params.project_name}"
    os.makedirs(base_path, exist_ok=True)

    with open(base_path+'/config.pkl','wb') as f:
        pickle.dump(config,f)

    # Load data
    target = config.global_params.target
    if target in ['jak2', 'braf', 'parp1','fa7', '5ht1b', 'pgk1', 'pgk2']:
        molecule_df = pickle.load(open(config.global_params.dataset_path,'rb')) # SMILES, ID
        #molecule_df = molecule_df[molecule_df[f'{target}_dockscores']!=0]
        # molecule_df.drop(columns=["indices"], inplace=True)
        smiles_2_dockscore_gt = None
    elif target == 'mt1r':
        molecule_df = pd.read_csv('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/data/lsd_dock_mt1r/40M_mols_w_dockscores.csv')
    elif target == 'mpro':
        with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/data/lsd_dock_mpro/778M_mols_w_dockscores.pkl','rb') as f:
            molecule_df = pickle.load(f)
        with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/data/lsd_dock_mpro/smiles_2_dockscore_gt.pkl','rb') as f:
            smiles_2_dockscore_gt = pickle.load(f)
    # smiles_2_dockscore_gt = dict(zip(molecule_df["smiles"], molecule_df[f"{target}_dockscores"])) # If this uncommented, then no docking, things will be fast. If this commented out, it will dock everytime
    
    used_zinc_ids = set()
    # Run first iteration
    
    # Initialize model and train
    docking_model = initialize_model(config.global_params.model_architecture)

    if config.threshold_params.given_cutoff:
        it0_data, used_zinc_ids = run_first_iteration(config, config.al_params.initial_budget, molecule_df, used_zinc_ids, smiles_2_dockscore_gt,
                                                    cutoff=config.threshold_params.given_cutoff,tokenizer=docking_model.tokenizer)
    else:
        it0_data, used_zinc_ids = run_first_iteration(config, config.al_params.initial_budget, molecule_df, used_zinc_ids, smiles_2_dockscore_gt,
                                                    cutoff=None,tokenizer=docking_model.tokenizer)
    dock_cutoff = it0_data.train.cutoff


    print("Train labels:", Counter(it0_data.train.dock_labels))
    print("Val labels:", Counter(it0_data.validation.dock_labels))
    print("Test labels:", Counter(it0_data.test.dock_labels))
    
    # Prepare datasets and DataLoaders
    train_dataset, val_dataset, test_dataset = prepare_datasets(it0_data, docking_model, config)
    train_loader, val_loader, test_loader = prepare_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=512)

    #wrap in DDP
    docking_model = docking_model.to(rank) if world_size>1 else docking_model.to(torch.device('cuda'))
    if world_size>1:
        docking_model = DDP(docking_model, device_ids=[rank])

    model_save_path = base_path+'/iteration_0/' 
    os.makedirs(model_save_path, exist_ok=True)

    val_metrics, test_metrics, val_proba, val_labels = train_and_evaluate_model(docking_model, train_loader, val_loader, test_loader, it0_data, config, model_save_path, rank, world_size)
 
    # Run active learning iterations
    run_subsequent_iterations_mul_gpu(
        docking_model, molecule_df, dock_cutoff, val_metrics, test_metrics,
        num_iterations=config.al_params.n_iterations, batch_size=100_000,
        top_k=config.al_params.al_budget, config=config, train_data_class={0: it0_data},
        used_zinc_ids=used_zinc_ids, rank=rank, world_size=world_size,
        smiles_2_dockscore_gt=smiles_2_dockscore_gt, tokenizer=docking_model.tokenizer,
        val_proba=val_proba, val_labels = val_labels
    )

if __name__ == '__main__':
    world_size = 1 #torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    if world_size > 1:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    main(0,1)
    