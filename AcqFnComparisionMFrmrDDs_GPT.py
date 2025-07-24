import os
import pickle
import numpy as np
import torch
from collections import Counter
from torch.utils.data import DataLoader
import sys
sys.path.append('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL')
from gpuvina import get_vina_scores_mul_gpu
# from model_train import predict_with_model, train_docking_model, train_molformer_model, predict_with_molformer
from metrics import log_metrics
from easydict import EasyDict
import yaml
import time
import argparse
from itertools import product
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from ALHelpers import (load_smiles_dockscore, 
    filter_data_by_dockscore, save_docking_scores, run_first_iteration,
    initialize_model, prepare_datasets, run_subsequent_iterations)
print('importing')
import wandb
wandb.login()


# Utility functions
def load_config(file_path):
    with open(file_path, 'r') as f:
        return EasyDict(yaml.safe_load(f))
    
# def load_data(molecule_data_path: str, target :str, smiles_dockscore_path: str):
#     """Load molecule data and ground truth docking scores."""
#     t0 = time.time()
#     molecule_df = pickle.load(open(molecule_data_path, 'rb'))
#     molecule_df = molecule_df[molecule_df[f'{target}_dockscores']!=0]
#     print('Loaded molecule_df from pickle in', time.time() - t0)
#     smiles_2_dockscore_gt = load_smiles_dockscore(smiles_dockscore_path)
#     return molecule_df, smiles_2_dockscore_gt

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
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    )

# from tqdm import tqdm
# def train_molformer_model(
#     model, train_loader, val_loader, num_epochs=40, lr=0.001,
#     weight_decay=0.01, model_save_path='/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/',
#     config=None, patience=5, rank=0
# ):
#     """
#     Train the MoLFormer model using PyTorch with train-time augmentation, label smoothing, 
#     and FocalLoss.

#     :param model: PyTorch MoLFormer instance.
#     :param train_loader: DataLoader for training data.
#     :param val_loader: DataLoader for validation data.
#     :param num_epochs: Number of training epochs.
#     :param lr: Learning rate.
#     :param weight_decay: Weight decay for the optimizer.
#     :param model_save_path: Path to save the trained model.
#     :param config: Configuration dictionary for logging and other settings.
#     """
#     device = next(model.parameters()).device #torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
#     # model = model.to(device)
#     print('model train.py device from model ', next(model.parameters()).device)
#     if not os.path.exists(model_save_path):
#         os.makedirs(model_save_path)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

#     if config.global_params.loss == "crossentropy":
#         criterion = torch.nn.CrossEntropyLoss()
#         print("Using CrossEntropy Loss")
#     elif config.global_params.loss == "focal":
#         criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
#         print("Using Focal Loss")
#     else:
#         raise ValueError(f"Unsupported loss type: {config.global_params.loss}")
    
#      # Early stopping parameters
#     best_val_loss = float("inf")
#     best_epoch = -1
#     early_stop_counter = 0
#     best_model_state = None
#     model.train()
#     for epoch in range(num_epochs):
#         # Training loop
#         total_train_loss = 0
#         for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
#             optimizer.zero_grad()
#             input_ids = batch["input_ids"].to(rank) #(device)
#             attention_mask = batch["attention_mask"].to(rank) #(device)
#             labels = batch["labels"].to(rank) #(device)

#             # Forward pass
#             outputs = model(input_ids, attention_mask=attention_mask)

#             # Compute loss
#             loss = criterion(outputs, labels)
#             total_train_loss += loss.item()

#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()

#         scheduler.step()
#         avg_train_loss = total_train_loss / len(train_loader)
#         print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

#         avg_train_loss_tensor = torch.tensor(avg_train_loss).to(rank)
#         print('avg_train_loss_tensor', avg_train_loss_tensor)
#         dist.all_reduce(avg_train_loss_tensor, op=dist.ReduceOp.SUM)
        
#         # keys = sorted(info_vals.keys())
#         # train_loss_list = torch.zeros(1).to(rank)
#         # world_size = 4
#         # for i, k in enumerate(keys):
#         #     try:
#         #         all_info_vals[i] = info_vals[k]
#         #     except Exception as e:
#         #         print(e,k)
#         #         # raise e
#         # dist.all_reduce(all_info_vals, op=dist.ReduceOp.SUM)
#         # for i, k in enumerate(keys):
#         #     info_vals[k] = all_info_vals[i].item() / world_size
#         # if rank == 0:
#         #     wandb.log(info_vals)
#         print('consolidated train loss  ', avg_train_loss_tensor)

 
#         # # Clear GPU memory
#         # torch.cuda.empty_cache()

#     # Save the best model
#     model_save_name = f'{model_save_path}/{config.global_params.model_architecture}_best_model_val_loss_{best_val_loss:.4f}.pt'
#     # print('model_train.py model save name ',model_save_name)
#     torch.save(best_model_state, model_save_name)
#     print(f"Best model saved to {model_save_name}")

#     # Load the best model before returning
#     model.load_state_dict(best_model_state)

#     return model, best_val_loss#, val_metrics




def train_and_evaluate_model(docking_model, train_loader, val_loader, test_loader, it0_data, config, output_path, rank):
    """Train, save, and evaluate the docking model."""
    if config.global_params.model_architecture in ('mlp3K','mlpTx'):
        docking_model, _, _ = train_docking_model(
            docking_model, train_loader, val_loader, al_iteration=0, num_epochs=config.model_hps.n_epochs,
            lr= config.model_hps.lr, weight_decay=0.01, model_save_path=output_path, config=config, rank=rank
        )
    elif config.global_params.model_architecture in ('molformer','enhanced_molformer','advanced_molformer'):
        docking_model, _, _ = train_molformer_model(
            docking_model, train_loader, val_loader, num_epochs=config.model_hps.n_epochs,
            lr= config.model_hps.lr, weight_decay=0.01, model_save_path=output_path, config=config, patience=config.model_hps.patience,
            rank=rank
        )
    torch.save(docking_model.state_dict(), f"{output_path}/docking_model_{config.global_params.model_architecture }.pt")

    # Save training data
    with open(f"{output_path}/train_data_class.pkl", "wb") as f:
        pickle.dump({0: it0_data}, f)

    # Evaluate model
    if config.global_params.model_architecture in ('mlp3K','mlpTx'):
        pred_val_labels, pred_val_proba, _ = predict_with_model(docking_model, val_loader, acquisition_function=config.al_params.acquisition_function)
    elif config.global_params.model_architecture in ('molformer','enhanced_molformer','advanced_molformer'):
        pred_val_labels, pred_val_proba, _ = predict_with_molformer(docking_model, val_loader, acquisition_function=config.al_params.acquisition_function)

    val_metrics = log_metrics(it0_data.validation.dock_labels, pred_val_labels, 0)
    print("Iteration 0 Validation Metrics:", val_metrics)

    if config.global_params.model_architecture in ('mlp3K','mlpTx'):
        pred_test_labels, pred_test_proba, _ = predict_with_model(docking_model, test_loader, acquisition_function=config.al_params.acquisition_function)
    elif config.global_params.model_architecture in ('molformer','enhanced_molformer','advanced_molformer'):
        pred_test_labels, pred_test_proba, _ = predict_with_molformer(docking_model, test_loader, acquisition_function=config.al_params.acquisition_function)
    true_test_labels = (it0_data.test.dock_scores < np.float64(config.threshold_params.fixed_cutoff)).astype(int)
    test_metrics = log_metrics(true_test_labels, pred_test_labels, 0)
    print(f"Iteration 0 Test Metrics: {test_metrics}")

    return val_metrics, test_metrics


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# # Main function
# def main(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
#     # initialize the process group
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)
#     print('rank is ', rank)
#     # Argument parser for seed_idx
#     parser = argparse.ArgumentParser(description="Acq Fn experiment configurations.")
#     parser.add_argument("seed_idx", type=int,  help="Index for experiment configuration")
#     args = parser.parse_args()
#     idx = args.seed_idx
#     config_path = '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/config/params.yml'
#     config = load_config(config_path)

#     targets_list = ['jak2', 'braf', 'parp1','fa7', '5ht1b']
#     acq_fn_list = ["random", "entropy", "least_confidence", "margin", "greedy", "ucb", "unc", "bald"]
#     arch_list = ['advanced_molformer', 'mlpTx']
#     expt_list = list(product(targets_list,acq_fn_list, arch_list))
#     expt = expt_list[idx]
#     print(expt)
#     config.global_params.model_architecture = expt[2]
#     config.global_params.target = expt[0]
#     config.al_params.acquisition_function = expt[1]

#     """Main function to run the first iteration and subsequent active learning loops."""
#     base_path = f"{config.global_params.project_path}/{config.global_params.project_name}"
#     os.makedirs(base_path, exist_ok=True)

#     # Load data
#     target = config.global_params.target
#     molecule_df = pickle.load(open(f'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Projects/dock_{target}/iteration_0/2M_mols_w_dockscores_{target}.pkl','rb'))
#     molecule_df = molecule_df[molecule_df[f'{target}_dockscores']!=0]
#     smiles_2_dockscore_gt=  dict(zip(molecule_df["smiles"], molecule_df[f"{target}_dockscores"]))
#     dock_cutoff = np.float64(config.threshold_params.fixed_cutoff)

#     # smiles_dockscore_path = f"/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Enamine2M_smiles2dockscore_{config.global_params.target}.pkl"
#     # molecule_df, smiles_2_dockscore_gt = load_data(molecule_data_path, config.global_params.target, smiles_dockscore_path)
#     used_zinc_ids = set()
#     # Run first iteration
#     it0_data, used_zinc_ids = run_first_iteration(config.al_params.initial_budget, molecule_df, used_zinc_ids, smiles_2_dockscore_gt,dock_cutoff)
#     # prepare_and_filter_data(it0_data, smiles_2_dockscore_gt)

#     # # Compute docking scores
#     # iteration_output_dir = f"{base_path}/iteration_0/vina_results"
#     # os.makedirs(iteration_output_dir, exist_ok=True)
#     # compute_docking_scores(it0_data, molecule_df, config, smiles_2_dockscore_gt, iteration_output_dir)

#     # # Set docking score cutoff and assign labels
#     # dock_cutoff = np.float64(config.threshold_params.fixed_cutoff)
#     # assign_labels(it0_data, dock_cutoff)

#     print("Train labels:", Counter(it0_data.train.dock_labels))
#     print("Val labels:", Counter(it0_data.validation.dock_labels))
#     print("Test labels:", Counter(it0_data.test.dock_labels))

#     # Initialize model and train
#     docking_model = initialize_model(config.global_params.model_architecture)
    
#     # Prepare datasets and DataLoaders
#     train_dataset, val_dataset, test_dataset = prepare_datasets(it0_data, docking_model, config)
#     train_loader, val_loader, test_loader = prepare_dataloaders(train_dataset, val_dataset, test_dataset)

#     #wrap in DDP
#     docking_model = docking_model.to(rank)
#     docking_model = DDP(docking_model, device_ids=[rank])

#     val_metrics, test_metrics = train_and_evaluate_model(docking_model, train_loader, val_loader, test_loader, it0_data, config, base_path, rank)

#     # Run active learning iterations
#     run_subsequent_iterations(
#         docking_model, molecule_df, dock_cutoff, val_metrics, test_metrics,
#         num_iterations=config.al_params.n_iterations, batch_size=100_000,
#         top_k=config.al_params.al_budget, config=config, train_data_class={0: it0_data},
#         used_zinc_ids=used_zinc_ids, rank=rank,
#         smiles_2_dockscore_gt=smiles_2_dockscore_gt
#     )


import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import copy

def train_one_molformer_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs):
    """Train the model for one epoch using Distributed Data Parallel."""
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", disable=(dist.get_rank() != 0)):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_loader)
    if dist.get_rank() == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
    
    return avg_train_loss


def val_one_molformer_epoch(model, val_loader, criterion, device, epoch, num_epochs):
    """Validate the model for one epoch using Distributed Data Parallel."""
    model.eval()
    val_losses = []
    predicted_labels_list = []
    all_labels = []
    correct, total = 0, 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}", disable=(dist.get_rank() != 0)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())

            _, predicted_labels = torch.max(outputs.data, 1)
            predicted_labels_list.extend(predicted_labels.cpu().detach().numpy())
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.flatten().cpu().detach().numpy())

    avg_val_loss = sum(val_losses) / len(val_losses)
    val_accuracy = correct / total

    val_metrics = log_metrics(all_labels, predicted_labels_list, 0)

    if dist.get_rank() == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, val metrics {val_metrics}")

    return avg_val_loss, val_metrics


def train_molformer_model(rank, world_size, model, train_dataset, val_dataset, test_dataset, num_epochs=40, lr=0.001,
                          weight_decay=0.01, model_save_path='/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/',
                          config=None, patience=5):
    
    """Train the MoLFormer model using PyTorch Distributed Data Parallel (DDP)."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    print('rank is ', rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    if not os.path.exists(model_save_path) and rank == 0:
        os.makedirs(model_save_path)


    train_loader, val_loader, test_loader = prepare_dataloaders(train_dataset, val_dataset, test_dataset)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    if config.global_params.loss == "crossentropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif config.global_params.loss == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
    else:
        raise ValueError(f"Unsupported loss type: {config.global_params.loss}")

    best_val_loss = float("inf")
    best_epoch = -1
    early_stop_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        avg_train_loss = train_one_molformer_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
        avg_val_loss, val_metrics = val_one_molformer_epoch(model, val_loader, criterion, device, epoch, num_epochs)

        scheduler.step()

        if rank == 0:  # Only rank 0 process saves models
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                early_stop_counter = 0
                best_model_state = copy.deepcopy(model.module.state_dict())  # DDP wraps model
                print(f"New best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
            else:
                early_stop_counter += 1
                print(f"No improvement in validation loss for {early_stop_counter} consecutive epochs.")

            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best epoch was {best_epoch + 1}.")
                break

    if rank == 0:
        model_save_name = f'{model_save_path}/{config.global_params.model_architecture}_best_model_val_loss_{best_val_loss:.4f}.pt'
        torch.save(best_model_state, model_save_name)
        print(f"Best model saved to {model_save_name}")

    dist.destroy_process_group()
    return best_val_loss


def main():
    world_size = 2 #torch.cuda.device_count()
    config_path = '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/config/params.yml'
    config = load_config(config_path)
    """Main function to run the first iteration and subsequent active learning loops."""
    base_path = f"{config.global_params.project_path}/{config.global_params.project_name}"
    os.makedirs(base_path, exist_ok=True)

    # Load data
    target = config.global_params.target
    molecule_df = pickle.load(open(f'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Projects/dock_{target}/iteration_0/2M_mols_w_dockscores_{target}.pkl','rb'))
    molecule_df = molecule_df[molecule_df[f'{target}_dockscores']!=0]
    smiles_2_dockscore_gt=  dict(zip(molecule_df["smiles"], molecule_df[f"{target}_dockscores"]))
    dock_cutoff = np.float64(config.threshold_params.fixed_cutoff)

    # smiles_dockscore_path = f"/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Enamine2M_smiles2dockscore_{config.global_params.target}.pkl"
    # molecule_df, smiles_2_dockscore_gt = load_data(molecule_data_path, config.global_params.target, smiles_dockscore_path)
    used_zinc_ids = set()
    # Run first iteration
    it0_data, used_zinc_ids = run_first_iteration(config.al_params.initial_budget, molecule_df, used_zinc_ids, smiles_2_dockscore_gt,dock_cutoff)
    # prepare_and_filter_data(it0_data, smiles_2_dockscore_gt)

    # # Compute docking scores
    # iteration_output_dir = f"{base_path}/iteration_0/vina_results"
    # os.makedirs(iteration_output_dir, exist_ok=True)
    # compute_docking_scores(it0_data, molecule_df, config, smiles_2_dockscore_gt, iteration_output_dir)

    # # Set docking score cutoff and assign labels
    # dock_cutoff = np.float64(config.threshold_params.fixed_cutoff)
    # assign_labels(it0_data, dock_cutoff)

    print("Train labels:", Counter(it0_data.train.dock_labels))
    print("Val labels:", Counter(it0_data.validation.dock_labels))
    print("Test labels:", Counter(it0_data.test.dock_labels))

    # Initialize model and train
    docking_model = initialize_model(config.global_params.model_architecture)
    
    # Prepare datasets and DataLoaders
    train_dataset, val_dataset, test_dataset = prepare_datasets(it0_data, docking_model, config)

    mp.spawn(train_molformer_model, args=(world_size, docking_model, train_dataset, val_dataset, test_dataset, config.model_hps.n_epochs,
                                          config.model_hps.lr, 0.01, base_path, config, config.model_hps.patience),nprocs=world_size, join=True)

if __name__ == "__main__":
    main()





# if __name__ == '__main__':
#     world_size = 4 #torch.cuda.device_count()
#     print('Let\'s use', world_size, 'GPUs!')
#     if world_size > 1:
#         mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
#     main(0,1)
    