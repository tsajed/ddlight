import time
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
from easydict import EasyDict
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import torch
import sys
#sys.path.append('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL')
sys.path.append('/groups/cherkasvgrp/tsajed/ddlight')
from gpuvina import get_vina_scores_mul_gpu, QuickVina2GPU
from glide_dock import get_glide_scores_mul_gpu
from graph_dataset import MolFormerDataset
from database_helper import molids_2_fps
from metrics import log_metrics, compute_enrichment
from model_train import predict_with_model, train_docking_model, predict_with_molformer, train_molformer_model
from architecture import MoLFormer, DockingModel3K, EnhancedMoLFormer, AdvancedMoLFormer, DockingModelTx
import random
import torch.distributed as dist
import heapq
import subprocess
from sklearn.metrics import precision_recall_curve
from torch.nn.parallel import DistributedDataParallel as DDP
import glob
import re
from pathlib import Path
import os
from FinalDocking import get_topK_mols
# import wandb
# wandb.login(key= '50c4542f2f5338f2591116005f2e2c8bd9f4d6d6')

# def load_molecule_data(pickle_path):
#     t0 = time.time()
#     molecule_df = pickle.load(open(pickle_path, 'rb'))
#     molecule_df = molecule_df[molecule_df[f'{target}_dockscores']!=0]
#     print('Loaded molecule_df from pickle in', time.time() - t0)
#     return molecule_df

def clear_iter(iteration_path, model_clean=False):
    delete_files = glob.glob(iteration_path+'/used_zinc_ids_*.pkl')
    delete_files.extend(glob.glob(iteration_path+'/split_mol_df_*.csv'))
    delete_files.extend(glob.glob(iteration_path+'/allmols_*.pkl'))
    delete_files.extend(glob.glob(iteration_path+'/*.out'))
    delete_files.extend(glob.glob(iteration_path+'/train_data_class.pkl'))
    # if model_clean:
    #     delete_files.extend(glob.glob(iteration_path+'/*.pt'))
    
    # print(iteration_path)
    # print(delete_files)
    for file_path in delete_files:
        if os.path.exists(file_path):  # Check if the file exists before deleting
            os.remove(file_path)
        #     print(f"Deleted: {file_path}")
        # else:
        #     print(f"File not found: {file_path}")


class VirtHitsCutoff():
    def __init__(self, cutoff_percent=1, cutoff=None):
        if cutoff:
            self.cur_cutoff = cutoff
        else:
            self.cur_cutoff = 0
        self.prev_cutoff = 0
        self.cutoff_per = cutoff_percent
    def getCutoff(self, rnd_virt_hits_ds, fixed=False):
        # print('virt cutoff rnd_virt_hits_ds ',rnd_virt_hits_ds)
        if fixed:
            return self.cur_cutoff
        else:
            tmp_cutoff =  np.quantile(rnd_virt_hits_ds,self.cutoff_per/100)
            if tmp_cutoff < self.cur_cutoff:
                self.cur_cutoff = np.float64(tmp_cutoff)
            return np.float64(self.cur_cutoff)

def load_smiles_dockscore(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def filter_data_by_dockscore(data, smiles_2_dockscore_gt):
    filtered_data = []
    for sm, mid, feat in zip(data.smiles, data.mol_ids, data.features):
        if smiles_2_dockscore_gt.get(sm, 1) < 0:
            filtered_data.append((sm, mid, feat))
    # Unzip the filtered data and convert each to a list
    return [list(x) for x in zip(*filtered_data)]


def save_docking_scores(dock_scores, file_path, mol_ids, features, smiles):
    with open(file_path, 'wb') as f:
        pickle.dump([mol_ids, dock_scores, features, smiles], f)

def wait_for_slurm_completion(sbatch_output):
    # Extract the job ID from sbatch output
    match = re.search(r"Submitted batch job (\d+)", sbatch_output.stdout)
    if not match:
        raise RuntimeError("Failed to retrieve Slurm job ID.")

    job_id = match.group(1)
    print(f"Submitted Slurm job {job_id}. Waiting for completion...")

    # Wait for the job to finish
    while True:
        squeue_output = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
        if job_id not in squeue_output.stdout:
            break  # Job has finished
        time.sleep(10)  # Check every 10 seconds

def write_inf_helper_script(iteration_dir, config):
    """Creates the bash script for parallelizing inference over num_gpus by running inference_helper 
    """
    script_content = f"""#!/bin/bash
#SBATCH --array=0-{config.model_hps.num_gpus-1}   # Array for snum_gpus tasks
#SBATCH --job-name=||inf
#SBATCH --partition=gpu-bigmem
#SBATCH --exclude=gn17
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20000M
#SBATCH -o {iteration_dir}/inference_%A_%a.out
source ~/miniconda3/etc/profile.d/conda.sh
conda activate {config.global_params.env_name}
python ~/phd/ddlight/ALMolformer_inference_helper.py $SLURM_ARRAY_TASK_ID {iteration_dir} {config.global_params.project_path}/{config.global_params.project_name}/config.pkl
    """
    #{config.global_params.code_path}/config/params.yml
    script_file = os.path.join(iteration_dir,f'parallel_inf.sh')  # sdf_dir.rsplit('/', 2)[0] + '/'    #os.path.join(sdf_dir, f'{name}_conf.sh') #
    print('inf.py script_file  ', script_file)
    with open(script_file, 'w') as f:
        f.write(script_content)
    return script_file

def fetch_random_batch(batch_size, molecule_df, used_zinc_ids, fingerprint = False, init_acq=False, tokenizer=None):
    """Fetch a random batch directly from the DataFrame."""
    fetched_batch, zinc_list, smiles_list = [],[],[]
    rnd_state = 420 if init_acq else None 
    # Filter the DataFrame to exclude the used zinc_ids
    available_df = molecule_df[~molecule_df['zinc_id'].isin(used_zinc_ids)]
    while len(fetched_batch) < batch_size and not available_df.empty:
        # Randomly sample batches from the available DataFrame
        sampled_batch = available_df.sample(n=min(batch_size - len(fetched_batch), len(available_df)), random_state= rnd_state)
        # print(f'alhelpers.py availabledf {len(available_df)}, sampled_batch {len(sampled_batch)}')
        for _, row in sampled_batch.iterrows():
            zinc_id, smiles = row['zinc_id'], row['smiles']
            if zinc_id not in used_zinc_ids:
                if init_acq:
                    used_zinc_ids.add(zinc_id)
                if fingerprint:
                    indices = np.frombuffer(row['indices'], dtype=np.int32)
                    dense_fp = np.zeros((1, 2048))
                    dense_fp[0, indices] = 1
                    fetched_batch.append((zinc_id, smiles, dense_fp))
                else:
                    zinc_list.append(zinc_id)
                    smiles_list.append(smiles)
                    fetched_batch = zinc_list # dummy fetched batch to get out of while loop when enough samples are accumulated
    if not fingerprint:
        dense_fp = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
        fetched_batch = (zinc_list, smiles_list, dense_fp)

        # # Update available_df to exclude the sampled batch
        # available_df = available_df[~available_df['zinc_id'].isin(fetched_batch)]

    return fetched_batch, used_zinc_ids

def prepare_datasets(it0_data, model, config):
    def prepare_mlp_datasets(it0_data, ros):
        X_train_resampled, y_train_resampled = ros.fit_resample(it0_data.train.features, it0_data.train.dock_labels)
        print('Resampled train_labels distribution:', Counter(y_train_resampled))

        train_dataset = TensorDataset(torch.tensor(X_train_resampled, dtype=torch.float32),
                                    torch.tensor(y_train_resampled, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(it0_data.validation.features, dtype=torch.float32),
                                    torch.tensor(it0_data.validation.dock_labels, dtype=torch.long))
        it0_test_dataset = TensorDataset(torch.tensor(it0_data.test.features, dtype=torch.float32),
                                        torch.tensor(it0_data.test.dock_labels, dtype=torch.long))
        return train_dataset, val_dataset, it0_test_dataset
    
    def prepare_molformer_datasets(it0_data, model, ros, config):
        train_smiles_resampled, train_labels_resampled = ros.fit_resample(
            np.array(it0_data.train.smiles).reshape(-1, 1), it0_data.train.dock_labels)
        train_smiles_resampled = train_smiles_resampled.flatten().tolist()
        print('Resampled train_labels distribution:', Counter(train_labels_resampled))
        # model = MoLFormer(dropout_rate=0.1)
        initialX, valX, testX = model.tokenizer(train_smiles_resampled, padding=True, truncation=True, return_tensors="pt"), \
                        model.tokenizer(it0_data.validation.smiles, padding=True, truncation=True, return_tensors="pt"), \
                        model.tokenizer(it0_data.test.smiles, padding = True, truncation = True, return_tensors ="pt")
        train_dataset = MolFormerDataset(initialX, torch.tensor(train_labels_resampled, dtype=torch.long))
        val_dataset = MolFormerDataset(valX, torch.tensor(it0_data.validation.dock_labels, dtype=torch.long))
        test_dataset = MolFormerDataset(testX, torch.tensor(it0_data.test.dock_labels, dtype=torch.long))
        return train_dataset, val_dataset, test_dataset

    ros = RandomOverSampler(random_state=42)
    if isinstance(model, DockingModel3K) or isinstance(model , DockingModelTx):
        return prepare_mlp_datasets(it0_data, ros)
    elif isinstance(model, AdvancedMoLFormer) or isinstance(model,EnhancedMoLFormer): # in  (MoLFormer, EnhancedMoLFormer, AdvancedMoLFormer): #('molformer', 'enhanced_molformer', 'advanced_molformer'):
        return prepare_molformer_datasets(it0_data, model, ros, config)

def initialize_model(model_architecture):
    """
    Initialize a docking model based on the specified architecture.

    Args:
        model_architecture (str): The name of the model architecture to initialize.

    Returns:
        nn.Module: An instance of the initialized model.
    """
    if model_architecture == 'mlpTx':
        return DockingModelTx(input_size=2048, p=0.1)
    elif model_architecture == 'mlp3K':
        docking_model =  DockingModel3K(input_size=2048, p=0.1)
        docking_model.tokenizer = None
        return docking_model
    elif model_architecture == 'molformer':
        return MoLFormer(dropout_rate=0.1)
    elif model_architecture == 'enhanced_molformer':
        return EnhancedMoLFormer(dropout_rate=0.1)
    elif model_architecture == 'advanced_molformer':
        return AdvancedMoLFormer(dropout_rate=0.5)
    else:
        raise ValueError(f"Unsupported model architecture: {model_architecture}")

def run_first_iteration(config, total_size, molecule_df, used_zinc_ids, smiles_2_dockscore_gt, cutoff, tokenizer=None ):
    def extract_batch_data(batch, fingerprint):
        if fingerprint:
            mol_id_list = [item[0] for item in batch]
            smiles_list = [item[1] for item in batch]
            features = np.vstack([item[2] for item in batch])
            return mol_id_list, smiles_list, features
        else:
            mol_id_list, smiles_list = batch[0], batch[1]
            features = batch[2]
            return mol_id_list, smiles_list, features
    """Run the first iteration and get train, validation, and test sets."""
    # Fetch and split batches
    if config.global_params.model_architecture in ('mlp3K', 'mlpTx'):
        fingerprint = True 
    elif config.global_params.model_architecture in ('advanced_molformer'):
        fingerprint = False
    train_batch,used_zinc_ids = fetch_random_batch(total_size, molecule_df, used_zinc_ids, init_acq=True, fingerprint=fingerprint, tokenizer = tokenizer)
    val_batch,used_zinc_ids = fetch_random_batch(50000, molecule_df, used_zinc_ids, init_acq= True, fingerprint=fingerprint, tokenizer=tokenizer)
    test_batch,used_zinc_ids = fetch_random_batch(50000, molecule_df, used_zinc_ids, init_acq=True, fingerprint=fingerprint, tokenizer=tokenizer)
    
    print("First Iteration:")
    print(f"Train Batch Size: {len(train_batch[0])}")
    print(f"Validation Batch Size: {len(val_batch[0])}")
    print(f"Test Batch Size: {len(test_batch[0])}")

    # Extract data for train, validation, and test batches
    train_data = extract_batch_data(train_batch, fingerprint=fingerprint)
    val_data = extract_batch_data(val_batch, fingerprint=fingerprint)
    test_data = extract_batch_data(test_batch, fingerprint=fingerprint)

    if smiles_2_dockscore_gt:
        train_dock_scores = [smiles_2_dockscore_gt[smiles] for smiles in train_data[1]]
        val_dock_scores = [smiles_2_dockscore_gt[smiles] for smiles in val_data[1]]
        test_dock_scores = [smiles_2_dockscore_gt[smiles] for smiles in test_data[1]]
        
    else:
        if config.global_params.dock_pgm == 'vina':
            train_dock_scores, _ = get_vina_scores_mul_gpu(train_data[1], molecule_df, config, num_gpus=config.model_hps.num_gpus, 
                                                                output_dir=f"{config.global_params.project_path}/{config.global_params.project_name}/iteration_0/vina_results",
                                                                dockscore_gt=smiles_2_dockscore_gt)
            val_dock_scores, _ = get_vina_scores_mul_gpu(val_data[1], molecule_df, config, num_gpus=config.model_hps.num_gpus, 
                                                                output_dir=f"{config.global_params.project_path}/{config.global_params.project_name}/iteration_0/vina_results",
                                                                dockscore_gt=smiles_2_dockscore_gt)
            test_dock_scores, _ = get_vina_scores_mul_gpu(test_data[1], molecule_df, config, num_gpus=config.model_hps.num_gpus, 
                                                                output_dir=f"{config.global_params.project_path}/{config.global_params.project_name}/iteration_0/vina_results",
                                                                dockscore_gt=smiles_2_dockscore_gt)
        elif config.global_params.dock_pgm == 'glide':
            batches = EasyDict({'train':EasyDict({'smiles':train_data[1],
                                                  'libID':train_data[0]}),
                                'validation':EasyDict({'smiles':val_data[1],
                                                  'libID':val_data[0]}),
                                'test':EasyDict({'smiles':test_data[1],
                                                  'libID':test_data[0]})})
            train_dock_scores, val_dock_scores, test_dock_scores = get_glide_scores_mul_gpu(batches, 0, config)
                                                   #num_gpus=config.model_hps.num_gpus) 
                                                                # output_dir=f"{config.global_params.project_path}/{config.global_params.project_name}/iteration_0/vina_results",
                                                                # dockscore_gt=smiles_2_dockscore_gt)
            print('train glide dock scores returned ', train_dock_scores)
            
            # train_dock_scores = get_glide_scores_mul_gpu(train_data[1], molecule_df, config, num_gpus=config.model_hps.num_gpus, 
            #                                                     output_dir=f"{config.global_params.project_path}/{config.global_params.project_name}/iteration_0/vina_results",
            #                                                     dockscore_gt=smiles_2_dockscore_gt)
            # val_dock_scores = get_glide_scores_mul_gpu(val_data[1], molecule_df, config, num_gpus=config.model_hps.num_gpus, 
            #                                                     output_dir=f"{config.global_params.project_path}/{config.global_params.project_name}/iteration_0/vina_results",
            #                                                     dockscore_gt=smiles_2_dockscore_gt)
            # test_dock_scores = get_glide_scores_mul_gpu(test_data[1], molecule_df, config, num_gpus=config.model_hps.num_gpus, 
            #                                                     output_dir=f"{config.global_params.project_path}/{config.global_params.project_name}/iteration_0/vina_results",
            #                                                     dockscore_gt=smiles_2_dockscore_gt)

    if cutoff is None:
        cutoff_fn = VirtHitsCutoff(cutoff_percent=1)
        cutoff = cutoff_fn.getCutoff(train_dock_scores, fixed=False)
        print('it0 cutoff ', cutoff)
    else:
        cutoff = np.float64(cutoff)
        cutoff_fn = VirtHitsCutoff(cutoff_percent=1, cutoff=cutoff)
        cutoff = cutoff_fn.getCutoff(train_dock_scores, fixed=True)
        print('it0 cutoff ', cutoff)
    
    train_labels = [1 if score <cutoff else 0 for score in train_dock_scores]
    val_labels = [1 if score <cutoff else 0 for score in val_dock_scores]
    test_labels = [1 if score <cutoff else 0 for score in test_dock_scores]

    data_dict = {
        "train": {
            "mol_ids": train_data[0],
            "smiles": train_data[1],
            "features":  train_data[2],
            "dock_scores": train_dock_scores,
            "dock_labels": train_labels,
            "cutoff":cutoff
        },
        "validation": {
            "mol_ids": val_data[0],
            "smiles": val_data[1],
            "features":  val_data[2],
            "dock_scores": val_dock_scores,
            "dock_labels": val_labels
        },
        "test": {
            "mol_ids": test_data[0],
            "smiles": test_data[1],
            "features":  test_data[2],
            "dock_scores": test_dock_scores,
            "dock_labels": test_labels
        }
    }
    os.makedirs("dataset", exist_ok=True)
    with open('dataset/it0data.pkl','wb') as f:
        pickle.dump(data_dict,f)
    return EasyDict(data_dict), used_zinc_ids


def fetch_unlabeled_batch(limit, offset, molecule_df, used_zinc_ids, fingerprint=True, tokenizer= None):
    """Fetch a batch of unlabeled molecules from the DataFrame."""
    fetched_batch = []

    # Filter the DataFrame to exclude the used zinc_ids
    available_df = molecule_df[~molecule_df['zinc_id'].isin(used_zinc_ids)]

    # Apply limit and offset for the batch
    batch_df = available_df.iloc[offset:offset+limit]
    smiles_list, zinc_list =[],[]
    # data_finish_flag = offset+limit == len(available_df)
    print('offset+limit ', offset+limit, 'available df ',len(available_df))
    for _, row in batch_df.iterrows():
        zinc_id = row['zinc_id']
        smiles = row['smiles']
        
        if fingerprint:
            indices_blob = row['indices']
            if zinc_id not in used_zinc_ids:
                # Convert binary blobs back to their original arrays
                indices = np.frombuffer(indices_blob, dtype=np.int32)
                dense_fp = np.zeros((1, 2048))
                dense_fp[0, indices] = 1
                fetched_batch.append((zinc_id, smiles, dense_fp))
        else:
            if zinc_id not in used_zinc_ids:
                smiles_list.append(smiles) 
                zinc_list.append(zinc_id)
    
    if not fingerprint:
        dense_fp = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
        fetched_batch = (zinc_list, smiles_list, dense_fp)
        # print('fetched batch', fetched_batch)
        # print(len(zinc_list), len(smiles_list))
        # print(dense_fp)
        # # assert len(zinc_list)==len(smiles_list)==len(dense_fp)
        # fetched_batch.append((zinc_id, smiles, fp) for zinc_id, smiles, fp in zip(zinc_list, smiles_list, dense_fp))
    data_finish_flag = (offset>0) and (len(fetched_batch[0])%offset!=0) 
    print(f'Returning {len(fetched_batch[0])} unlabeled examples from fetch_unlabeled_batch() offset ', offset, (len(fetched_batch[0])%offset!=0) )
    return fetched_batch, len(available_df), data_finish_flag


def run_inference(model, molecule_df, batch_size, top_k, used_zinc_ids, config, validation_dock_labels, pred_val_proba, tokenizer=None):
    '''returns top-k (based on acquisition func.) acquired molecules 
        also returns random #topk number of virtual hits (to be used as val set of next iteration)
        validation_dock_labels and pred_val_proba are used to compute precision-recall curve for drop_db
        validation_dock_labels : true labels for validation set
    '''
    offset = 0
    top_molecules, virtual_hits = [], []
    excluded_ids = set()  # To track IDs already in top_molecules and virtual_hits
    uncertainty_scores = []  # Store scores for threshold computation
    low_uncertainty_threshold = 1000
    while True:
        # Fetch an unlabeled batch from the DataFrame
        # unlabeled_batch = fetch_unlabeled_batch(batch_size, offset, molecule_df, used_zinc_ids)
        # Model prediction and scoring
        if config.global_params.model_architecture in ('mlpTx','mlp3K'):
            unlabeled_batch, len_av_df = fetch_unlabeled_batch(batch_size, offset, molecule_df, used_zinc_ids, fingerprint=True, tokenizer=tokenizer)
            mol_ids, smiles_list, features = zip(*unlabeled_batch)
            features = np.vstack(features)
            dataset = TensorDataset(
                torch.tensor(features, dtype=torch.float32),
                torch.tensor([1] * len(features), dtype=torch.long)  # dummy labels
            )
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=1024 * 2,
                shuffle=False,
                num_workers=1
            )
            pred_labels, pred_proba, scores = predict_with_model(model, dataloader, 
                                                             acquisition_function=config.al_params.acquisition_function)  # scores is uncertainty (or acquisition score)
        elif config.global_params.model_architecture in ('molformer', 'enhanced_molformer', 'advanced_molformer'):
            unlabeled_batch, len_av_df, has_data_finshed = fetch_unlabeled_batch(batch_size, offset, molecule_df, used_zinc_ids, fingerprint=False, tokenizer=tokenizer)
            mol_ids, smiles_list, molformer_features = unlabeled_batch
            # molformer_features = (
            #     model.module.tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
            #     if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            #     else model.tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
            # )
            dataset = MolFormerDataset(molformer_features, 
                                      torch.tensor([1] * len(molformer_features['input_ids']), dtype=torch.long)) #dummy labels 
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=1024 // 2,
                shuffle=False,
                num_workers=0
            )
            pred_labels, pred_proba, scores = predict_with_molformer(model, dataloader,
                                                             acquisition_function=config.al_params.acquisition_function) # scores is uncertainty (or acquisition score)
            
        uncertainty_scores.extend(scores)  # Collect uncertainty scores
        
        # Iterate over the batch to filter top_k and virtual hits. 
        if config.al_params.drop_db and config.al_params.drop_db_recall:
            precision, recall, thresholds = precision_recall_curve(validation_dock_labels, pred_val_proba)

        for mol_id, smiles, label, pred_proba, score in zip(mol_ids, smiles_list, pred_labels, pred_proba, scores):
            # Handle virtual hits (label == 1)
            if label == 1:
                virtual_hits.append((mol_id, smiles, pred_proba))
            else:
                if config.al_params.drop_db and (config.al_params.acquisition_function =='greedy' or config.al_params.acquisition_function =='random'): # drop predicted inactives by considering them used, so that they dont participate in next al iteration.
                    used_zinc_ids.add(mol_id)

            if config.al_params.drop_db and config.al_params.acquisition_function in ('unc','ucb','bald','margin','entropy','least_confidence'):
                low_uncertainty_threshold = min(low_uncertainty_threshold,np.quantile(uncertainty_scores, 0.10))# 0.25)
                if score< low_uncertainty_threshold: # drop most certain prediction mols by considering them used, so that they dont participate in next al iteration.
                    used_zinc_ids.add(mol_id) 
                    
            # Apply 90% recall-based filtering
            if config.al_params.drop_db and config.al_params.drop_db_recall:
                indices = (recall >= 0.90) # recall_threshold = 0.90
                if not any(indices):
                    # print("Warning: 90% recall not achieved. Falling back to max recall threshold.")
                    prob_thresh = thresholds[np.argmax(recall)]
                else:
                    prob_thresh = thresholds[indices][0]  # first threshold where recall ≥ 90%
                if  pred_proba > prob_thresh:
                    used_zinc_ids.add(mol_id)

            # Handle Top-k selection with a min-heap
            if len(top_molecules) < top_k:
                heapq.heappush(top_molecules, (score, mol_id, smiles))
            else:
                heapq.heappushpop(top_molecules, (score, mol_id, smiles))
        
        offset += batch_size
        print(f"alhelpers.py Processed {offset} molecules")
        
        # Keep virtual_hits manageable in memory by sampling randomly as it grows
        if len(virtual_hits) > 10 * batch_size: 
            virtual_hits = random.sample(virtual_hits, min(len(virtual_hits), 10 * batch_size))

        # print("there's a break in run_inference() of mainvina.py for testing" )
        # break
        if has_data_finshed: #len(unlabeled_batch)<batch_size: # inference over entire dataset finished.
            break
        
    # Final selection of |#topK| random virtual hits not in top molecules
    top_mol_ids = {(mol_id, smiles) for _, mol_id, smiles in top_molecules}
    virtual_hits_filtered = [(mol_id, smiles, pred_proba) for mol_id, smiles, pred_proba in virtual_hits if (mol_id, smiles) not in top_mol_ids]
    if len(virtual_hits_filtered) < top_k:
        print('NOT ENOUGH Virtual HITS found in Unlabeled Pool - virtual_hits_filtered', len(virtual_hits_filtered))
    selected_virtual_hits = random.sample(virtual_hits_filtered, min(top_k, len(virtual_hits_filtered)))

    # # Fetch additional random molecules for testing. TODO: following can be simplified. Just random_batch = fetch_random_batch(top_k) should suffice
    # random_samples = []
    # while len(random_samples) < top_k:
    #     random_batch = fetch_random_batch(top_k)
    #     for mol_id, smiles, _ in random_batch:
    #         if mol_id not in excluded_ids:
    #             random_samples.append((mol_id,smiles))
    #             excluded_ids.add(mol_id)
    #         if len(random_samples) >= top_k:
    #             break

    # Fetch additional random molecules with low uncertainty
    # Compute low-uncertainty threshold using the collected scores TODO: what's the logic of this for greedy (other non uncertainity based acq)
    low_uncertainty_threshold = np.quantile(uncertainty_scores, 0.10)# 0.25)
    random_samples = []
    while len(random_samples) < top_k//10:
        # print('mainvina.py len(rand_samples) ', len(random_samples), f'Fetching rand low uncertain test samples from DB; need {top_k//10 - len(random_samples)} more samples')
        if config.global_params.model_architecture in ('mlp3K', 'mlpTx'):
            rand_batch, used_zinc_ids = fetch_random_batch(top_k, molecule_df, used_zinc_ids, fingerprint=True, tokenizer=tokenizer)  # Fetch a batch of random molecules
            rand_test_smiles = [item[1] for item in rand_batch]
            rand_test_features = [item[2] for item in rand_batch] #molids_2_fps(cursor, [item[0] for item in rand_batch])  # Convert to features
            dataset = TensorDataset(
                torch.tensor(np.vstack(rand_test_features), dtype=torch.float32),
                torch.tensor([1] * len(rand_test_features), dtype=torch.long)  # dummy labels
            )
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=1024 * 2,
                shuffle=False,
                num_workers=1
            )
            pred_rand_test_docking_labels, pred_rand_test_proba, rand_scores = predict_with_model(
                model,
                dataloader,
                acquisition_function=config.al_params.acquisition_function
            ) 
            
        elif config.global_params.model_architecture in ('molformer', 'enhanced_molformer', 'advanced_molformer'):
            rand_batch, used_zinc_ids = fetch_random_batch(top_k, molecule_df, used_zinc_ids, fingerprint=False, tokenizer=tokenizer)  # Fetch a batch of random molecules
            rand_test_smiles = rand_batch[1] # [item[1] for item in rand_batch]
            molformer_features = rand_batch[2] #[item[2] for item in rand_batch]
            # molformer_features = (
            #     model.module.tokenizer(rand_test_smiles, padding=True, truncation=True, return_tensors="pt")
            #     if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            #     else model.tokenizer(rand_test_smiles, padding=True, truncation=True, return_tensors="pt")
            # )
            dataset = MolFormerDataset(molformer_features, 
                                      torch.tensor([1] * len(molformer_features['input_ids']), dtype=torch.long))  # dummy labels)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=1024 // 2,
                shuffle=False,
                num_workers=0
            )
            pred_rand_test_docking_labels, pred_rand_test_proba, rand_scores = predict_with_molformer(model, dataloader, 
                                                             acquisition_function=config.al_params.acquisition_function) # scores is uncertainty (or acquisition score)
        
         # Get scores for the random batch
        # predict_with_model(
        #     model,
        #     np.vstack(rand_test_features),
        #     acquisition_function=config.al_params.acquisition_function
        # )  # Get scores for the random batch

        for mol_id, smiles, _, score, pred_label, pred_proba in zip(*rand_batch, rand_scores, pred_rand_test_docking_labels, pred_rand_test_proba):
            print('ALhelper.py rand smiles, score, pred_label', smiles, score, pred_label)
            if config.al_params.acquisition_function =='greedy':
                # print('random scores ', score)
                if score> 0.5: #i.e. class =1
                    random_samples.append((mol_id, smiles, score, pred_label, pred_proba))
                    excluded_ids.add(mol_id)
            elif config.al_params.acquisition_function =='random':
                # print('random scores ', score)
                # if score> 0.5: #i.e. class =1
                random_samples.append((mol_id, smiles, score, pred_label, pred_proba))
                excluded_ids.add(mol_id)
            else: # config.al_params.acquisition_function in ['ucb','unc', 'bald']:
                if score <= low_uncertainty_threshold and mol_id not in excluded_ids:
                    random_samples.append((mol_id, smiles, score, pred_label, pred_proba))
                    excluded_ids.add(mol_id)

            # print('len rand samples ', len(random_samples))
            if len(random_samples) >= top_k//4:
                break

    # Return top-k molecule IDs and scores along with random virtual hits
    return {
        "top_acq_mols": [(mol_id, smiles, score) for score, mol_id, smiles in sorted(top_molecules, reverse=True)],
        "rand_virtual_hits": selected_virtual_hits,
        "rand_samples": random_samples,
        "len_av_df": len_av_df
    }, used_zinc_ids


def fetch_unlabeled_batch_mul_gpus(split_molecule_df, used_zinc_ids, fingerprint=True, tokenizer= None):
    """Fetch a batch of unlabeled molecules from the DataFrame."""
    fetched_batch = []

    # Filter the DataFrame to exclude the used zinc_ids
    available_df = split_molecule_df[~split_molecule_df['zinc_id'].isin(used_zinc_ids)]

    # Apply limit and offset for the batch
    batch_df = available_df
    smiles_list, zinc_list =[],[]
    # data_finish_flag = offset+limit == len(available_df)
    for _, row in batch_df.iterrows():
        zinc_id = row['zinc_id']
        smiles = row['smiles']
        
        if fingerprint:
            indices_blob = row['indices']
            if zinc_id not in used_zinc_ids:
                # Convert binary blobs back to their original arrays
                indices = np.frombuffer(indices_blob, dtype=np.int32)
                dense_fp = np.zeros((1, 2048))
                dense_fp[0, indices] = 1
                fetched_batch.append((zinc_id, smiles, dense_fp))
        else:
            if zinc_id not in used_zinc_ids:
                smiles_list.append(smiles) 
                zinc_list.append(zinc_id)
    
    if not fingerprint:
        dense_fp = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
        fetched_batch = (zinc_list, smiles_list, dense_fp)
        # print('fetched batch', fetched_batch)
        # print(len(zinc_list), len(smiles_list))
        # print(dense_fp)
        # # assert len(zinc_list)==len(smiles_list)==len(dense_fp)
        # fetched_batch.append((zinc_id, smiles, fp) for zinc_id, smiles, fp in zip(zinc_list, smiles_list, dense_fp))

    print(f'Returning {len(fetched_batch[0])} unlabeled examples from fetch_unlabeled_batch()')
    return fetched_batch, len(available_df)

def run_inference_scatter(model, iteration_dir, split_molecule_df, used_zinc_ids, config, tokenizer=None, task_id=0, num_tasks=1):
    top_molecules, virtual_hits, all_molecules = [], [], []
    uncertainty_scores = []  # Store scores for threshold computation
    low_uncertainty_threshold = 1000
    if config.global_params.model_architecture in ('mlpTx','mlp3K'):
        # unlabeled_batch, len_av_df = fetch_unlabeled_batch(batch_size, offset, molecule_df, used_zinc_ids, fingerprint=True, tokenizer=tokenizer)
        unlabeled_batch, len_av_df = fetch_unlabeled_batch_mul_gpus(split_molecule_df, used_zinc_ids, fingerprint=True, tokenizer=tokenizer)
        mol_ids, smiles_list, features = zip(*unlabeled_batch)
        features = np.vstack(features)
        dataset = TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor([1] * len(features), dtype=torch.long)  # dummy labels
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1024 * 2,
            shuffle=False,
            num_workers=1
        )
        pred_labels, pred_proba, scores = predict_with_model(model, dataloader, 
                                                            acquisition_function=config.al_params.acquisition_function)  # scores is uncertainty (or acquisition score)
    elif config.global_params.model_architecture in ('molformer', 'enhanced_molformer', 'advanced_molformer'):
        unlabeled_batch, len_av_df = fetch_unlabeled_batch_mul_gpus(split_molecule_df, used_zinc_ids, fingerprint=False, tokenizer=tokenizer)
        mol_ids, smiles_list, molformer_features = unlabeled_batch
        dataset = MolFormerDataset(molformer_features, 
                                    torch.tensor([1] * len(molformer_features['input_ids']), dtype=torch.long)) #dummy labels 
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1024 // 2,
            shuffle=False,
            num_workers=0
        )
        pred_labels, pred_proba, scores = predict_with_molformer(model, dataloader,
                                                            acquisition_function=config.al_params.acquisition_function) # scores is uncertainty (or acquisition score)
        
    uncertainty_scores.extend(scores)  # Collect uncertainty scores
    
    print(f'{sum(pred_labels)}/{len(pred_labels)} predicted as actives')
    # Iterate over the batch to filter top_k and virtual hits. 
    # Convert lists to NumPy arrays for vectorized operations
    mol_ids = np.array(mol_ids)
    smiles_list = np.array(smiles_list)
    pred_labels = np.array(pred_labels)
    pred_proba = np.array(pred_proba)
    scores = np.array(scores)

    # Identify virtual hits (label == 1)
    is_virtual_hit = pred_labels == 1
    virtual_hits.extend(zip(mol_ids[is_virtual_hit], smiles_list[is_virtual_hit], pred_proba[is_virtual_hit]))

    # Precompute uncertainty threshold if necessary
    if config.al_params.drop_db and config.al_params.acquisition_function in ('unc', 'ucb', 'bald', 'margin', 'entropy', 'least_confidence'):
        low_uncertainty_threshold = min(low_uncertainty_threshold, np.quantile(uncertainty_scores, 0.10))

    # Identify molecules to drop
    if config.al_params.drop_db:
        if config.al_params.acquisition_function in ('greedy', 'random'):
            used_zinc_ids.update(mol_ids[~is_virtual_hit])  # Directly add all predicted inactives
        elif config.al_params.acquisition_function in ('unc', 'ucb', 'bald', 'margin', 'entropy', 'least_confidence'):
            most_certain_inactives = scores < low_uncertainty_threshold
            used_zinc_ids.update(mol_ids[~is_virtual_hit & most_certain_inactives])

    # Collect all molecules in a single operation
    all_molecules.extend(zip(scores, mol_ids, smiles_list, pred_labels))

    with open(f'{iteration_dir}/allmols_virthits_{task_id}.pkl','wb') as f:
        pickle.dump((all_molecules, virtual_hits),f)
    with open(f'{iteration_dir}/used_zinc_ids_{task_id}.pkl','wb') as f:
        pickle.dump(used_zinc_ids,f)
    

def results_gather(iteration_dir, top_k, full_mol_df):
    top_molecules_heap = []  # Min-heap for global top molecules
    virtual_hits_list = []  # Collect all virtual hits
    used_zinc_ids = []
    res_files = glob.glob(str(iteration_dir+'/allmols_virthits_*.pkl'))
    taskid_2_res_dict = {}
    # print(res_files)
    for file in res_files:
        with open(file,'rb') as f:
            (all_molecules, virtual_hits) = pickle.load(f)
        filename = Path(file).stem  # Extract filename without extension
        match = re.search(r'allmols_virthits_(\d+)', filename)  # Extract the number
        if match:
            task_id = int(match.group(1))  # Convert to integer
            taskid_2_res_dict[task_id] = {'all_molecules': all_molecules,
                                          'virtual_hits':virtual_hits} 
        # print('taskid_2_res_dict ', taskid_2_res_dict)
            
    #Top-K molecules across entire library
    for task_id, results in taskid_2_res_dict.items():
        for score, mol_id, smiles, label in results['all_molecules']:
            if len(top_molecules_heap) < top_k:
                heapq.heappush(top_molecules_heap, (score, mol_id, smiles, label))
            else:
                heapq.heappushpop(top_molecules_heap, (score, mol_id, smiles, label))

        # Collect all virtual hits
        virtual_hits_list.extend(results['virtual_hits'])

    # Sort top molecules in descending order of score
    top_molecules_heap.sort(reverse=True, key=lambda x: x[0])


    #global used_zinc_ids
    for file in glob.glob(str(iteration_dir + '/used_zinc_ids_*.pkl')):
        with open(file,'rb') as f:
            used_zinc_ids.extend(list(pickle.load(f)))
    
    used_zinc_ids = set(used_zinc_ids)

    # # Save final top molecules and virtual hits
    # with open(f'{iteration_dir}/final_top_molecules.pkl', 'wb') as f:
    #     pickle.dump(top_molecules_heap, f)

    # with open(f'{iteration_dir}/final_virtual_hits.pkl', 'wb') as f:
    #     pickle.dump(virtual_hits_list, f)    

    return {
        "top_acq_mols": top_molecules_heap, #[(mol_id, smiles, score) for score, mol_id, smiles in sorted(top_molecules, reverse=True)],
        "len_av_df": len(full_mol_df)-len(used_zinc_ids)
    }, used_zinc_ids


def get_val_data_from_av_df(molecule_df, used_zinc_ids, tokenizer, config, iteration, smiles_2_dockscore_gt=None):
    available_df =  molecule_df[~molecule_df['zinc_id'].isin(used_zinc_ids)]
    sampled_batch = available_df.sample(n=min(50000, len(available_df)), random_state= 42)
    zinc_list, smiles_list = sampled_batch['zinc_id'].tolist(), sampled_batch['smiles'].tolist()
    if config.global_params.model_architecture in ('mlp3K','mlpTx'):
        zinc_ids_set = set(zinc_list)  # for faster lookup
        filtered_df = molecule_df[molecule_df['zinc_id'].isin(zinc_ids_set)].copy()
        dense_fp = np.zeros((len(filtered_df), 2048), dtype=np.uint8)
        for i, indices in enumerate(filtered_df['indices']):
            dense_fp[i, indices] = 1
        # dense_fp = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
    elif config.global_params.model_architecture in ('advanced_molformer'):
        dense_fp = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
    if config.global_params.target in ['jak2', 'braf', 'parp1','fa7', '5ht1b', 'pgk1', 'pgk2']:
        docking_scores, _ = get_vina_scores_mul_gpu(smiles_list, molecule_df, config, num_gpus=config.model_hps.num_gpus, 
                                                            output_dir=f"{config.global_params.project_path}/{config.global_params.project_name}/iteration_{iteration}/vina_results",
                                                            dockscore_gt=smiles_2_dockscore_gt)
    if smiles_2_dockscore_gt is not None:
        docking_scores = [smiles_2_dockscore_gt[smiles] for smiles in smiles_list] 

    return {'mol_ids':zinc_list, #Considering apriori validation set as validation
            'features':dense_fp,
            'dock_scores':docking_scores,
            'smiles': smiles_list
        }

def run_subsequent_iterations_mul_gpu(initial_model, molecule_df, dd_cutoff, it0val_metrics, it0test_metrics, num_iterations, batch_size, top_k, config, train_data_class,used_zinc_ids, 
                              rank, world_size, smiles_2_dockscore_gt=None, tokenizer = None, val_proba = None, val_labels = None):
    
    import wandb
    wandb.login()
    
    """
    Executes an active learning loop over a large molecule dataset, iteratively acquiring 
    the top-k molecules based on model predictions and retraining the model.

    Args:
        initial_model (object): 
            The machine learning model to be trained, implementing `fit()` and `predict()` methods.
        molecule_df (pd.DataFrame): 
            A DataFrame containing molecule information, including `zinc_id`, `smiles`, and docking scores.
        num_iterations (int): 
            Number of active learning iterations to run.
        batch_size (int): 
            Number of molecules to process per batch during inference.
        top_k (int): 
            Number of top molecules to acquire and add to the labeled dataset in each iteration.
    
    Returns:
        None

    Example:
        >>> initial_model = RandomForestClassifier()
        >>> molecule_df = pd.read_csv("path_to_molecule_data.csv")
        >>> num_iterations = 10
        >>> batch_size = 1_000_000
        >>> top_k = 10_000
        >>> run_subsequent_iterations(initial_model, molecule_df, num_iterations, batch_size, top_k)
    """
    model = initial_model
    it0_data = train_data_class[0]
    labeled_data = [(mol_id, None) for mol_id in it0_data.train.mol_ids] # Stores the labeled dataset
    
    if config.threshold_params.given_cutoff:
        dd_cutoff = np.float64(config.threshold_params.given_cutoff) #it0cutoff = cutoff_fn_al.getCutoff(it0_data.train.dock_scores, fixed = True) #prev_dd_cutoff
    else:
        cutoff_fn_al = VirtHitsCutoff(cutoff_percent=1)
        dd_cutoff = it0cutoff = cutoff_fn_al.getCutoff(it0_data.train.dock_scores, fixed = False) #prev_dd_cutoff

    with wandb.init(project = 'DDS_AL_2M_AllTgt',
                    config = config, mode = None): #None
        print(wandb.run.id, wandb.run.name)
        wandb.run.name=f'{config.global_params.model_architecture}_{config.global_params.target}_{config.al_params.acquisition_function}_dropDB{config.al_params.drop_db}_fxcut{config.threshold_params.fixed_cutoff}_{wandb.run.name}'
        t0 = time.time()
        info_vals = {'time':time.time()-t0,
                    'iteration':0,
                    'available_mols':len(molecule_df)}
        tmp_dict = {'Val_Precision':it0val_metrics['precision'],
                        'Val_Recall':it0val_metrics['recall'],
                        'Val_F1':it0val_metrics['f1'],
                        'Val_AUC':it0val_metrics['auc'],
                        'Val_FP':it0val_metrics['fp'],
                        'Val_TP':it0val_metrics['tp'],
                        'Val_FN':it0val_metrics['fn'],
                        'Val_TN':it0val_metrics['tn'],
                        # 'Val_enrichment1':it0val_metrics['enrichment_1']
                        # 'Val_loss':it0val_metrics
                        }
        info_vals.update(tmp_dict)
        tmp_dict = {'AprioriTest_Precision':it0test_metrics['precision'],
                        'AprioriTest_Recall':it0test_metrics['recall'],
                        'AprioriTest_F1':it0test_metrics['f1'],
                        'AprioriTest_AUC':it0test_metrics['auc'],
                        'Apriori_FP':it0test_metrics['fp'],
                        'Apriori_TP':it0test_metrics['tp'],
                        'Apriori_FN':it0test_metrics['fn'],
                        'Apriori_TN':it0test_metrics['tn'],
                        'Apriori_enrichment1':it0test_metrics['enrichment_1']}
        info_vals.update(tmp_dict)
        keys = sorted(info_vals.keys())
        wandb.log(info_vals)

        # Make sure you use the last iteration at the end and avoid local variable error
        vina = None
        iteration = 0
        iteration_dir = os.path.join(config.global_params.project_path, 
                                     config.global_params.project_name, f'iteration_{iteration}')

        for iteration in range(1, num_iterations):
            info_vals = {'time':time.time()-t0,
                         'iteration':iteration}
            print(f"Iteration {iteration}/{num_iterations}")
            # Inference to acquire top-k molecules. For decaying acq size do top_k//iteration
            num_tasks = config.model_hps.num_gpus
            if num_tasks>1:
                iteration_dir = os.path.join(config.global_params.project_path, 
                                        config.global_params.project_name, f'iteration_{iteration}')
                os.makedirs(iteration_dir,exist_ok=True)
                clear_iter(os.path.join(config.global_params.project_path, 
                                    config.global_params.project_name, f'iteration_{iteration-1}'))
                
                if config.al_params.drop_db:
                    available_df =  molecule_df[~molecule_df['zinc_id'].isin(used_zinc_ids)]
                else:
                    available_df = molecule_df
                available_df_numpy_array = available_df.to_numpy()
                for task_id in range(0, config.model_hps.num_gpus):
                    # Split dataset for parallel processing
                    print('drop db is ',  config.al_params.drop_db )
                    
                    total_mols = len(available_df)
                    chunk_size = (total_mols + num_tasks - 1) // num_tasks  # Ensure even splitting
                    start_idx = task_id * chunk_size
                    end_idx = min((task_id + 1) * chunk_size, total_mols)
                    # split_molecule_df = available_df.iloc[start_idx:end_idx]  # Process only assigned chunk
                    # split_molecule_df = pd.DataFrame(available_df.values[start_idx:end_idx], columns=available_df.columns)
                    split_molecule_df = pd.DataFrame(available_df_numpy_array[start_idx:end_idx], columns=available_df.columns)
                    print(f"SLURM Task {task_id}: Processing molecules {start_idx} to {end_idx}")
                    # split_molecule_df.to_csv(iteration_dir+f'/split_mol_df_{task_id}.csv', index=False)
                    pickle.dump(split_molecule_df, open(iteration_dir+f'/split_mol_df_{task_id}.pkl','wb'))
                    with open(iteration_dir+'/used_zinc_ids.pkl','wb') as f:
                        pickle.dump(used_zinc_ids, f)
                parallel_inf_sh_path = write_inf_helper_script(iteration_dir, config)
                cmd = ["sbatch", parallel_inf_sh_path ]
                sbatch_output = subprocess.run(cmd, capture_output=True, text=True, check=True)
                wait_for_slurm_completion(sbatch_output)

                # TODO aggregate results of split_mol_df inference to get top_k_molecules, rand_virtual_hits, rand_test_samples, len_av_df
                inf_res, used_zinc_ids = results_gather(iteration_dir, top_k, molecule_df)
                top_k_molecules, len_av_df = inf_res['top_acq_mols'], inf_res['len_av_df']
                print('top_k_molecules after gather ', len(top_k_molecules), 'used zinc ids ', len(used_zinc_ids),'available df ', len_av_df)

            else:
                inf_res, used_zinc_ids = run_inference(model, molecule_df, batch_size, top_k//1, used_zinc_ids, config, validation_dock_labels, pred_val_proba, tokenizer=tokenizer) # acquire topk, then topk/2, then topk/3,...., to not change the training distribution too much from apriori (real-world) distribution
                top_k_molecules, rand_virtual_hits, rand_test_samples, len_av_df = inf_res['top_acq_mols'], inf_res['rand_virtual_hits'], inf_res['rand_samples'], inf_res['len_av_df']
            info_vals.update({'available_mols':len_av_df})
            print(f'{len(top_k_molecules)} top molecules acquired after iteration {iteration}')
            # print('top_k_molecules[0:3] ', top_k_molecules[0:3])

            # Retrieve molecule IDs and add them to the labeled dataset
            print(f'iteration {iteration} used_zinc_ids before adding topK acq molids {len(used_zinc_ids)}')
            scores, mol_ids, smiles_list, _ = zip(*top_k_molecules) 
            print(f'iteration{iteration} topk_mol_ids ', len(mol_ids))
            used_zinc_ids = used_zinc_ids.union(set(list(mol_ids)))
            print(f'iteration {iteration} used_zinc_ids {len(used_zinc_ids)}')

            new_data = [(mol_id, smiles, score) for mol_id, smiles, score in zip(mol_ids, smiles_list, scores)]
            labeled_data.extend(new_data)
            print(f'main.py Itearaion {iteration} total train set size {len(labeled_data)}')
            # Load features for newly acquired molecules
            if config.global_params.target in ['jak2', 'braf', 'parp1','fa7', '5ht1b', 'pgk1', 'pgk2']:
                if config.global_params.dock_pgm == 'vina':
                    vina = QuickVina2GPU(vina_path="/groups/cherkasvgrp/Vina-GPU-2.1/QuickVina2-GPU-2.1/QuickVina2-GPU-2-1", #QuickVina2-GPU-2-1"', # Avoiding global initialization because _teardown deletes tmp dirs
                                        target=config.global_params.target)
                    new_docking_scores, _ = get_vina_scores_mul_gpu([item[1] for item in new_data], molecule_df, config, num_gpus=config.model_hps.num_gpus, 
                                                                output_dir=f"{config.global_params.project_path}/{config.global_params.project_name}/iteration_{iteration}/vina_results",
                                                                dockscore_gt=smiles_2_dockscore_gt)
                elif config.global_params.dock_pgm == 'glide':
                    batches = EasyDict({'train':EasyDict({'smiles':[item[1] for item in new_data],
                                                  'libID':[item[0] for item in new_data]}),
                                        })
                    #train_dock_scores, val_dock_scores, test_dock_scores = get_glide_scores_mul_gpu(batches, 0, config)
                    new_docking_scores, _, _ = get_glide_scores_mul_gpu(batches, 0, config)
                    
            new_docking_scores = [smiles_2_dockscore_gt[item[1]] for item in new_data] if smiles_2_dockscore_gt else new_docking_scores
            if config.global_params.model_architecture in ('mlp3K','mlpTx'):
                new_features = molids_2_fps(molecule_df=molecule_df, mol_ids=mol_ids, fast=True)
            elif config.global_params.model_architecture in ('advanced_molformer'):
                new_features = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
            # print('main vina new_features[0:3]', new_features[0:3])
                assert len(new_docking_scores)== len(new_features['input_ids'])
            print(f"Acquired {len(new_data)} molecules (b4 0dock filtering) for Iteration {iteration + 1} ")


            # # Model validation of virtual hits
            # # vina = QuickVina2GPU(vina_path="/groups/cherkasvgrp/Vina-GPU-2.1/QuickVina2-GPU-2.1/QuickVina2-GPU-2-1", #QuickVina2-GPU-2-1"', # Avoiding global initialization because _teardown deletes tmp dirs
            # #                     target=config.global_params.target)
            # val_docking_scores = get_vina_scores_mul_gpu([item[1] for item in rand_virtual_hits], molecule_df, config, num_gpus=config.model_hps.num_gpus, 
            #                                             output_dir=f"{config.global_params.project_path}/{config.global_params.project_name}/iteration_{iteration}/vina_results",
            #                                             dockscore_gt=smiles_2_dockscore_gt)
            # val_docking_scores = [smiles_2_dockscore_gt[item[1]] for item in rand_virtual_hits]
            # if config.global_params.model_architecture in ('mlp3K','mlpTx'):
            #     val_features = molids_2_fps(molecule_df=molecule_df, mol_ids= [item[0] for item in rand_virtual_hits], fast=True)
            # elif config.global_params.model_architecture in ('advanced_molformer'):
            #     val_features = tokenizer([item[1] for item in rand_virtual_hits], padding=True, truncation=True, return_tensors="pt")
            #     assert len(val_docking_scores)==len(val_features['input_ids'])
            # # print('mainvina.py val_docking_scores ', val_docking_scores, len(val_docking_scores))

           
            #augment train dataset with new acquired data
            # Filter molecules with docking_score < 0
            filtered_mol_ids = [mol_id for mol_id, score in zip(mol_ids, new_docking_scores) if score < 0]
            if tokenizer:
                filtered_features = [feature for feature, score in zip(new_features['input_ids'], new_docking_scores) if score < 0]
            else:
                filtered_features = [feature for feature, score in zip(new_features, new_docking_scores) if score < 0]
            filtered_scores = [score for score in new_docking_scores if score < 0]
            filtered_smiles = [smile for smile, score in zip([item[1] for item in new_data], new_docking_scores) if score < 0]

            new_it_data = EasyDict({'train':{'mol_ids': filtered_mol_ids, # list(mol_ids),
                                    'features': np.vstack(filtered_features), #np.vstack(new_features),
                                    'dock_scores': filtered_scores, #new_docking_scores,
                                    'smiles': filtered_smiles #[item[1] for item in new_data]
                                    },

                                    
                        # 'validation':{'mol_ids':it0_data.test.mol_ids, #Considering apriori test set as validation
                        #               'features':it0_data.test.features,
                        #               'dock_scores':it0_data.test.dock_scores,
                        #               'smiles': it0_data.test.smiles
                        #             }
                            })
            
            # # Create a reduced version of the data
            # reduced_data = EasyDict({'train': {'mol_ids': new_it_data.train.mol_ids,
            #                                 'dock_scores': new_it_data.train.dock_scores},
            #                         'validation': {'mol_ids': new_it_data.validation.mol_ids,
            #                                         'dock_scores': new_it_data.validation.dock_scores}})

            # # Save the reduced data to a .pkl file
            # output_path = f"{config.global_params.project_path}/{config.global_params.project_name}/iteration_{iteration}/acquired_data_reduced.pkl"
            # with open(output_path, 'wb') as f:
            #     pickle.dump(reduced_data, f)
                

            prev_it_data = train_data_class[iteration-1]
            # print(prev_it_data.train.features.shape, np.vstack(new_it_data.train.features).shape)
            # print('updated train features ',np.concatenate((prev_it_data.train.features, new_it_data.train.features)).shape)

            if config.threshold_params.given_cutoff:
                dd_cutoff = np.float64(config.threshold_params.given_cutoff)
            else:
                dd_cutoff = cutoff_fn_al.getCutoff(np.concatenate((prev_it_data.train.dock_scores, new_it_data.train.dock_scores)), fixed = config.threshold_params.fixed_cutoff) #prev_dd_cutoff
            
            train_data_class[iteration]= EasyDict({
                'train': {'mol_ids': prev_it_data.train.mol_ids + new_it_data.train.mol_ids,
                        # 'features': np.concatenate((prev_it_data.train.features, new_it_data.train.features)),
                        'dock_scores': np.concatenate((prev_it_data.train.dock_scores, new_it_data.train.dock_scores)),
                        'cutoff': dd_cutoff,
                        'smiles': prev_it_data.train.smiles + new_it_data.train.smiles,
                        },
                'validation': {'mol_ids':it0_data.validation.mol_ids, #Considering apriori validation set as validation
                                      'features':it0_data.validation.features,
                                      'dock_scores':it0_data.validation.dock_scores,
                                      'smiles': it0_data.validation.smiles
                            }  if config.threshold_params.fixed_cutoff else get_val_data_from_av_df(molecule_df, used_zinc_ids,tokenizer, config, iteration, smiles_2_dockscore_gt=smiles_2_dockscore_gt) #new_it_data.validation
            })
            if config.global_params.model_architecture == 'mlp3K':
                train_data_class[iteration]['train']['features'] = np.concatenate(
                    (prev_it_data.train.features, new_it_data.train.features)
                )
                      
            dd_count = np.sum(train_data_class[iteration].validation.dock_scores<dd_cutoff) # np.sum(val_docking_scores< dd_cutoff)
            print(f"mainvina.py DD2Cutoff - Cutoff: {dd_cutoff}, Apriori Molecules below cutoff: {dd_count}")
            info_vals.update({'DDcutoff':dd_cutoff,
                              "Val_mols_bel_cutoff":dd_count})

            # new_docking_labels = (new_docking_scores < dd_cutoff).astype(int)
            # true_val_docking_labels = (val_docking_scores < dd_cutoff).astype(int)  #TODO fix it to make sure only non zero docking scores are considered
            # true_val_docking_labels = it0_data.validation.dock_labels
            

            # # Save the reducefull data to a .pkl file
            output_path = f"{config.global_params.project_path}/{config.global_params.project_name}/iteration_{iteration}/train_data_class.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump(train_data_class, f)

            train_dock_labels = (train_data_class[iteration].train.dock_scores < train_data_class[iteration].train.cutoff).astype(int)
            validation_dock_labels = (train_data_class[iteration].validation.dock_scores < train_data_class[iteration].train.cutoff).astype(int)
            train_data_class[iteration].validation.dock_labels = validation_dock_labels
            print('main.py train_dock_labels distribution ', Counter(train_dock_labels) , 'cutoff ',train_data_class[iteration].train.cutoff)
            print('main.py val_dock_labels distribution ', Counter(validation_dock_labels))
        
            if config.al_params.model_retrain:
                # print('main.py new_docking_labels oversampled distribution ', Counter(y_train_resampled))
                model.fit(X_train_resampled, y_train_resampled)  
            else:
                if config.global_params.model_architecture in ['mlpTx', 'mlp3K']:
                    if config.global_params.model_architecture == 'mlpTx':
                        model = DockingModelTx(input_size=2048) 
                    else:
                        model = DockingModel3K(input_size=2048)
                    ros = RandomOverSampler(random_state=42)
                    X_train_resampled, y_train_resampled = ros.fit_resample(train_data_class[iteration].train.features, train_dock_labels)
                    # Convert training and validation data into DataLoaders
                    train_dataset = TensorDataset(
                        torch.tensor(X_train_resampled, dtype=torch.float32),
                        torch.tensor(y_train_resampled, dtype=torch.long)
                    )
                    val_dataset = TensorDataset(
                        torch.tensor(train_data_class[iteration].validation.features, dtype=torch.float32),
                        torch.tensor(validation_dock_labels, dtype=torch.long)
                    )
                    train_loader = DataLoader(
                        dataset=train_dataset,
                        batch_size=1024 * 2,
                        shuffle=True,
                        num_workers=1
                    )
                    val_loader = DataLoader(
                        dataset=val_dataset,
                        batch_size=1024 * 2,
                        shuffle=False,
                        num_workers=1
                    )
                    try:
                        model, total_val_loss, _, val_proba, val_labels = train_docking_model(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            al_iteration=iteration,
                            num_epochs=40,
                            lr=0.001,
                            weight_decay=0.01,
                            model_save_path='/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/',
                            config=config,
                            rank=rank
                        )
                        pred_val_docking_labels, pred_val_proba, _ = predict_with_model(model, val_loader,acquisition_function='greedy') #greedy to not use dropout for inference to get correct metric computation #config.al_params.acquisition_function) #predict_with_model(model, np.vstack(val_features),acquisition_function=config.al_params.acquisition_function)
                    except Exception as e:
                        print('Early termination of AL ', e)
                        break

                elif config.global_params.model_architecture in ('enhanced_molformer', 'advanced_molformer'):
                    # os.environ['MASTER_ADDR'] = 'localhost'
                    # os.environ['MASTER_PORT'] = '12355'
                    # print('rank is ', rank)
                    # dist.init_process_group("gloo", rank=rank, world_size=world_size)

                    if config.global_params.model_architecture == 'enhanced_molformer':
                        model = EnhancedMoLFormer(dropout_rate=0.1).to(rank)
                    else:
                        model = AdvancedMoLFormer(dropout_rate=0.5).to(rank)
                    
                    if world_size>1:
                        model = DDP(model, device_ids=[rank])
                    ros = RandomOverSampler(random_state=42)
                    train_smiles_resampled, train_labels_resampled = ros.fit_resample(
                        np.array(train_data_class[iteration].train.smiles).reshape(-1, 1),
                        train_dock_labels
                    )
                    train_smiles_resampled = train_smiles_resampled.flatten().tolist()
                    print('Resampled train_labels distribution:', Counter(train_labels_resampled))
                    initalX = (
                            model.module.tokenizer(train_smiles_resampled, padding=True, truncation=True, return_tensors="pt")
                            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                            else model.tokenizer(train_smiles_resampled, padding=True, truncation=True, return_tensors="pt")
                        )
                    valX = (
                            model.module.tokenizer(train_data_class[iteration].validation.smiles, padding=True, truncation=True, return_tensors="pt")
                            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                            else model.tokenizer(train_data_class[iteration].validation.smiles, padding=True, truncation=True, return_tensors="pt")
                        )
                    
                    # Create datasets
                    train_dataset = MolFormerDataset(
                        initalX, torch.tensor(train_labels_resampled,dtype = torch.long)
                    )
                    val_dataset = MolFormerDataset(
                        valX, torch.tensor(validation_dock_labels, dtype = torch.long)
                    )

                    train_loader = DataLoader(
                        dataset=train_dataset,
                        batch_size=512,
                        shuffle=True,
                        num_workers=0
                    )
                    val_loader = DataLoader(
                        dataset=val_dataset,
                        batch_size=512,
                        shuffle=False,
                        num_workers=0
                    )

                    # model = model.to(rank)
                    # model = DDP(model, device_ids=[rank])
                    try:
                        model, total_val_loss, _, _, _ = train_molformer_model(
                            model, train_loader, val_loader, num_epochs=config.model_hps.n_epochs,
                            lr= config.model_hps.lr, weight_decay=0.01, model_save_path=f"{config.global_params.project_path}/{config.global_params.project_name}/iteration_{iteration}/",
                            config=config, patience=config.model_hps.patience, rank=rank)
                    except Exception as e:
                        print('Early termination of AL ', e)
                    
                    pred_val_docking_labels, pred_val_proba, _ = predict_with_molformer(model, val_loader,acquisition_function='greedy', rank=rank) #greedy to not use dropout for inference to get correct metric computation #config.al_params.acquisition_function) #predict_with_model(model, np.vstack(val_features),acquisition_function=config.al_params.acquisition_function)
                    # dist.destroy_process_group()
            torch.save(model.state_dict(), f'{config.global_params.project_path}/{config.global_params.project_name}/iteration_{iteration}/docking_model_{config.global_params.model_architecture}.pt')

            print('mainvina.py true_val_docking_labels', np.array(validation_dock_labels))
            print('mainvina.py pred_val_docking_labels', np.array(pred_val_docking_labels))
            val_metrics = log_metrics(validation_dock_labels,pred_val_docking_labels, iteration)
            tmp_dict = {'Val_Precision':val_metrics['precision'],
                        'Val_Recall':val_metrics['recall'],
                        'Val_F1':val_metrics['f1'],
                        'Val_AUC':val_metrics['auc'],
                        'Val_loss':total_val_loss,
                        'Val_FP':val_metrics['fp'],
                        'Val_TP':val_metrics['tp'],
                        'Val_FN':val_metrics['fn'],
                        'Val_TN':val_metrics['tn'],
                        # 'Val_enrichment1':val_metrics['enrichment_1']
                        }
            info_vals.update(tmp_dict)


            # Model testing on apriori sampled random 
            if config.global_params.model_architecture in ('mlp3K','mlpTx'):
                dataset = TensorDataset(
                    torch.tensor(it0_data.test.features, dtype=torch.float32),
                    # torch.tensor(np.vstack(it0_data.test.features), dtype=torch.float32),
                    torch.tensor([1]*len(it0_data.test.features), dtype=torch.long) #dummy labels
                    )
                dataloader = DataLoader(
                        dataset=dataset,
                        batch_size=1024 * 2,
                        shuffle=False,
                        num_workers=1
                    )
                pred_test_docking_labels, pred_test_proba, _ = predict_with_model(model, dataloader,acquisition_function="greedy") #greedy to not use dropout for inference to get correct metric computation #config.al_params.acquisition_function) #predict_with_model(model, np.vstack(it0_data.test.features),acquisition_function=config.al_params.acquisition_function)
            
            elif config.global_params.model_architecture in ('molformer','enhanced_molformer','advanced_molformer'):
                # print('alhelpers.py apriori smiles ', len(it0_data.test.smiles))
                molformer_features = (
                            model.module.tokenizer(it0_data.test.smiles, padding=True, truncation=True, return_tensors="pt")
                            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                            else model.tokenizer(it0_data.test.smiles, padding=True, truncation=True, return_tensors="pt")
                        )
                # print('alhelpers.py apriori molformer features ', molformer_features)
                dataset = MolFormerDataset(molformer_features, 
                                        torch.tensor([1] * len(molformer_features['input_ids']), dtype=torch.long))
                dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=512,
                    shuffle=False,
                    num_workers=0
                )
                pred_test_docking_labels, pred_test_proba, _ = predict_with_molformer(model, dataloader,acquisition_function="greedy", rank=rank) #greedy to not use dropout for inference to get correct metric computation #config.al_params.acquisition_function) #predict_with_model(model, np.vstack(it0_data.test.features),acquisition_function=config.al_params.acquisition_function)

            
            true_test_docking_labels = (it0_data.test.dock_scores < dd_cutoff).astype(int)

            print(f'iteration {iteration} apriori test set w/ cutoff {dd_cutoff}, true test label distribution {Counter(true_test_docking_labels)} ')
            apriori_test_metrics = log_metrics(true_test_docking_labels,pred_test_docking_labels, iteration)
            apriori_test_metrics.update({'enrichment_1':compute_enrichment(pred_test_proba,it0_data.test.dock_labels)})
            tmp_dict = {'AprioriTest_Precision':apriori_test_metrics['precision'],
                        'AprioriTest_Recall':apriori_test_metrics['recall'],
                        'AprioriTest_F1':apriori_test_metrics['f1'],
                        'AprioriTest_AUC':apriori_test_metrics['auc'],
                        'Apriori_FP':apriori_test_metrics['fp'],
                        'Apriori_TP':apriori_test_metrics['tp'],
                        'Apriori_FN':apriori_test_metrics['fn'],
                        'Apriori_TN':apriori_test_metrics['tn'],
                        'Apriori_enrichment1':apriori_test_metrics['enrichment_1']}
            info_vals.update(tmp_dict)


            if config.al_params.test_rand_samples:
                # # Model testing on randomly drawn test samples
                rand_test_docking_scores, _ = get_vina_scores_mul_gpu([item[1] for item in rand_test_samples], molecule_df, config, num_gpus=config.model_hps.num_gpus, 
                                                            output_dir=f"{config.global_params.project_path}/{config.global_params.project_name}/iteration_{iteration}/vina_results",
                                                            dockscore_gt=smiles_2_dockscore_gt)
                rand_test_features = molids_2_fps(mol_ids= [item[0] for item in rand_test_samples], molecule_df=molecule_df, fast=True)
                rand_test_smiles = [item[1] for item in rand_test_samples]
                rand_test_docking_scores = [smiles_2_dockscore_gt[smiles] for smiles in rand_test_smiles]

                if config.global_params.model_architecture in ('mlp3K','mlpTx'):
                    rnd_dataset = TensorDataset(
                        torch.tensor(np.vstack(rand_test_features), dtype=torch.float32),
                        # torch.tensor(np.vstack(it0_data.test.features), dtype=torch.float32),
                        torch.tensor([1]*len(rand_test_features), dtype=torch.long) #dummy labels
                        )
                    rnd_dataloader = DataLoader(
                            dataset=rnd_dataset,
                            batch_size=1024 * 2,
                            shuffle=False,
                            num_workers=1
                        )
                    pred_rand_test_docking_labels, pred_rand_test_proba, _ = predict_with_model(model, rnd_dataloader,acquisition_function="greedy") #greedy to not use dropout for inference to get correct metric computation #config.al_params.acquisition_function)
                elif config.global_params.model_architecture in ('molformer','enhanced_molformer','advanced_molformer'):
                    molformer_features = (
                            model.module.tokenizer(rand_test_smiles, padding=True, truncation=True, return_tensors="pt")
                            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                            else model.tokenizer(rand_test_smiles, padding=True, truncation=True, return_tensors="pt")
                        )
                   
                    rnd_dataset = MolFormerDataset(molformer_features, 
                                            torch.tensor([1] * len(molformer_features['input_ids']), dtype=torch.long))
                    rnd_dataloader = DataLoader(
                        dataset=rnd_dataset,
                        batch_size=512,
                        shuffle=False,
                        num_workers=0
                    )
                    pred_rand_test_docking_labels, pred_rand_test_proba, _ = predict_with_molformer(model, rnd_dataloader, acquisition_function="greedy", rank=rank) #greedy to not use dropout for inference to get correct metric computation  #config.al_params.acquisition_function)
                
                # pred_rand_test_docking_labels = [item[3] for item in rand_test_samples]
                true_rand_test_docking_labels = (rand_test_docking_scores < dd_cutoff).astype(int)
                print(f'iteration {iteration} RANDOM test set w/ cutoff {dd_cutoff}, true test label distribution {Counter(true_rand_test_docking_labels)} ')
                rnd_test_metrics = log_metrics(true_rand_test_docking_labels,pred_rand_test_docking_labels,iteration)
                tmp_dict = {'RndTest_Precision':rnd_test_metrics['precision'],
                            'RndTest_Recall':rnd_test_metrics['recall'],
                            'RndTest_F1':rnd_test_metrics['f1'],
                            'RndTest_AUC':rnd_test_metrics['auc']}
                info_vals.update(tmp_dict)
            keys = sorted(info_vals.keys())

            wandb.log(info_vals)
    
    print("Active learning completed.")
    train_data_class[iteration].test = it0_data.test
    with open(f'{iteration_dir}/all_docked_mols.pkl','wb') as f:
        pickle.dump(train_data_class[iteration], f)
        
    #Final DOCKING
    all_mols_virthits_files = glob.glob(f'{iteration_dir}/allmols_virthits_*.pkl')
    all_virtual_hits = []
    for f in all_mols_virthits_files:
        all_virtual_hits.extend(pickle.load(open(f,'rb'))[1])
    
    topkdf = get_topK_mols(all_docked_mols=train_data_class[iteration],
                           all_virtual_hits=all_virtual_hits, config = config, topK = config.global_params.topK, 
                           dock_tolerance=0.1)
    
    cols = [c for c in topkdf.columns if c != "mol"]
    topkdf.to_csv(f'{iteration_dir}/final_dock_res.csv', columns=cols)
    write_final_sdf_from_df(topkdf, f'{iteration_dir}/final_dock_mols.sdf')

    # clear_iter(os.path.join(config.global_params.project_path, 
    #                                 config.global_params.project_name, f'iteration_{iteration}'),
    #                                 model_clean = False)
    if vina is not None: vina._teardown()


from rdkit import Chem
from rdkit.Chem import SDWriter
import pandas as pd

def write_final_sdf_from_df(df_topK: pd.DataFrame, sdf_path: str = "final_dock_mols.sdf"):
    """
    Write an SDF of docked poses from a DataFrame that has columns:
      - 'mol_id'     : str/int identifier
      - 'smiles'     : input SMILES (string)
      - 'dock_score' : float (e.g., kcal/mol; lower is better)
      - 'mol'        : RDKit Mol (docked 3D pose)  [Vina branch]
    Rows with mol=None are skipped. Order is preserved (matches CSV if you save df_topK as-is).
    """
    required = {"mol_id", "smiles", "dock_score", "mol"}
    missing = required - set(df_topK.columns)
    if missing:
        print(f"[write_final_sdf_from_df] Missing columns: {sorted(missing)}. Skipping SDF write.")
        return

    w = SDWriter(sdf_path)
    n_written, n_skipped = 0, 0

    for _, row in df_topK.iterrows():
        mol = row["mol"]
        if mol is None:
            n_skipped += 1
            continue

        # Make a shallow copy to avoid mutating original object’s props
        m = Chem.Mol(mol)

        # Set SD properties (always strings)
        m.SetProp("mol_id", str(row["mol_id"]))
        m.SetProp("smiles", str(row["smiles"]))
        m.SetProp("dock_score", str(row["dock_score"]))

        # Optional: keep name field in SDF as mol_id
        try:
            m.SetProp("_Name", str(row["mol_id"]))
        except Exception:
            pass

        w.write(m)
        n_written += 1

    w.close()
    print(f"Wrote {n_written} molecules to {sdf_path} (skipped {n_skipped} with no pose).")

