import pandas as pd
import pickle
import sys
sys.path.append('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/Experiments/')
import argparse
# from AcqFnComparision_molformer import load_config
from ALHelpers_molformer import fetch_unlabeled_batch_mul_gpus, MolFormerDataset, predict_with_molformer, wait_for_slurm_completion, initialize_model
import subprocess
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from easydict import EasyDict
import yaml

def load_config(file_path):
    with open(file_path, 'r') as f:
        return EasyDict(yaml.safe_load(f))


def write_inf_helper_script(docking_model_path, iteration_dir, config):
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
source ~/anaconda3/etc/profile.d/conda.sh
conda activate {config.global_params.env_name}
python {config.global_params.code_path}/VinaAL/Experiments/Final_Regression/topk_classfn_helper.py $SLURM_ARRAY_TASK_ID {docking_model_path} {iteration_dir} '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/config/params.yml'
    """
    #{config.global_params.code_path}/config/params.yml
    script_file = os.path.join(iteration_dir,f'parallel_inf.sh')  # sdf_dir.rsplit('/', 2)[0] + '/'    #os.path.join(sdf_dir, f'{name}_conf.sh') #
    print('inf.py script_file  ', script_file)
    with open(script_file, 'w') as f:
        f.write(script_content)
    return script_file


def run_inference_scatter(model, iteration_dir, split_molecule_df, config, tokenizer=None, task_id=0, num_tasks=1):
    top_molecules, virtual_hits, all_molecules = [], [], []
    if config.global_params.model_architecture in ('molformer', 'enhanced_molformer', 'advanced_molformer'):
        unlabeled_batch, len_av_df = fetch_unlabeled_batch_mul_gpus(split_molecule_df, used_zinc_ids=[], fingerprint=False, tokenizer=tokenizer)
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
                                                            acquisition_function='greedy') # scores is uncertainty (or acquisition score)
        
    print(f'{sum(pred_labels)}/{len(pred_labels)} predicted as actives')
    # Iterate over the batch to filter top_k and virtual hits. 
    # Convert lists to NumPy arrays for vectorized operations
    mol_ids = np.array(mol_ids)
    smiles_list = np.array(smiles_list)
    pred_labels = np.array(pred_labels)
    pred_proba = np.array(pred_proba)

    # Identify virtual hits (label == 1)
    is_virtual_hit = pred_labels == 1
    virtual_hits.extend(zip(mol_ids[is_virtual_hit], smiles_list[is_virtual_hit], pred_proba[is_virtual_hit]))

    # Collect all molecules in a single operation
    all_molecules.extend(zip(mol_ids, smiles_list, pred_labels, pred_proba))

    with open(f'{iteration_dir}/allmols_virthits_{task_id}.pkl','wb') as f:
        pickle.dump((all_molecules, virtual_hits),f)


def gather(iteration_dir, top_k, full_mol_df):
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
    
    pass

def main(seed_idx, used_zinc_ids, iteration_dir, config):
    split_molecule_df = pd.read_csv(args.iteration_dir+f'/split_mol_df_{seed_idx}.csv')
    model = initialize_model(config.global_params.model_architecture)
    iteration_num = int(iteration_dir.rstrip('/').split('_')[-1])  # Extracts '1' and converts to int
    prev_iteration_num = iteration_num - 1
    prev_iteration_dir = iteration_dir.replace(f'iteration_{iteration_num}', f'iteration_{prev_iteration_num}')

    # Define model path for previous iteration
    model_load_path = os.path.join(prev_iteration_dir, f'docking_model_{config.global_params.model_architecture}.pt')

    best_model_state = torch.load(model_load_path, map_location=torch.device('cpu'))  # Adjust map_location if using GPU
    model.load_state_dict(best_model_state)
    print(torch.cuda.is_available())  
    print(torch.cuda.device_count())  
    print(torch.cuda.get_device_name(0))  

    model.to(torch.device('cuda:0'))
    model.eval()
    run_inference_scatter(model, iteration_dir, split_molecule_df, used_zinc_ids, config, tokenizer=model.tokenizer, task_id=seed_idx)
    pass

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dock_model_path", type=str,  help="Path to the classification model to be useed")
    parser.add_argument("project_path", type=str)
    args = parser.parse_args()
    config_path = '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/config/params.yml'
    config = load_config(config_path)
    if config.global_params.target == 'mpro':
        with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/data/lsd_dock_mpro/778M_mols_w_dockscores.pkl','rb') as f:
            molecule_df = pickle.load(f)
        # with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/data/lsd_dock_mpro/smiles_2_dockscore_gt.pkl','rb') as f:
        #     smiles_2_dockscore_gt = pickle.load(f)
    elif config.global_params.target == 'mt1r':
        molecule_df = pd.read_csv('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/data/lsd_dock_mt1r/40M_mols_w_dockscores.csv')
    if config.global_params.target in ['jak2', 'braf', 'parp1','fa7', '5ht1b']:
        molecule_df = pickle.load(open(f'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Projects/dock_{config.global_params.target}/iteration_0/2M_mols_w_dockscores_{config.global_params.target}.pkl','rb'))
        # molecule_df = molecule_df[molecule_df[f'{config.global_params.target}_dockscores']!=0]
        molecule_df.drop(columns=["indices"], inplace=True)
    
    os.makedirs(args.project_path+'/Regression/',exist_ok=True)
            
    available_df_numpy_array = molecule_df.to_numpy()
    for task_id in range(0, config.model_hps.num_gpus):
        # Split dataset for parallel processing
        print('drop db is ',  config.al_params.drop_db )
        
        total_mols = len(molecule_df)
        chunk_size = (total_mols + config.model_hps.num_gpus - 1) // config.model_hps.num_gpus  # Ensure even splitting
        start_idx = task_id * chunk_size
        end_idx = min((task_id + 1) * chunk_size, total_mols)
        # split_molecule_df = available_df.iloc[start_idx:end_idx]  # Process only assigned chunk
        # split_molecule_df = pd.DataFrame(available_df.values[start_idx:end_idx], columns=available_df.columns)
        split_molecule_df = pd.DataFrame(available_df_numpy_array[start_idx:end_idx], columns=molecule_df.columns)
        print(f"SLURM Task {task_id}: Processing molecules {start_idx} to {end_idx}")
        split_molecule_df.to_csv(args.project_path+'/Regression'+f'/split_mol_df_{task_id}.csv', index=False)

    parallel_inf_sh_path = write_inf_helper_script(args.dock_model_path, args.project_path+'/Regression/', config)
    cmd = ["sbatch", parallel_inf_sh_path ]
    sbatch_output = subprocess.run(cmd, capture_output=True, text=True, check=True)
    wait_for_slurm_completion(sbatch_output)