import pandas as pd
import argparse
import yaml
from easydict import EasyDict
import pickle
import torch
import os
import sys
sys.path.append('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL')
from ALHelpers_molformer import initialize_model, run_inference_scatter

def main(seed_idx, used_zinc_ids, iteration_dir, config):
    # split_molecule_df = pd.read_csv(args.iteration_dir+f'/split_mol_df_{seed_idx}.csv')
    split_molecule_df = pickle.load(open(args.iteration_dir+f'/split_mol_df_{seed_idx}.pkl','rb'))
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
    parser.add_argument("seed_idx", type=int,  help="Index for experiment configuration")
    parser.add_argument("iteration_dir", type=str,  help=" Path where split molecule df files are stored")
    # parser.add_argument("used_id_path",type=str,  help=" Path where used_zinc_ids so far is stored")
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()
    config_path = args.config_path #'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/config/params.yml'
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
        # config =  EasyDict(yaml.safe_load(f))
    with open(args.iteration_dir+'/used_zinc_ids.pkl','rb') as f:
        used_zinc_ids = pickle.load(f)
    main(args.seed_idx, used_zinc_ids, args.iteration_dir, config)
