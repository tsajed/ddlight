import pandas as pd
import argparse
import yaml
from easydict import EasyDict
import pickle
import torch
import os
import sys
sys.path.append('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/Experiments/')
from ALHelpers_molformer import initialize_model
from Final_Regression_inf import run_inference_scatter

def main(seed_idx, docking_model_path, iteration_dir, config):
    split_molecule_df = pd.read_csv(iteration_dir+f'/split_mol_df_{seed_idx}.csv')
    model = initialize_model(config.global_params.model_architecture)
    best_model_state = torch.load(docking_model_path, map_location=torch.device('cpu'))  # Adjust map_location if using GPU
    model.load_state_dict(best_model_state)
    print(torch.cuda.is_available())  
    print(torch.cuda.device_count())  
    print(torch.cuda.get_device_name(0))  
    model.to(torch.device('cuda:0'))
    model.eval()
    run_inference_scatter(model, iteration_dir, split_molecule_df, config, tokenizer=model.tokenizer, task_id=seed_idx)
    pass

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("seed_idx", type=int,  help="Index for experiment configuration")
    parser.add_argument("docking_model_path", type=str,  help=" Path where docking_model_for classification is stored")
    parser.add_argument("split_df_path", type=str,  help=" Path where split molecule df files are stored")
    # parser.add_argument("used_id_path",type=str,  help=" Path where used_zinc_ids so far is stored")
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()
    config_path = args.config_path #'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/config/params.yml'
    with open(config_path, 'rb') as f:
        config =  EasyDict(yaml.safe_load(f))
    main(args.seed_idx, args.docking_model_path, args.split_df_path, config)
