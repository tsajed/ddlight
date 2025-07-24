import pandas as pd
import torch
import sys
sys.path.append('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL')
from gpuvina import get_vina_scores_mul_gpu#, QuickVina2GPU
from easydict import EasyDict
import yaml
from typing import List, Tuple

def sort_by_pred_proba(
    mol_list: List[Tuple[str, str, float]]
) -> List[Tuple[str, str, float]]:
    return sorted(mol_list, key=lambda x: x[2], reverse=True)

with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/config/params.yml', 'r') as f:
    config = EasyDict(yaml.safe_load(f))

def get_topK_mols(all_docked_mols, all_virtual_hits, config, topK=1000, dock_tolerance = 0.1):
    '''dock tolerance : what % more molecules to dock beyond topK to get real topK'''
    dock_thresh = all_docked_mols.train.cutoff
    result = []
    for key in all_docked_mols:
        mols_ids = all_docked_mols[key].mol_ids
        smiles = all_docked_mols[key].smiles
        dock_scores = all_docked_mols[key].dock_scores
        for m,s,d in (zip(mols_ids,smiles,dock_scores)):
            if d<dock_thresh:
                result.append((m,s,d))
    top_virt_hits = sort_by_pred_proba(all_virtual_hits)[0:int(1+dock_tolerance)*(max(topK-len(result),0))]
    print(len(top_virt_hits))
    
    # dock top virt hits 
    dock_mol_ids = [mol[0] for mol in top_virt_hits]
    dock_smiles_list = [mol[1] for mol in top_virt_hits]
    dock_scores = get_vina_scores_mul_gpu(dock_smiles_list, None, config, num_gpus=config.model_hps.num_gpus, 
                                        output_dir=f"{config.global_params.project_path}/{config.global_params.project_name}/final_docking/",
                                        dockscore_gt=None)
    top_virt_result = [(m,s,d) for m,s,d in zip(dock_mol_ids,dock_smiles_list, dock_scores)]
    # Combine both and sort by docking score
    combined = result + top_virt_result
    combined_sorted = sorted(combined, key=lambda x: x[2])  # Lower score = better

    # Take top-K
    topK_mols = combined_sorted[:topK]

    # Convert to DataFrame
    df_topK = pd.DataFrame(topK_mols, columns=["mol_id", "smiles", "dock_score"])
    
    return df_topK
    
# topkdf = get_topK_mols(all_docked_mols, all_virutal_hits,config, topK=2000)
# topkdf.tail(100)