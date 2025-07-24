import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
from architecture import MyModel, DockingModel3K, DockingModel6K, DockingModelTx, MoLFormer, HybridMolTx, EnhancedMoLFormer, AdvancedMoLFormer
from graph_dataset import MyDataset, smi_collate_fn, MolFormerDataset
from imblearn.over_sampling import RandomOverSampler
from model_train_arch_comp_wandb import train_graph_model, train_hybridMolTx_model, train_docking_model, train_random_forest_model, train_molformer_model
from metrics import log_metrics
from collections import Counter
import numpy as np
from functools import partial
from torch.utils.data import DataLoader, TensorDataset
from easydict import EasyDict
import argparse
import torch
import wandb
from sklearn.model_selection import train_test_split

wandb.login(key= '50c4542f2f5338f2591116005f2e2c8bd9f4d6d6')
            # '0e682f0e8b41a00854807fc3bd4e6e0c58e73f29')


def run_experiment(target, architecture):
    config = EasyDict({
        'Project_ID': 'Manuscript_arch_comp',
        'arch': architecture,
        'oversampling': True,
        'undersampling': False,
        'it0_dock_cutoff': -10,
        'end': 100000,
        'lr': 0.000001,
        "global_params": {
            "target": target,
            "model_architecture": architecture,
            "loss": 'crossentropy'
        }
    })
    print(f"Running experiment for target: {target}, architecture: {architecture}")
    print(config)

    def create_dense_fp(indices):
        """Helper function to create a dense fingerprint from sparse indices."""
        indices = np.frombuffer(indices, dtype=np.int32)
        dense_fp = np.zeros((1, 2048))
        dense_fp[0, indices] = 1
        return dense_fp.squeeze()
    molecule_df = pickle.load(open(f'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Projects/dock_{target}/iteration_0/2M_mols_w_dockscores_{target}.pkl','rb'))
    molecule_df = molecule_df[molecule_df[f'{target}_dockscores']!=0]
    molecule_df = molecule_df.sample(config.end*2, random_state=42)
    config.it0_dock_cutoff = np.percentile(molecule_df[f'{target}_dockscores'].tolist(), 1)

    train_df, val_df = train_test_split(molecule_df, test_size=0.5, random_state=42)
    it0_data_train_mol_ids = train_df['zinc_id'].tolist()
    it0_data_train_dock_scores = train_df[f'{target}_dockscores'].tolist()
    it0_train_features = [create_dense_fp(indices) for indices in train_df['indices']]
    it0_data_train_smiles_list = train_df['smiles'].tolist()
    it0_data_val_mol_ids = val_df['zinc_id'].tolist()
    it0_data_val_dock_scores = val_df[f'{target}_dockscores'].tolist()
    it0_val_features = [create_dense_fp(indices) for indices in val_df['indices']]
    it0_data_val_smiles_list = val_df['smiles'].tolist()


    if architecture in ('gcn', 'gat', 'gin', 'gine'):
        train_labels = [1 if value < config.it0_dock_cutoff else 0 for value in it0_data_train_dock_scores[0:config.end]]
        val_labels = [1 if value < config.it0_dock_cutoff else 0 for value in it0_data_val_dock_scores[0:config.end]]
        print('Initial train_labels distribution:', Counter(train_labels))
        print('Initial val_labels distribution:', Counter(val_labels))

        if config.oversampling:
            ros = RandomOverSampler(random_state=42)
            train_smiles_resampled, train_labels_resampled = ros.fit_resample(
                np.array(it0_data_train_smiles_list[0:config.end]).reshape(-1, 1),
                train_labels
            )
            train_smiles_resampled = train_smiles_resampled.flatten().tolist()
            print('Resampled train_labels distribution:', Counter(train_labels_resampled))
            train_dataset = MyDataset(
                it0_data_train_mol_ids[0:config.end],
                train_smiles_resampled,
                train_labels_resampled
            )
        else:
            train_dataset = MyDataset(it0_data_train_mol_ids[0:config.end], it0_data_train_smiles_list[0:config.end], train_labels)

        val_dataset = MyDataset(it0_data_val_mol_ids[0:config.end], it0_data_val_smiles_list[0:config.end], val_labels)
        collate_fn = partial(smi_collate_fn)
        train_loader = DataLoader(train_dataset, batch_size=1024 * 2, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=1024 * 2, shuffle=False, collate_fn=collate_fn)

        model = MyModel(model_type=architecture, out_dim=2, readout='pma')
        train_graph_model(model, train_loader, val_loader, num_epochs=40, config=config, patience=5)

    elif architecture in ('mlp6K', 'mlp3K', 'mlpTx'):
        train_labels = [1 if value < config.it0_dock_cutoff else 0 for value in it0_data_train_dock_scores[0:config.end]]
        val_labels = [1 if value < config.it0_dock_cutoff else 0 for value in it0_data_val_dock_scores[0:config.end]]
        print('Initial train_labels distribution:', Counter(train_labels))
        print('Initial val_labels distribution:', Counter(val_labels))

        if config.oversampling:
            ros = RandomOverSampler(random_state=42)
            X_train_resampled, y_train_resampled = ros.fit_resample(it0_train_features[0:config.end], train_labels)
            print('Resampled train_labels distribution:', Counter(y_train_resampled))
        else:
            X_train_resampled, y_train_resampled = it0_train_features[0:config.end], train_labels

        train_dataset = TensorDataset(torch.tensor(X_train_resampled, dtype=torch.float32), torch.tensor(y_train_resampled, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(it0_val_features[0:config.end], dtype=torch.float32), torch.tensor(val_labels, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=1024 * 2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024 * 2, shuffle=False)

        if architecture == 'mlpTx':
            model = DockingModelTx(input_size=2048, p=0.1)
        elif architecture == 'mlp6K':
            model = DockingModel6K(input_size=2048, p=0.1)
        elif architecture == 'mlp3K':
            model = DockingModel3K(input_size=2048, p=0.1)

        train_docking_model(model, train_loader, val_loader, num_epochs=40, lr=config.lr, config=config,
                            model_save_path='/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/Experiments/Archcomp/')


    elif architecture in ['molformer', 'enhanced_molformer', 'advanced_molformer']:
        if architecture == 'molformer':
            model = MoLFormer()
        elif architecture == 'enhanced_molformer':
            model = EnhancedMoLFormer()
        elif architecture == 'advanced_molformer':
            model = AdvancedMoLFormer()

        train_labels = [1 if value < config.it0_dock_cutoff else 0 for value in it0_data_train_dock_scores[0:config.end]]
        val_labels = [1 if value < config.it0_dock_cutoff else 0 for value in it0_data_val_dock_scores[0:config.end]]
        print('Initial train_labels distribution:', Counter(train_labels))
        print('Initial val_labels distribution:', Counter(val_labels))

        if config.oversampling:
            ros = RandomOverSampler(random_state=42)
            train_smiles_resampled, train_labels_resampled = ros.fit_resample(np.array(it0_data_train_smiles_list[0:config.end]).reshape(-1, 1), train_labels)
            train_smiles_resampled = train_smiles_resampled.flatten().tolist()
        else:
            train_smiles_resampled, train_labels_resampled = it0_data_train_smiles_list[0:config.end], train_labels

        initialX = model.tokenizer(train_smiles_resampled, padding=True, truncation=True, return_tensors="pt")
        valX = model.tokenizer(it0_data_val_smiles_list[0:config.end], padding=True, truncation=True, return_tensors="pt")

        train_dataset = MolFormerDataset(initialX, torch.tensor(train_labels_resampled, dtype=torch.long))
        val_dataset = MolFormerDataset(valX, torch.tensor(val_labels, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=1024 // 2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024 // 2, shuffle=False)

        train_molformer_model(model, train_loader, val_loader, num_epochs=40, lr=config.lr, config=config, patience=10,
                                        model_save_path='/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/Experiments/Archcomp/')

    elif architecture == 'rf':
        # Create labels for a limited dataset [0:config.end]
        train_labels = [1 if value < config.it0_dock_cutoff else 0 for value in it0_data_train_dock_scores[0:config.end]]
        val_labels = [1 if value < config.it0_dock_cutoff else 0 for value in it0_data_val_dock_scores[0:config.end]]

        print('Initial train_labels distribution:', Counter(train_labels))
        print('Initial val_labels distribution:', Counter(val_labels))

        if config.oversampling:
            ros = RandomOverSampler(random_state=42)
            X_train_resampled, y_train_resampled = ros.fit_resample(
                it0_train_features[0:config.end], train_labels
            )
            print('Resampled train_labels distribution:', Counter(y_train_resampled))
        else:
            # Limit training features to [0:config.end]
            X_train_resampled, y_train_resampled = it0_train_features[0:config.end], train_labels

        # Limit validation features to [0:config.end]
        val_features_limited = it0_val_features[0:config.end]

        # Train the random forest model
        rf_model = train_random_forest_model(
            train_features=X_train_resampled,
            train_labels=y_train_resampled,
            val_features=val_features_limited,
            val_labels=val_labels,
            n_estimators=100,
            max_depth=None,
            config=config,
            model_save_path='/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/Experiments/Archcomp/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run architecture experiment")
    parser.add_argument('--target', type=str, required=True, help="Target name")
    parser.add_argument('--architecture', type=str, required=True, help="Architecture type")

    args = parser.parse_args()
    target = args.target
    architecture = args.architecture

    # Call the function with specified target and architecture
    run_experiment(target, architecture)