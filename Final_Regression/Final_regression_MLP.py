# %%
import pickle
from tqdm import tqdm_notebook, tqdm
import pandas as pd
from rdkit.Chem import AllChem
import numpy as np
from rdkit import Chem
tqdm.pandas()
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# %%
with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Projects/DDS_AL_2M_2_dynamicVAL_mpro_bald_advanced_molformer_False_True/Regression/pred_active_mols_100K_df.pkl','rb') as f:
    pred_active_mols_250K_df = pickle.load(f)

with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Projects/DDS_AL_2M_2_dynamicVAL_mpro_bald_advanced_molformer_False_True/Regression/true_top1pct_df.pkl','rb') as f:
    true_top1pct_df = pickle.load(f)

# %%
def generate_morgan_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality = not True, radius=radius, nBits=nBits)
    bit_string = fp.ToBitString()
    return np.array([list(map(int, bit_string))])

# %%
pred_active_mols_250K_df['morgan']= pred_active_mols_250K_df.smiles_x.progress_apply(lambda x: generate_morgan_fingerprint(x))

# %%
pred_active_mols_250K_df

# %%
# Split dataset into train (80%), validation (10%), and test (10%)
train_df, temp_df = train_test_split(pred_active_mols_250K_df, test_size=0.2, random_state=42)  # 80% train
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)  # 10% val, 10% test

X_train, X_val, X_test = np.concatenate(train_df.morgan.values), np.concatenate(val_df.morgan.values), np.concatenate(test_df.morgan.values)
y_train, y_val, y_test = train_df.mpro_dockscores.values, val_df.mpro_dockscores.values, test_df.mpro_dockscores.values

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=1
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=1
)
test_loader = DataLoader(
    dataset=val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=1
)

# %%
class OverparamMLPTxRegression(nn.Module):
    def __init__(self, dropout_rate=0.1, num_dense_layers=6, num_residual_blocks=10, num_attention_layers=3):
        super(OverparamMLPTxRegression, self).__init__()
        # smiles_model = "ibm/MoLFormer-XL-both-10pct"

        # self.tokenizer = AutoTokenizer.from_pretrained(smiles_model, trust_remote_code=True)
        # self.base_model = AutoModel.from_pretrained(smiles_model, deterministic_eval=True, trust_remote_code=True)

        hidden_dim = 2048 #self.base_model.config.hidden_size
        intermediate_dim = 5 * hidden_dim  # Increased intermediate layer size

        # Multi-layer attention pooling
        self.attention_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_attention_layers)
        ])

        # Deep Dense Layers (Overparameterization)
        self.dense_layers = nn.ModuleList([
            nn.Linear(hidden_dim*num_attention_layers + i * hidden_dim, hidden_dim) for i in range(num_dense_layers)
        ])

        concatenated_size = hidden_dim * num_attention_layers + (hidden_dim * num_dense_layers)
        self.project_dense = nn.Linear(concatenated_size, hidden_dim)

        # Multihead Attention Layers (Stacked for overparameterization)
        self.multihead_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=16, dropout=dropout_rate, batch_first=True)
            for _ in range(num_attention_layers)
        ])

        # Deep Residual Blocks
        self.residual_blocks = nn.Sequential(*[
            self._create_residual_block(hidden_dim, intermediate_dim, dropout_rate)
            for _ in range(num_residual_blocks)
        ])

        # Fully connected overparameterized layers
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.norm1 = nn.LayerNorm(intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, intermediate_dim // 2)
        self.norm2 = nn.LayerNorm(intermediate_dim // 2)
        self.fc3 = nn.Linear(intermediate_dim // 2, hidden_dim)  # Additional FC layer

        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(hidden_dim, 1)  # Regression output

    def _create_residual_block(self, input_dim, output_dim, dropout_rate):
        """Creates a deep residual block with overparameterization"""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(output_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.PReLU(),
        )

    def forward(self, x):
        base_outputs = x.unsqueeze(dim=1)
        # base_outputs = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state

        # Multi-layer attention pooling
        # pooled_output = base_outputs
        pooled_outputs = []
        for i, attn_layer in enumerate(self.attention_layers):
            # print('tryng attn layer base output', base_outputs.shape)
            # print(attn_layer(base_outputs).shape)
            # print(i)
            attention_weights = torch.softmax(attn_layer(base_outputs), dim=1)
            pooled_output = torch.sum(attention_weights * base_outputs, dim=1)
            pooled_outputs.append(pooled_output)
            # print('pooled_output', pooled_output.shape)
        # Concatenate pooled outputs across all attention layers
        pooled_output = torch.cat(pooled_outputs, dim=1)  # Ensures shape consistency

        # Deep dense connections
        dense_out = pooled_output
        # print('dense out shape ',dense_out.shape)
        for dense_layer in self.dense_layers:
            dense_out = torch.cat([dense_out, dense_layer(dense_out)], dim=1)
            # print('dense out ', dense_out.shape)

        dense_out = self.project_dense(dense_out)

        # Stacked Multihead Attention
        attention_out = dense_out.unsqueeze(1)
        for attn_layer in self.multihead_attention_layers:
            attention_out, _ = attn_layer(attention_out, attention_out, attention_out)

        attention_out = attention_out.squeeze(1)

        # Deep Residual Blocks
        x = self.residual_blocks(attention_out)
        # print('x ', x.shape)
        # Overparameterized Fully Connected Layers
        x = self.norm1(torch.relu(self.fc1(x)))
        x = self.norm2(torch.relu(self.fc2(x)))
        x = torch.relu(self.fc3(x))  # Extra FC layer

        x = self.dropout(x)
        regression_output = self.output_layer(x)

        return regression_output.squeeze()

# %%
import copy
def train_mlp_regression(model, train_loader, val_loader, num_epochs=50, lr=1e-4, patience=9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-8)

    best_val_loss = float("inf")
    early_stop_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_features, batch_labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        
        avg_val_loss = val_loss
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
            print(f"New best validation loss: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss for {early_stop_counter} consecutive epochs.")

        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best epoch was {epoch-early_stop_counter+1}.")
            break
        
    model.load_state_dict(best_model_state)
    return model
        
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), "best_molformer_regression.pt")
        #     print("Best model saved.")



# %%
model = train_mlp_regression(OverparamMLPTxRegression(num_dense_layers=6, dropout_rate=0.1), train_loader, val_loader, lr = 1e-3, patience=9)

# %%
from scipy.stats import pearsonr

def plot_predictions(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    true_scores, pred_scores = [], []

    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)

            true_scores.extend(batch_labels.detach().cpu())
            pred_scores.extend(outputs.detach().cpu())

    # Convert to NumPy arrays
    true_scores = np.array(true_scores)
    pred_scores = np.array(pred_scores).flatten()
    pred_scores = np.round(pred_scores, 2)

    # print(true_scores, pred_scores)

    # Compute Pearson correlation coefficient
    correlation, _ = pearsonr(true_scores, pred_scores)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=true_scores, y=pred_scores, alpha=0.5)
    plt.xlabel("True Docking Scores")
    plt.ylabel("Predicted Docking Scores")
    plt.title(f"Regression Results: True vs. Predicted\nPearson Correlation: {correlation:.4f}")
    
    # Draw diagonal line
    min_val = min(true_scores.min(), pred_scores.min())
    max_val = max(true_scores.max(), pred_scores.max())
    # plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")

    plt.legend()
    plt.show()

    print(f"Pearson Correlation Coefficient: {correlation:.4f}")

# Call the function after training
plot_predictions(model, val_loader)


# %%
batch_labels[0]

# %%



