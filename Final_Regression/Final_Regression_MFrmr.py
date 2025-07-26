# %%
import pickle
import pandas as pd
import glob
from tqdm import tqdm_notebook

# # %%
# allmols_virthits_list = glob.glob('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Projects/DDS_AL_2M_2_dynamicVAL_mpro_bald_advanced_molformer_False_True/Regression/allmols_*.pkl')
# all_molecules_list = []
# for file in tqdm_notebook(allmols_virthits_list):
#     with open(file,'rb') as f:
#         (all_molecules, virtual_hits) = pickle.load(f)
#     all_molecules_list.extend(all_molecules)
#     # break

# %%
# with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Projects/DDS_AL_2M_2_dynamicVAL_mpro_bald_advanced_molformer_False_True/Regression/all_molecules_list.pkl','wb') as f:
#     pickle.dump(all_molecules_list,f)
# len(all_molecules_list)

# %%
# all_molecules_list[0:3]

# %%
# target = 'jak2'
# mol_df = pickle.load(open(f'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Projects/dock_{target}/iteration_0/2M_mols_w_dockscores_{target}.pkl','rb'))
# with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/data/lsd_dock_mpro/778M_mols_w_dockscores.pkl','rb') as f:
#     mol_df = pickle.load(f)

# # %%
# mol_df.head(2)

# # %%
# threshold = -37.48 # -9.6
# mol_df['labels'] = mol_df.mpro_dockscores<threshold
# mol_df

# # %%
# import pandas as pd

# # Load ground truth dataframe
# ground_truth_df = mol_df  # Load your actual ground truth CSV

# # Convert predictions into a DataFrame
# predictions = all_molecules_list
# predictions_df = pd.DataFrame(predictions, columns=["zinc_id", "smiles", "pred_label"])

# # Merge predictions with ground truth
# merged_df = ground_truth_df.merge(predictions_df, on="zinc_id", how="inner")

# # Sort by predicted labels (assuming higher scores are better)
# merged_df = merged_df.sort_values(by="pred_label", ascending=False)
# merged_df

# # Compute enrichment
# top_n = int(0.01 * len(merged_df))  # Consider top 1% ranked molecules
# top_n_df = merged_df.iloc[:top_n]

# # Compute fractions
# total_true_fraction = merged_df["labels"].sum() / len(merged_df)
# top_true_fraction = top_n_df["labels"].sum() / top_n

# enrichment = top_true_fraction / total_true_fraction

# print(f"Enrichment Factor (EF1%): {enrichment:.2f}")


# # %%
# # Define screening percentages
# import numpy as np
# from matplotlib import pyplot as plt

# percentages = np.linspace(0.01, 0.10, 10)  # 1% to 10%
# enrichment_values = []

# total_actives = merged_df["labels"].sum()  # Total number of positives in dataset

# for p in percentages:
#     top_n = int(p * len(merged_df))
#     top_n_df = merged_df.iloc[:top_n]
#     top_true_fraction = top_n_df["labels"].sum() / total_actives  # Fraction of actives retrieved
#     enrichment_values.append(top_true_fraction)

# # Plot the Enrichment Curve
# plt.figure(figsize=(8, 6))
# plt.plot(percentages * 100, enrichment_values, marker='o', linestyle='-', color='b', label="Enrichment Curve")
# plt.axhline(y=percentages[-1], linestyle="--", color="gray", label="Random Selection")
# plt.xlabel("Percentage of Dataset Screened (%)")
# plt.ylabel("Fraction of True Positives Retrieved")
# plt.title("Enrichment Plot")
# plt.legend()
# plt.grid(True)
# plt.show()

# # %%
# merged_df.head(2)

# # %%
# merged_df.drop(columns=['smiles_y'], inplace=True)
# # with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Projects/DDS_AL_2M_2_dynamicVAL_mpro_bald_advanced_molformer_False_True/Regression/merged_df.pkl','wb') as f:
# #     pickle.dump(merged_df,f)

# # %% [markdown]
# # # Regression by sampling 250K getting top-1% of molecules from predicted labels and plot scatter plot 

# # %%
# pred_active_mols_250K_df = merged_df[merged_df.pred_label==1].sample(frac=1, random_state=42).head(50000)
# pred_active_mols_250K_df

# %%
# # save pred_active_mols_250K_df and create and save top1pct
# true_top1pct_df = merged_df[merged_df.mpro_dockscores<=threshold]
# true_top1pct_df
# with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Projects/DDS_AL_2M_2_dynamicVAL_mpro_bald_advanced_molformer_False_True/Regression/pred_active_mols_100K_df.pkl','wb') as f:
#     pickle.dump(pred_active_mols_250K_df,f)

# with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Projects/DDS_AL_2M_2_dynamicVAL_mpro_bald_advanced_molformer_False_True/Regression/true_top1pct_df.pkl','wb') as f:
#     pickle.dump(true_top1pct_df,f)
    
with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Projects/DDS_AL_2M_2_dynamicVAL_mpro_bald_advanced_molformer_False_True/Regression/pred_active_mols_100K_df.pkl','rb') as f:
    pred_active_mols_250K_df = pickle.load(f)

with open('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/Projects/DDS_AL_2M_2_dynamicVAL_mpro_bald_advanced_molformer_False_True/Regression/true_top1pct_df.pkl','rb') as f:
    true_top1pct_df = pickle.load(f)

# %%
pred_active_mols_250K_df['len'] = pred_active_mols_250K_df.smiles_x.apply(lambda x: len(x))
pred_active_mols_250K_df.len.describe()

# %%
pred_active_mols_250K_df.mpro_dockscores.hist(bins=100)

# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# %%
class SMILESDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row["smiles_x"]  # Use the correct column containing SMILES
        docking_score = row["mpro_dockscores"]  # True docking score

        encoding = self.tokenizer(
            smiles,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "docking_score": torch.tensor(docking_score, dtype=torch.float),
        }


# %%
class AdvancedMoLFormerRegression(nn.Module):
    def __init__(self, dropout_rate=0.1, num_dense_layers=3):
        super(AdvancedMoLFormerRegression, self).__init__()
        smiles_model = "ibm/MoLFormer-XL-both-10pct"

        self.tokenizer = AutoTokenizer.from_pretrained(smiles_model, trust_remote_code=True)
        self.base_model = AutoModel.from_pretrained(smiles_model, deterministic_eval=True, trust_remote_code=True)

        hidden_dim = self.base_model.config.hidden_size
        intermediate_dim = 10 * hidden_dim

        self.attention_layer = nn.Linear(hidden_dim, 1)

        self.dense_layers = nn.ModuleList([
            nn.Linear(hidden_dim + i * hidden_dim, hidden_dim) for i in range(num_dense_layers)
        ])

        concatenated_size = hidden_dim * (1 + num_dense_layers)
        self.project_dense = nn.Linear(concatenated_size, hidden_dim)

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=dropout_rate, batch_first=True
        )

        self.residual_blocks = nn.Sequential(
            self._create_residual_block(hidden_dim, intermediate_dim, dropout_rate),
            self._create_residual_block(hidden_dim, intermediate_dim, dropout_rate),
            self._create_residual_block(hidden_dim, intermediate_dim, dropout_rate),
            self._create_residual_block(hidden_dim, intermediate_dim, dropout_rate),
            self._create_residual_block(hidden_dim, intermediate_dim, dropout_rate),
            self._create_residual_block(hidden_dim, intermediate_dim, dropout_rate),
        )

        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.norm1 = nn.LayerNorm(intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(hidden_dim, 1)  # Single output for regression

    def _create_residual_block(self, input_dim, output_dim, dropout_rate):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(output_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.PReLU(),
        )

    def forward(self, input_ids, attention_mask):
        base_outputs = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state
        attention_weights = torch.softmax(self.attention_layer(base_outputs), dim=1)
        pooled_output = torch.sum(attention_weights * base_outputs, dim=1)

        dense_out = pooled_output
        for dense_layer in self.dense_layers:
            dense_out = torch.cat([dense_out, dense_layer(dense_out)], dim=1)

        dense_out = self.project_dense(dense_out)

        attention_out, _ = self.multihead_attention(
            dense_out.unsqueeze(1), dense_out.unsqueeze(1), dense_out.unsqueeze(1)
        )

        attention_out = attention_out.squeeze(1)
        x = self.residual_blocks(attention_out)

        x = self.norm1(torch.relu(self.fc1(x)))
        x = self.norm2(torch.relu(self.fc2(x)))

        x = self.dropout(x)
        regression_output = self.output_layer(x)

        return regression_output.squeeze()


# %%
class OverparamMoLFormerRegression(nn.Module):
    def __init__(self, dropout_rate=0.1, num_dense_layers=6, num_residual_blocks=10, num_attention_layers=3):
        super(OverparamMoLFormerRegression, self).__init__()
        smiles_model = "ibm/MoLFormer-XL-both-10pct"

        self.tokenizer = AutoTokenizer.from_pretrained(smiles_model, trust_remote_code=True)
        self.base_model = AutoModel.from_pretrained(smiles_model, deterministic_eval=True, trust_remote_code=True)

        hidden_dim = self.base_model.config.hidden_size
        intermediate_dim = 20 * hidden_dim  # Increased intermediate layer size

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

    def forward(self, input_ids, attention_mask):
        base_outputs = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state

        # Multi-layer attention pooling
        # pooled_output = base_outputs
        pooled_outputs = []
        for i, attn_layer in enumerate(self.attention_layers):
            # print('tryng attn layer base output', base_outputs.shape)
            # print('attn layer opt ', attn_layer(base_outputs).shape)
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
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

# Split dataset into train (80%), validation (10%), and test (10%)
train_df, temp_df = train_test_split(pred_active_mols_250K_df, test_size=0.2, random_state=42)  # 80% train
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)  # 10% val, 10% test

# Create dataset objects
train_dataset = SMILESDataset(train_df, tokenizer)
val_dataset = SMILESDataset(val_df, tokenizer)
test_dataset = SMILESDataset(test_df, tokenizer)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# %%
import copy

def train_molformer_regression(model, train_loader, val_loader, num_epochs=50, lr=1e-4, patience=9):
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

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            docking_scores = batch["docking_score"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, docking_scores)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                docking_scores = batch["docking_score"].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, docking_scores)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        
        avg_val_loss = val_loss
        # scheduler.step(avg_val_loss)
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

model = train_molformer_regression(OverparamMoLFormerRegression(num_dense_layers=10, dropout_rate=0.5), train_loader, val_loader, lr = 1e-3, patience=10)


# %%
from scipy.stats import pearsonr

def plot_predictions(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    true_scores, pred_scores = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            docking_scores = batch["docking_score"].cpu().numpy()

            outputs = model(input_ids, attention_mask).cpu().numpy()

            true_scores.extend(docking_scores)
            pred_scores.extend(outputs)

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
pred_scores[0]

# %%
true_scores = [
    -22.85, -37.78, -36.89, -26.65, -31.91, -21.6, -33.8, -32.1, -46.5, -32.16,
    -31.62, -42.65, -38.87, -32.7, -31.94, -34.07, -36.94, -35.45, -25.01, -28.74,
    -37.42, -35.07, -26.27, -49.66, -35.06, -25.21, -35.98, -29.62, -31.41, -38.0,
    -21.57, -31.78
]

pred_scores = [
    -32.659695, -32.659695, -32.659695, -32.659695, -32.659695, -32.659695,
    -32.6597, -32.6597, -32.659695, -32.659695, -32.659695, -32.659695,
    -32.659695, -32.659695, -32.659695, -32.659695, -32.659695, -32.659695,
    -32.659695, -32.659695, -32.659695, -32.659695, -32.659695, -32.659695,
    -32.659695, -32.659695, -32.659695, -32.659695, -32.659695, -32.659695,
    -32.659695, -32.659695
]
pred_scores = np.round(pred_scores, 2)
# true_scores, pred_scores = [-35,-50], [-32,-32]
plt.figure(figsize=(8, 6))
plt.scatter(pred_scores, true_scores) #(true_scores, pred_scores)
plt.show()

# %%
device = torch.device('cuda')
for batch in tqdm(val_loader):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    docking_scores = batch["docking_score"].to(device)

    outputs = model(input_ids, attention_mask)
    print(docking_scores, outputs)
    break

# %% [markdown]
# # Pearson on true top1% mols in dataset
# 

# %%

true_top1pct_df

# %%
true_top1pct_df = merged_df[merged_df.mpro_dockscores<=threshold]
print(true_top1pct_df.shape)
true_top1pct_dataset =  SMILESDataset(true_top1pct_df, tokenizer)
true_top1pct_dataloader = DataLoader(true_top1pct_dataset, batch_size=32, shuffle=True)
plot_predictions(model, true_top1pct_dataloader)


# %% [markdown]
# 


