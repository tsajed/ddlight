import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import r2_score
from metrics import log_metrics
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.utils.class_weight import compute_class_weight
import joblib
import torch.distributed as dist
import wandb 
# wandb.login()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# def train_docking_model(model, train_features, train_labels, val_features, val_labels, num_epochs=40, batch_size=32, lr=0.001):
#     """
#     Train the DockingModel using PyTorch.

#     :param model: PyTorch DockingModel instance.
#     :param train_features: Numpy array of training features.
#     :param train_labels: Numpy array of training labels.
#     :param val_features: Numpy array of validation features.
#     :param val_labels: Numpy array of validation labels.
#     :param num_epochs: Number of training epochs.
#     :param batch_size: Batch size.
#     :param lr: Learning rate.
#     """
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     # Convert data to tensors
#     train_features = torch.tensor(train_features, dtype=torch.float32).to(device)
#     train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
#     val_features = torch.tensor(val_features, dtype=torch.float32).to(device)
#     val_labels = torch.tensor(val_labels, dtype=torch.long).to(device)

#     train_dataset = TensorDataset(train_features, train_labels)
#     val_dataset = TensorDataset(val_features, val_labels)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         for batch_features, batch_labels in train_loader:
#             batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(batch_features)
#             loss = criterion(outputs, batch_labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f}")

#         # Validation
#         model.eval()
#         val_correct, total_val_loss = 0, 0 
#         with torch.no_grad():
#             for batch_features, batch_labels in val_loader:
#                 batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
#                 outputs = model(batch_features)
#                 loss = criterion(outputs, batch_labels)
#                 total_val_loss += loss.item()
#                 _, predicted = torch.max(outputs, 1)
#                 val_correct += (predicted == batch_labels).sum().item()

#         val_accuracy = val_correct / len(val_dataset)
#         print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Accuracy: {val_accuracy:.4f}")
#         torch.cuda.empty_cache()
#     return model, total_val_loss

#train-time augmentation, label smoothing, and WeightedCrossEntropyLoss
# def train_docking_model(
#     model, train_features, train_labels, val_features, val_labels, num_epochs=40, batch_size=32, lr=0.001
# ):
#     """
#     Train the DockingModel using PyTorch with train-time augmentation, label smoothing, 
#     and WeightedCrossEntropyLoss.

#     :param model: PyTorch DockingModel instance.
#     :param train_features: Numpy array of training features.
#     :param train_labels: Numpy array of training labels.
#     :param val_features: Numpy array of validation features.
#     :param val_labels: Numpy array of validation labels.
#     :param num_epochs: Number of training epochs.
#     :param batch_size: Batch size.
#     :param lr: Learning rate.
#     """
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

#     train_features = torch.tensor(train_features, dtype=torch.float32).to(device)
#     train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
#     val_features = torch.tensor(val_features, dtype=torch.float32).to(device)
#     val_labels = torch.tensor(val_labels, dtype=torch.long).to(device)

#     train_dataset = TensorDataset(train_features, train_labels)
#     val_dataset = TensorDataset(val_features, val_labels)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     # WeightedCrossEntropyLoss to handle class imbalance
#     class_weights = torch.tensor([1.0, 10000.0]).to(device)  
#     criterion = nn.CrossEntropyLoss(weight=class_weights)

#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0

#         for batch_features, batch_labels in train_loader:
#             batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

#             # Train-time augmentation: Add Gaussian noise to features
#             noise = torch.normal(0, 0.01, size=batch_features.size()).to(device)
#             augmented_features = batch_features + noise

#             # Apply label smoothing
#             smooth_factor = 0.1
#             smoothed_labels = (
#                 1 - smooth_factor
#             ) * F.one_hot(batch_labels, num_classes=2).float() + smooth_factor / 2

#             optimizer.zero_grad()
#             outputs = model(augmented_features)

#             # Compute loss
#             loss = -torch.sum(smoothed_labels * F.log_softmax(outputs, dim=1), dim=1).mean()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f}")

#         # Validation
#         model.eval()
#         val_correct, total_val_loss = 0, 0
#         with torch.no_grad():
#             for batch_features, batch_labels in val_loader:
#                 batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
#                 outputs = model(batch_features)
#                 loss = criterion(outputs, batch_labels)
#                 total_val_loss += loss.item()
#                 _, predicted = torch.max(outputs, 1)
#                 val_correct += (predicted == batch_labels).sum().item()

#         val_accuracy = val_correct / len(val_dataset)
#         print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Accuracy: {val_accuracy:.4f}")
#         torch.cuda.empty_cache()

#     return model, total_val_loss

#train-time augmentation, label smoothing, and Focal loss
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        :param alpha: Weighting factor for the rare class (float).
        :param gamma: Focusing parameter to adjust the rate at which easy examples are down-weighted (float).
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Compute the focal loss.

        :param logits: Logits from the model (before softmax).
        :param targets: Ground-truth labels.
        :return: Focal loss.
        """
        probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        
        pt = torch.sum(targets_one_hot * probs, dim=1)  # Get probabilities corresponding to the target class
        log_pt = torch.log(pt + 1e-6)  # Avoid log(0)

        focal_loss = -self.alpha * ((1 - pt) ** self.gamma) * log_pt  # Compute focal loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

# def train_docking_model(
#     model, train_features, train_labels, val_features, val_labels, num_epochs=40, batch_size=32, lr=0.001, config=None
# ):
#     """
#     Train the DockingModel using PyTorch with train-time augmentation, label smoothing, 
#     and FocalLoss.

#     :param model: PyTorch DockingModel instance.
#     :param train_features: Numpy array of training features.
#     :param train_labels: Numpy array of training labels.
#     :param val_features: Numpy array of validation features.
#     :param val_labels: Numpy array of validation labels.
#     :param num_epochs: Number of training epochs.
#     :param batch_size: Batch size.
#     :param lr: Learning rate.
#     """
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

#     # Convert data to tensors
#     train_features = torch.tensor(train_features, dtype=torch.float32).to(device)
#     train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
#     val_features = torch.tensor(val_features, dtype=torch.float32).to(device)
#     val_labels = torch.tensor(val_labels, dtype=torch.long).to(device)

#     train_dataset = TensorDataset(train_features, train_labels)
#     val_dataset = TensorDataset(val_features, val_labels)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     # Use Focal Loss
#     criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     # Train loop
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0

#         for batch_features, batch_labels in train_loader:
#             batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

#             # Train-time augmentation: Add Gaussian noise to features
#             noise = torch.normal(0, 0.01, size=batch_features.size()).to(device)
#             augmented_features = batch_features + noise

#             # Apply label smoothing
#             smooth_factor = 0.1
#             smoothed_labels = (
#                 1 - smooth_factor
#             ) * F.one_hot(batch_labels, num_classes=2).float() + smooth_factor / 2

#             optimizer.zero_grad()
#             outputs = model(augmented_features)

#             # Compute loss with Focal Loss
#             loss = criterion(outputs, batch_labels) 
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f}")

#         # Validation
#         model.eval()
#         val_correct, total_val_loss = 0, 0
#         with torch.no_grad():
#             for batch_features, batch_labels in val_loader:
#                 batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
#                 outputs = model(batch_features)
#                 loss = criterion(outputs, batch_labels)
#                 total_val_loss += loss.item()
#                 _, predicted = torch.max(outputs, 1)
#                 val_correct += (predicted == batch_labels).sum().item()

#         val_accuracy = val_correct / len(val_dataset)
#         print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {total_val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")
#         torch.cuda.empty_cache()

#     return model, total_val_loss

# def train_docking_model(
#     model, train_loader, val_loader, num_epochs=40, lr=0.001, 
#     weight_decay=0.01, model_save_path='/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/',
#     config=None, patience = 5
# ):
#     """
#     Train the DockingModel using PyTorch with train-time augmentation, label smoothing, 
#     and FocalLoss.

#     :param model: PyTorch DockingModel instance.
#     :param train_loader: DataLoader for training data.
#     :param val_loader: DataLoader for validation data.
#     :param num_epochs: Number of training epochs.
#     :param lr: Learning rate.
#     :param weight_decay: Weight decay for the optimizer.
#     :param model_save_path: Path to save the trained model.
#     :param config: Configuration dictionary for logging and other settings.
#     """
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
#     total_val_loss, val_metrics = None, None

#     # Use Focal Loss for classification tasks
#     criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

#     with wandb.init(project='DDS', config=config, mode=None):
#         print(wandb.run.id, wandb.run.name)
#         info_vals = {}

#         for epoch in range(num_epochs):
#             # print('epoch ', epoch)
#             model.train()
#             total_loss = 0

#             for batch_features, batch_labels in tqdm(train_loader):
#                 batch_features, batch_labels = batch_features.to(device), batch_labels.long().to(device)

#                 # Train-time augmentation: Add Gaussian noise to features
#                 noise = torch.normal(0, 0.01, size=batch_features.size()).to(device)
#                 augmented_features = batch_features + noise

#                 # Apply label smoothing
#                 smooth_factor = 0.1
#                 smoothed_labels = (
#                     1 - smooth_factor
#                 ) * F.one_hot(batch_labels, num_classes=2).float() + smooth_factor / 2

#                 optimizer.zero_grad()
#                 outputs = model(augmented_features)

#                 # Compute loss
#                 loss = criterion(outputs, batch_labels)
#                 loss.backward()
#                 optimizer.step()

#                 total_loss += loss.item()

#             scheduler.step()
#             print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f}")
#             info_vals.update({'epoch': epoch, 'Train Loss': total_loss / len(train_loader)})

#             # Validation
#             model.eval()
#             total_val_loss = 0
#             val_correct = 0
#             val_labels, pred_val_docking_labels = [], []

#             with torch.no_grad():
#                 for batch_features, batch_labels in tqdm(val_loader):
#                     batch_features, batch_labels = batch_features.to(device), batch_labels.long().to(device)
#                     val_labels.extend(batch_labels.cpu().detach().numpy())

#                     outputs = model(batch_features)
#                     loss = criterion(outputs, batch_labels)
#                     total_val_loss += loss.item()

#                     _, predicted = torch.max(outputs, 1)
#                     pred_val_docking_labels.extend(predicted.cpu().detach().numpy())
#                     val_correct += (predicted == batch_labels).sum().item()

#             val_accuracy = val_correct / len(val_loader.dataset)
#             print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {total_val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")
#             info_vals.update({'Val Loss': total_val_loss / len(val_loader), 'Val Accuracy': val_accuracy})

#             # print('True val_docking_labels:', val_labels)
#             # print('Predicted val_docking_labels:', pred_val_docking_labels)

#             val_metrics = log_metrics(val_labels, pred_val_docking_labels, 0)
#             info_vals.update(val_metrics)
#             wandb.log(info_vals)

#             torch.cuda.empty_cache()
            
#             # print('model_train.py train_dock_model() True val_docking_labels:', val_labels)
#             # print('model_train.py train_dock_model() Predicted val_docking_labels:', pred_val_docking_labels)
#             torch.save(model.state_dict(), f'{model_save_path}/{config.global_params.model_architecture}_epoch_{epoch + 1}_val_loss_{total_val_loss:.4f}.pt')
#             print(f"Model saved to {model_save_path}")
#     return model, total_val_loss, val_metrics

# def train_docking_model(
#     model, train_loader, val_loader, al_iteration, num_epochs=40, lr=0.001, 
#     weight_decay=0.01, model_save_path='/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/',
#     config=None, patience=5, rank =0
# ):
#     """
#     Train the DockingModel using PyTorch with train-time augmentation, label smoothing, 
#     and FocalLoss, including early stopping.

#     :param model: PyTorch DockingModel instance.
#     :param train_loader: DataLoader for training data.
#     :param val_loader: DataLoader for validation data.
#     :param num_epochs: Number of training epochs.
#     :param lr: Learning rate.
#     :param weight_decay: Weight decay for the optimizer.
#     :param model_save_path: Path to save the trained model.
#     :param config: Configuration dictionary for logging and other settings.
#     :param patience: Number of epochs to wait for improvement in validation loss before stopping.
#     """
#     device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

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

#     # Early stopping parameters
#     best_val_loss = float("inf")
#     best_epoch = -1
#     early_stop_counter = 0
#     best_model_state = None

#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0

#         for batch_features, batch_labels in tqdm(train_loader):
#             batch_features, batch_labels = batch_features.to(device), batch_labels.long().to(device)

#             # Train-time augmentation: Add Gaussian noise to features
#             noise = torch.normal(0, 0.01, size=batch_features.size()).to(device)
#             augmented_features = batch_features + noise

#             # Apply label smoothing
#             smooth_factor = 0.1
#             smoothed_labels = (
#                 1 - smooth_factor
#             ) * F.one_hot(batch_labels, num_classes=2).float() + smooth_factor / 2

#             optimizer.zero_grad()
#             outputs = model(augmented_features)

#             # Compute loss
#             loss = criterion(outputs, batch_labels)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         scheduler.step()
#         print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f}")

#         # Validation
#         model.eval()
#         total_val_loss = 0
#         val_correct = 0
#         val_labels, pred_val_docking_labels = [], []

#         with torch.no_grad():
#             for batch_features, batch_labels in tqdm(val_loader):
#                 batch_features, batch_labels = batch_features.to(device), batch_labels.long().to(device)
#                 val_labels.extend(batch_labels.cpu().detach().numpy())

#                 outputs = model(batch_features)
#                 loss = criterion(outputs, batch_labels)
#                 total_val_loss += loss.item()

#                 _, predicted = torch.max(outputs, 1)
#                 pred_val_docking_labels.extend(predicted.cpu().detach().numpy())
#                 val_correct += (predicted == batch_labels).sum().item()

#         val_accuracy = val_correct / len(val_loader.dataset)
#         avg_val_loss = total_val_loss / len(val_loader)
#         print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

#         val_metrics = log_metrics(val_labels, pred_val_docking_labels, al_iteration)

#         # Early stopping logic
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_epoch = epoch
#             early_stop_counter = 0
#             best_model_state = model.state_dict()
#             print(f"New best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
#         else:
#             early_stop_counter += 1
#             print(f"No improvement in validation loss for {early_stop_counter} consecutive epochs.")

#         if early_stop_counter >= patience:
#             print(f"Early stopping triggered at epoch {epoch + 1}. Best epoch was {best_epoch + 1}.")
#             break

#         torch.cuda.empty_cache()

#     # Save the best model
#     model_save_name = f'{model_save_path}/{config.global_params.model_architecture}_best_model_val_loss_{best_val_loss:.4f}.pt'
#     torch.save(best_model_state, model_save_name)
#     print(f"Best model saved to {model_save_name}")

#     # Load the best model before returning
#     model.load_state_dict(best_model_state)

#     return model, best_val_loss, val_metrics


def train_docking_model(
    model, train_loader, val_loader, al_iteration, num_epochs=40, lr=0.001, 
    weight_decay=0.01, model_save_path='/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/',
    config=None, patience=5, rank=0
):
    """
    Train the DockingModel using PyTorch with train-time augmentation, label smoothing, 
    and FocalLoss, including early stopping.
    """
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    if config.global_params.loss == "crossentropy":
        criterion = torch.nn.CrossEntropyLoss()
        print("Using CrossEntropy Loss")
    elif config.global_params.loss == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
        print("Using Focal Loss")
    else:
        raise ValueError(f"Unsupported loss type: {config.global_params.loss}")

    best_val_loss = float("inf")
    best_epoch = -1
    early_stop_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_features, batch_labels in tqdm(train_loader):
            batch_features, batch_labels = batch_features.to(device), batch_labels.long().to(device)

            noise = torch.normal(0, 0.01, size=batch_features.size()).to(device)
            augmented_features = batch_features + noise

            smooth_factor = 0.1
            smoothed_labels = (
                1 - smooth_factor
            ) * F.one_hot(batch_labels, num_classes=2).float() + smooth_factor / 2

            optimizer.zero_grad()
            outputs = model(augmented_features)

            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f}")

        # === Validation ===
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_labels = []
        pred_val_docking_labels = []
        predicted_proba = []

        with torch.no_grad():
            for batch_features, batch_labels in tqdm(val_loader):
                batch_features, batch_labels = batch_features.to(device), batch_labels.long().to(device)

                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                val_labels.extend(batch_labels.cpu().numpy())
                pred_val_docking_labels.extend(predicted.cpu().numpy())
                predicted_proba.extend(outputs.softmax(dim=1)[:, 1].cpu().numpy())  # probability for class 1
                val_correct += (predicted == batch_labels).sum().item()

        val_accuracy = val_correct / len(val_loader.dataset)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        val_metrics = log_metrics(val_labels, pred_val_docking_labels, al_iteration)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            early_stop_counter = 0
            best_model_state = model.state_dict()
            print(f"New best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss for {early_stop_counter} consecutive epochs.")

        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}. Best epoch was {best_epoch + 1}.")
            break

        torch.cuda.empty_cache()

    # Save and load best model
    model_save_name = f'{model_save_path}/{config.global_params.model_architecture}_best_model_val_loss_{best_val_loss:.4f}.pt'
    torch.save(best_model_state, model_save_name)
    print(f"Best model saved to {model_save_name}")
    model.load_state_dict(best_model_state)

    return model, best_val_loss, val_metrics, predicted_proba, val_labels



def train_random_forest_model(
    train_features, train_labels, val_features, val_labels, 
    n_estimators=100, max_depth=None, class_weight="balanced", 
    model_save_path='/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/',
    config=None
):
    """
    Train a Random Forest model with logging and evaluation.

    :param train_features: Training feature matrix.
    :param train_labels: Training labels.
    :param val_features: Validation feature matrix.
    :param val_labels: Validation labels.
    :param n_estimators: Number of trees in the Random Forest.
    :param max_depth: Maximum depth of the trees.
    :param class_weight: Weighting strategy for classes.
    :param model_save_path: Path to save the trained model.
    :param config: Configuration dictionary for logging and other settings.
    """
    # Initialize WandB for logging
    with wandb.init(project='DDS', config=config, mode=None):
        print(wandb.run.id, wandb.run.name)
        info_vals = {}

        # Compute class weights
        class_weights = compute_class_weight(
            class_weight, classes=np.array([0, 1]), y=train_labels
        )
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        # Initialize the Random Forest model
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weights_dict,
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        rf_model.fit(train_features, train_labels)
        print("Random Forest model training complete.")
        
        # Evaluate on training data
        train_preds = rf_model.predict(train_features)
        train_probs = rf_model.predict_proba(train_features)
        train_loss = log_loss(train_labels, train_probs)
        train_accuracy = accuracy_score(train_labels, train_preds)
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        info_vals.update({'Train Loss': train_loss, 'Train Accuracy': train_accuracy})

        # Evaluate on validation data
        val_preds = rf_model.predict(val_features)
        val_probs = rf_model.predict_proba(val_features)
        val_loss = log_loss(val_labels, val_probs)
        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        info_vals.update({'Val Loss': val_loss, 'Val Accuracy': val_accuracy})

        # Log metrics to WandB
        val_metrics = classification_report(val_labels, val_preds, output_dict=True)
        print("Validation Metrics:")
        print(classification_report(val_labels, val_preds))
        info_vals.update(val_metrics)
        wandb.log(info_vals)

        # Save the model
        model_path = f"{model_save_path}/{config.arch}_rf_model.pkl"
        joblib.dump(rf_model, model_path)
        print(f"Random Forest model saved to {model_path}")

    return rf_model

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def move_to_device(data, device, rank, world_size):
    return data.to(device) if world_size == 1 else data.to(rank)

def train_molformer_model(
    model, train_loader, val_loader,  num_epochs=40, lr=0.001,
    weight_decay=0.01, model_save_path='/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/',
    config=None, patience=5, rank=0, world_size=4
):
    """
    Train the MoLFormer model using PyTorch with train-time augmentation, label smoothing, 
    and FocalLoss.

    :param model: PyTorch MoLFormer instance.
    :param train_loader: DataLoader for training data.
    :param val_loader: DataLoader for validation data.
    :param num_epochs: Number of training epochs.
    :param lr: Learning rate.
    :param weight_decay: Weight decay for the optimizer.
    :param model_save_path: Path to save the trained model.
    :param config: Configuration dictionary for logging and other settings.
    """
    base_lr, max_lr = 1e-6, 1e-3
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # print('rank is ', rank)
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # torch.cuda.set_device(rank)
    # device = torch.device(f"cuda:{rank}")
    # model = model.to(rank)
    # model = DDP(model, device_ids=[rank], output_device=rank)
    device = next(model.parameters()).device #torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    # # model = model.to(device)
    # print('model train.py device from model ', next(model.parameters()).device)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    # scheduler  = optim.lr_scheduler.CyclicLR(
    #     optimizer,
    #     base_lr=base_lr,
    #     max_lr=max_lr,
    #     step_size_up=len(train_loader) // 2,  # Adjust based on dataset size
    #     mode="triangular2",  # Alternative: "exp_range", "triangular"
    #     cycle_momentum=False,  # Disable momentum updates for Adam
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, eps=1e-1)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=0.1, step_size_up=200)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min= 1e-8)
    if config.global_params.loss == "crossentropy":
        criterion = torch.nn.CrossEntropyLoss()
        print("Using CrossEntropy Loss")
    elif config.global_params.loss == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
        print("Using Focal Loss")
    else:
        raise ValueError(f"Unsupported loss type: {config.global_params.loss}")
    
     # Early stopping parameters
    best_val_loss = float("inf")
    best_epoch = -1
    early_stop_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            input_ids = move_to_device(batch["input_ids"], device, rank, world_size)
            attention_mask = move_to_device(batch["attention_mask"], device, rank, world_size)
            labels = move_to_device(batch["labels"], device, rank, world_size)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)

            # Compute loss
            loss = criterion(outputs, labels)
            total_train_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            # scheduler.step() #if cyclic

        avg_train_loss = total_train_loss / len(train_loader)
        if rank==0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
        

        # Validation loop
        model.eval()
        val_losses = []
        predicted_proba = []
        all_labels = []
        predicted_labels_list = []
        correct, total = 0, 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                input_ids = move_to_device(batch["input_ids"], device, rank, world_size)
                attention_mask = move_to_device(batch["attention_mask"], device, rank, world_size)
                labels = move_to_device(batch["labels"], device, rank, world_size)

                # Forward pass
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

                _, predicted_labels = torch.max(outputs.data, 1)
                predicted_labels_list.extend(predicted_labels.cpu().detach().numpy())
                correct += (predicted_labels == labels).sum().item()
                total += labels.size(0)

                predicted_proba.extend(outputs.flatten().cpu().detach().numpy())
                all_labels.extend(labels.flatten().cpu().detach().numpy())

        avg_val_loss = sum(val_losses) / len(val_losses)
        scheduler.step(avg_val_loss) # if step

        val_accuracy = correct / total
        if rank==0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # print('True validation labels:', all_labels)
        # print('Predicted validation labels:', predicted_labels_list)

        # Compute additional metrics and log them
        val_metrics = log_metrics(all_labels, predicted_labels_list, 0)

        # Early stopping logic
        import copy
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            early_stop_counter = 0
            # best_model_state = model.state_dict()
            best_model_state = copy.deepcopy(model.state_dict())  # Deep copy to avoid overwriting
            print(f"New best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss for {early_stop_counter} consecutive epochs.")

        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}. Best epoch was {best_epoch + 1}.")
            break

        # Clear GPU memory
        torch.cuda.empty_cache()

    # # Save the best model
    # model_save_name = f'{model_save_path}/{config.global_params.model_architecture}_best_model.pt' #_val_loss_{best_val_loss:.4f}.pt'
    # # print('model_train.py model save name ',model_save_name)
    # torch.save(best_model_state, model_save_name)
    # print(f"Best model saved to {model_save_name}")

    # Load the best model before returning
    model.load_state_dict(best_model_state)
    # dist.destroy_process_group()

    return model, best_val_loss, val_metrics, predicted_proba, all_labels





def train_molformer_model_wandb(
    model, train_loader, val_loader, num_epochs=40, lr=0.001,
    weight_decay=0.01, model_save_path='/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/',
    config=None, patience=5, rank = 0
):
    """
    Train the MoLFormer model using PyTorch with train-time augmentation, label smoothing, 
    and FocalLoss.

    :param model: PyTorch MoLFormer instance.
    :param train_loader: DataLoader for training data.
    :param val_loader: DataLoader for validation data.
    :param num_epochs: Number of training epochs.
    :param lr: Learning rate.
    :param weight_decay: Weight decay for the optimizer.
    :param model_save_path: Path to save the trained model.
    :param config: Configuration dictionary for logging and other settings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    if config.global_params.loss == "crossentropy":
        criterion = torch.nn.CrossEntropyLoss()
        print("Using CrossEntropy Loss")
    elif config.global_params.loss == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
        print("Using Focal Loss")
    else:
        raise ValueError(f"Unsupported loss type: {config.global_params.loss}")
    
     # Early stopping parameters
    best_val_loss = float("inf")
    best_epoch = -1
    early_stop_counter = 0
    best_model_state = None

    with wandb.init(project='DDS', config=config, mode=None):
        print(wandb.run.id, wandb.run.name)
        info_vals = {}

        for epoch in range(num_epochs):
            # Training loop
            model.train()
            total_train_loss = 0
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                outputs = model(input_ids, attention_mask=attention_mask)

                # Compute loss
                loss = criterion(outputs, labels)
                total_train_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

            scheduler.step()
            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
            info_vals.update({'epoch': epoch, 'Train Loss': avg_train_loss})
            keys = sorted(info_vals.keys())
            all_info_vals = torch.zeros(len(keys)).to(rank)
            world_size = 4
            for i, k in enumerate(keys):
                try:
                    all_info_vals[i] = info_vals[k]
                except Exception as e:
                    print(e,k)
                    # raise e
            dist.all_reduce(all_info_vals, op=dist.ReduceOp.SUM)
            for i, k in enumerate(keys):
                info_vals[k] = all_info_vals[i].item() / world_size
            if rank == 0:
                wandb.log(info_vals)
            print('consolidated info_Vals ', info_vals)
            

            # Validation loop
            model.eval()
            val_losses = []
            predicted_proba = []
            all_labels = []
            predicted_labels_list = []
            correct, total = 0, 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    # Forward pass
                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)
                    val_losses.append(loss.item())

                    _, predicted_labels = torch.max(outputs.data, 1)
                    predicted_labels_list.extend(predicted_labels.cpu().detach().numpy())
                    correct += (predicted_labels == labels).sum().item()
                    total += labels.size(0)

                    predicted_proba.extend(outputs.flatten().cpu().detach().numpy())
                    all_labels.extend(labels.flatten().cpu().detach().numpy())

            avg_val_loss = sum(val_losses) / len(val_losses)
            val_accuracy = correct / total
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            info_vals.update({'Val Loss': avg_val_loss, 'Val Accuracy': val_accuracy})

            # print('True validation labels:', all_labels)
            # print('Predicted validation labels:', predicted_labels_list)

            # Compute additional metrics and log them
            val_metrics = log_metrics(all_labels, predicted_labels_list, 0)
            info_vals.update(val_metrics)
            wandb.log(info_vals)

            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                early_stop_counter = 0
                best_model_state = model.state_dict()
                print(f"New best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
            else:
                early_stop_counter += 1
                print(f"No improvement in validation loss for {early_stop_counter} consecutive epochs.")

            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best epoch was {best_epoch + 1}.")
                break

            # Clear GPU memory
            torch.cuda.empty_cache()

        # Save the best model
        model_save_name = f'{model_save_path}/{config.global_params.model_architecture}_best_model_val_loss_{best_val_loss:.4f}.pt'
        torch.save(best_model_state, model_save_name)
        print(f"Best model saved to {model_save_name}")

    return model, best_val_loss, val_metrics



# def predict_with_molformer(
#     model, dataloader, acquisition_function: str = "random", n_iter=3
# ):
#     """
#     Predict labels, probabilities, and uncertainty using the trained MoLFormer model
#     with support for various active learning acquisition functions.

#     :param model: Trained PyTorch MoLFormer instance.
#     :param dataloader: DataLoader for the dataset to predict.
#     :param acquisition_function: Acquisition function for prioritizing samples.
#                                  Options: "random", "entropy", "least_confidence", 
#                                  "margin", "greedy", "ucb", "unc".
#     :param n_iter: Number of Monte Carlo iterations for uncertainty estimation.
#     :return: predicted_labels, predicted_proba, uncertainty_scores
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()

#     all_predicted_labels = []
#     all_predicted_proba = []
#     all_uncertainty_scores = []

#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Predicting"):
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)

#             # Use MC Dropout for uncertainty-based acquisition functions
#             if acquisition_function in {"ucb", "unc"}:
#                 mean_pred, std_dev = model.mc_dropout_forward(input_ids, attention_mask=attention_mask, n_iter=n_iter)
#                 probabilities = torch.softmax(mean_pred, dim=1).cpu().numpy()
#             else:
#                 logits = model(input_ids, attention_mask=attention_mask)
#                 probabilities = torch.softmax(logits, dim=1).cpu().numpy()
#                 mean_pred, std_dev = logits, None

#             predicted_proba = probabilities[:, 1]
#             predicted_labels = np.argmax(probabilities, axis=1)

#             # Compute uncertainty scores
#             if acquisition_function == "entropy":
#                 uncertainty = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
#             elif acquisition_function == "least_confidence":
#                 uncertainty = 1 - np.max(probabilities, axis=1)
#             elif acquisition_function == "margin":
#                 sorted_probs = np.sort(probabilities, axis=1)
#                 uncertainty = sorted_probs[:, -1] - sorted_probs[:, -2]
#             elif acquisition_function == "random":
#                 uncertainty = np.random.rand(input_ids.size(0))
#             elif acquisition_function == "greedy":
#                 uncertainty = torch.softmax(mean_pred, dim=1).cpu().numpy()[:, 1]
#             elif acquisition_function == "ucb":
#                 uncertainty = mean_pred[:, 1].cpu().numpy() + 2 * std_dev[:, 1].cpu().numpy()
#             elif acquisition_function == "unc":
#                 uncertainty = std_dev[:, 1].cpu().numpy()
#             else:
#                 raise ValueError(f"Unknown acquisition function: {acquisition_function}")

#             # Append batch predictions to results
#             all_predicted_labels.extend(predicted_labels)
#             all_predicted_proba.extend(predicted_proba)
#             all_uncertainty_scores.extend(uncertainty)

#     torch.cuda.empty_cache()

#     return (
#         np.array(all_predicted_labels),
#         np.array(all_predicted_proba),
#         np.array(all_uncertainty_scores),
#     )



import torch
import numpy as np
from tqdm import tqdm

def predict_with_molformer(
    model, dataloader, acquisition_function: str = "random", n_iter=3, rank =0
):
    """
    Predict labels, probabilities, and uncertainty using the trained MoLFormer model
    with support for various active learning acquisition functions, including BALD.

    :param model: Trained PyTorch MoLFormer instance.
    :param dataloader: DataLoader for the dataset to predict.
    :param acquisition_function: Acquisition function for prioritizing samples.
                                 Options: "random", "entropy", "least_confidence", 
                                          "margin", "greedy", "ucb", "unc", "bald".
    :param n_iter: Number of Monte Carlo iterations for uncertainty estimation.
    :return: predicted_labels, predicted_proba, uncertainty_scores
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    model.eval()

    all_predicted_labels = []
    all_predicted_proba = []
    all_uncertainty_scores = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(rank)
            attention_mask = batch["attention_mask"].to(rank)

            if acquisition_function in {"ucb", "unc", "bald"}:
                mc_predictions = []
                for _ in range(n_iter):
                    model.train()
                    logits = model(input_ids, attention_mask=attention_mask)
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                    mc_predictions.append(probabilities)
                
                mc_predictions = np.array(mc_predictions)  # Shape: (n_iter, batch_size, num_classes)
                mean_pred = np.mean(mc_predictions, axis=0)  # Expected probability over MC samples
                std_dev = np.std(mc_predictions, axis=0)  # Standard deviation for UCB/uncertainty
            else:
                logits = model(input_ids, attention_mask=attention_mask)
                mean_pred = torch.softmax(logits, dim=1).cpu().numpy()
                std_dev = None

            predicted_proba = mean_pred[:, 1]
            predicted_labels = np.argmax(mean_pred, axis=1)

            # Compute uncertainty scores
            if acquisition_function == "entropy":
                uncertainty = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=1)
            elif acquisition_function == "least_confidence":
                uncertainty = 1 - np.max(mean_pred, axis=1)
            elif acquisition_function == "margin":
                sorted_probs = np.sort(mean_pred, axis=1)
                uncertainty = sorted_probs[:, -1] - sorted_probs[:, -2]
            elif acquisition_function == "random":
                uncertainty = np.random.rand(input_ids.size(0))
            elif acquisition_function == "greedy":
                uncertainty = mean_pred[:, 1]
            elif acquisition_function == "ucb":
                uncertainty = mean_pred[:, 1] + 2 * std_dev[:, 1]
            elif acquisition_function == "unc":
                uncertainty = std_dev[:, 1]
            elif acquisition_function == "bald":
                # Compute entropy of the mean prediction
                expected_entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=1)
                
                # Compute expected entropy across MC samples
                entropy_per_sample = -np.sum(mc_predictions * np.log(mc_predictions + 1e-10), axis=2)  # Shape: (n_iter, batch_size)
                mean_entropy = np.mean(entropy_per_sample, axis=0)  # Average over MC samples
                
                # BALD score: Mutual Information
                uncertainty = expected_entropy - mean_entropy
            else:
                raise ValueError(f"Unknown acquisition function: {acquisition_function}")

            # Append batch predictions to results
            all_predicted_labels.extend(predicted_labels)
            all_predicted_proba.extend(predicted_proba)
            all_uncertainty_scores.extend(uncertainty)

    torch.cuda.empty_cache()

    return (
        np.array(all_predicted_labels),
        np.array(all_predicted_proba),
        np.array(all_uncertainty_scores),
    )


import numpy as np
import torch

# def predict_with_model(
#     model, features, acquisition_function: str = "random", n_iter=3):
#     """
#     Predict labels, probabilities, and uncertainty using the trained PyTorch model.
#     Incorporates support for various active learning acquisition functions.

#     :param model: Trained PyTorch DockingModel.
#     :param features: Numpy array of features.
#     :param acquisition_function: Acquisition function for prioritizing samples.
#                                  Options: "random", "entropy", "least_confidence", 
#                                  "margin", "greedy", "ucb", "unc".
#     :param top_k: Number of top uncertain samples to select. If None, return all samples.
#     :param n_iter: Number of Monte Carlo iterations for uncertainty estimation.
#     """
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()

#     with torch.no_grad():
#         features_tensor = torch.tensor(features, dtype=torch.float32).to(device)

#         # Use MC Dropout for uncertainty-based acquisition functions
#         if acquisition_function in {"ucb", "unc"}:
#             mean_pred, std_dev = model.mc_dropout_forward(features_tensor, n_iter=n_iter)
#             probabilities = torch.softmax(mean_pred, dim=1).cpu().detach().numpy()
#         else:
#             logits = model(features_tensor)
#             probabilities = torch.softmax(logits, dim=1).cpu().detach().numpy()
#             mean_pred, std_dev = logits, None

#         predicted_proba = probabilities[:, 1]
#         predicted_labels = np.argmax(probabilities, axis=1)

#         # Compute uncertainty scores
#         if acquisition_function == "entropy":
#             uncertainty = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
#         elif acquisition_function == "least_confidence":
#             uncertainty = 1 - np.max(probabilities, axis=1)
#         elif acquisition_function == "margin":
#             sorted_probs = np.sort(probabilities, axis=1)
#             uncertainty = sorted_probs[:, -1] - sorted_probs[:, -2]
#         elif acquisition_function == "random":
#             uncertainty = np.random.rand(features.shape[0])
#         elif acquisition_function == "greedy":
#             uncertainty = mean_pred[:, 1].cpu().numpy()  # Use predicted mean
#         elif acquisition_function == "ucb":
#             uncertainty = mean_pred[:, 1].cpu().numpy() + 2 * std_dev[:, 1].cpu().numpy()
#         elif acquisition_function == "unc":
#             uncertainty = std_dev[:, 1].cpu().numpy()
#         else:
#             raise ValueError(f"Unknown acquisition function: {acquisition_function}")

#     torch.cuda.empty_cache()

#     return  predicted_labels, predicted_proba, uncertainty

# def predict_with_model(
#     model, dataloader, acquisition_function: str = "random", n_iter=3
# ):
#     #TODO return true labels as well to ensure robustness of metric computation
#     """
#     Predict labels, probabilities, and uncertainty using the trained PyTorch model
#     with support for various active learning acquisition functions and DataLoader-based batching.

#     :param model: Trained PyTorch DockingModel.
#     :param dataloader: DataLoader for the dataset to predict.
#     :param acquisition_function: Acquisition function for prioritizing samples.
#                                  Options: "random", "entropy", "least_confidence", 
#                                  "margin", "greedy", "ucb", "unc".
#     :param n_iter: Number of Monte Carlo iterations for uncertainty estimation.
#     :return: predicted_labels, predicted_proba, uncertainty_scores
#     """
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()

#     all_predicted_labels = []
#     all_predicted_proba = []
#     all_uncertainty_scores = []

#     with torch.no_grad():
#         for batch_features, batch_labels in tqdm(dataloader, desc="Predicting"):
#             batch_features = batch_features.to(device)

#             # Use MC Dropout for uncertainty-based acquisition functions
#             if acquisition_function in {"ucb", "unc"}:
#                 mean_pred, std_dev = model.mc_dropout_forward(batch_features, n_iter=n_iter)
#                 probabilities = torch.softmax(mean_pred, dim=1).cpu().numpy()
#             else:
#                 logits = model(batch_features)
#                 probabilities = torch.softmax(logits, dim=1).cpu().numpy()
#                 mean_pred, std_dev = logits, None

#             predicted_proba = probabilities[:, 1]
#             predicted_labels = np.argmax(probabilities, axis=1)

#             # Compute uncertainty scores
#             if acquisition_function == "entropy":
#                 uncertainty = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
#             elif acquisition_function == "least_confidence":
#                 uncertainty = 1 - np.max(probabilities, axis=1)
#             elif acquisition_function == "margin":
#                 sorted_probs = np.sort(probabilities, axis=1)
#                 uncertainty = sorted_probs[:, -1] - sorted_probs[:, -2]
#             elif acquisition_function == "random":
#                 uncertainty = np.random.rand(batch_features.size(0))
#             elif acquisition_function == "greedy":
#                 uncertainty = torch.softmax(mean_pred, dim=1).cpu().numpy()[:, 1]    #mean_pred[:, 1].cpu().numpy()  # Use predicted mean
#             elif acquisition_function == "ucb":
#                 uncertainty = mean_pred[:, 1].cpu().numpy() + 2 * std_dev[:, 1].cpu().numpy()
#             elif acquisition_function == "unc":
#                 uncertainty = std_dev[:, 1].cpu().numpy()
#             else:
#                 raise ValueError(f"Unknown acquisition function: {acquisition_function}")

#             # Append batch predictions to results
#             all_predicted_labels.extend(predicted_labels)
#             all_predicted_proba.extend(predicted_proba)
#             all_uncertainty_scores.extend(uncertainty)

#     torch.cuda.empty_cache()

#     return (
#         np.array(all_predicted_labels),
#         np.array(all_predicted_proba),
#         np.array(all_uncertainty_scores),
#     )


def predict_with_model(
    model, dataloader, acquisition_function: str = "random", n_iter=3
):
    # TODO: Return true labels as well to ensure robustness of metric computation
    """
    Predict labels, probabilities, and uncertainty using the trained PyTorch model
    with support for various active learning acquisition functions, including BALD.

    :param model: Trained PyTorch DockingModel.
    :param dataloader: DataLoader for the dataset to predict.
    :param acquisition_function: Acquisition function for prioritizing samples.
                                 Options: "random", "entropy", "least_confidence", 
                                          "margin", "greedy", "ucb", "unc", "bald".
    :param n_iter: Number of Monte Carlo iterations for uncertainty estimation.
    :return: predicted_labels, predicted_proba, uncertainty_scores
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_predicted_labels = []
    all_predicted_proba = []
    all_uncertainty_scores = []

    with torch.no_grad():
        for batch_features, batch_labels in tqdm(dataloader, desc="Predicting"):
            batch_features = batch_features.to(device)

            if acquisition_function in {"ucb", "unc", "bald"}:
                mc_predictions = []
                for _ in range(n_iter):
                    model.train()
                    logits = model(batch_features)
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                    mc_predictions.append(probabilities)

                mc_predictions = np.array(mc_predictions)  # Shape: (n_iter, batch_size, num_classes)
                mean_pred = np.mean(mc_predictions, axis=0)  # Expected probability over MC samples
                std_dev = np.std(mc_predictions, axis=0)  # Standard deviation for UCB/uncertainty

            else:
                logits = model(batch_features)
                mean_pred = torch.softmax(logits, dim=1).cpu().numpy()
                std_dev = None

            predicted_proba = mean_pred[:, 1]
            predicted_labels = np.argmax(mean_pred, axis=1)

            # Compute uncertainty scores
            if acquisition_function == "entropy":
                uncertainty = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=1)
            elif acquisition_function == "least_confidence":
                uncertainty = 1 - np.max(mean_pred, axis=1)
            elif acquisition_function == "margin":
                sorted_probs = np.sort(mean_pred, axis=1)
                uncertainty = sorted_probs[:, -1] - sorted_probs[:, -2]
            elif acquisition_function == "random":
                uncertainty = np.random.rand(batch_features.size(0))
            elif acquisition_function == "greedy":
                uncertainty = mean_pred[:, 1]  # Use predicted mean
            elif acquisition_function == "ucb":
                uncertainty = mean_pred[:, 1] + 2 * std_dev[:, 1]
            elif acquisition_function == "unc":
                uncertainty = std_dev[:, 1]
            elif acquisition_function == "bald":
                # Compute entropy of the mean prediction
                expected_entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=1)
                
                # Compute expected entropy across MC samples
                entropy_per_sample = -np.sum(mc_predictions * np.log(mc_predictions + 1e-10), axis=2)  # Shape: (n_iter, batch_size)
                mean_entropy = np.mean(entropy_per_sample, axis=0)  # Average over MC samples
                
                # BALD score: Mutual Information
                uncertainty = expected_entropy - mean_entropy
            else:
                raise ValueError(f"Unknown acquisition function: {acquisition_function}")

            # Append batch predictions to results
            all_predicted_labels.extend(predicted_labels)
            all_predicted_proba.extend(predicted_proba)
            all_uncertainty_scores.extend(uncertainty)

    torch.cuda.empty_cache()

    return (
        np.array(all_predicted_labels),
        np.array(all_predicted_proba),
        np.array(all_uncertainty_scores),
    )


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

# def train_graph_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, 
#                       weight_decay=0.01, model_save_path = '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/'):
#     """
#     Train the DockingModel using PyTorch.

#     :param model: PyTorch DockingModel instance.
#     :param train_loader: DataLoader for training data.
#     :param val_loader: DataLoader for validation data.
#     :param num_epochs: Number of training epochs.
#     :param lr: Learning rate.
#     :param weight_decay: Weight decay for the optimizer.
#     """
#     device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    
#     criterion = nn.MSELoss()  # Assuming regression task; change if necessary

#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         y_list, pred_list = [], []
        
#         for batch in tqdm(train_loader):
#             graph, y = batch[0].to(device), batch[1].float().to(device)
#             feat = graph.ndata['h'].to(device)
            
#             optimizer.zero_grad()
#             pred, _ = model(graph, feat, training=True)
#             loss = criterion(pred[:, 0], y)
            
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#             y_list.append(y)
#             pred_list.append(pred[:, 0].detach().cpu().numpy())

#         scheduler.step()
        
#         print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f}")
        
#         # Validation
#         model.eval()
#         val_loss = 0
#         y_val_list, pred_val_list = [], []
        
#         with torch.no_grad():
#             for batch in val_loader:
#                 graph, y = batch[0].to(device), batch[1].float().to(device)
#                 feat = graph.ndata['h'].to(device)
                
#                 pred, _ = model(graph, feat, training=False)
#                 loss = criterion(pred[:, 0], y)
#                 val_loss += loss.item()
                
#                 y_val_list.append(y)
#                 pred_val_list.append(pred[:, 0].cpu().numpy())
        
#         print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss / len(val_loader):.4f}")
#         torch.cuda.empty_cache()
    
#     torch.save(model.state_dict(), f'{model_save_path}/gine_{epoch}_{val_loss}.pt')
#     print(f"Model saved to {model_save_path}")
#     return model

def train_graph_model(
    model, train_loader, val_loader, num_epochs=10, lr=0.001, 
    weight_decay=0.01, model_save_path='/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/',
    config =None
):
    """
    Train the DockingModel using PyTorch with train-time augmentation, label smoothing, 
    and FocalLoss.

    :param model: PyTorch DockingModel instance.
    :param train_loader: DataLoader for training data.
    :param val_loader: DataLoader for validation data.
    :param num_epochs: Number of training epochs.
    :param lr: Learning rate.
    :param weight_decay: Weight decay for the optimizer.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    
    # Use Focal Loss for classification tasks
    # criterion = nn.CrossEntropyLoss() #
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
    with wandb.init(project = 'DDS', config = config, mode = 'disabled'):
        print(wandb.run.id, wandb.run.name)
        info_vals = {}
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader):
                graph, y = batch[0].to(device), batch[1].long().to(device)  # Assume classification task
                feat = graph.ndata['h'].to(device)

                # Train-time augmentation: Add Gaussian noise to features
                noise = torch.normal(0, 0.01, size=feat.size()).to(device)
                augmented_feat = feat + noise

                # Apply label smoothing
                smooth_factor = 0.1
                smoothed_labels = (
                    1 - smooth_factor
                ) * F.one_hot(y, num_classes=2).float() + smooth_factor / 2

                optimizer.zero_grad()
                pred, _ = model(graph, augmented_feat, training=True)

                loss = criterion(pred, y)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f}")
            info_vals.update({'epoch':epoch,
                'Train Loss':total_loss / len(train_loader)})

            
            # Validation
            model.eval()
            total_val_loss = 0
            val_correct = 0
            val_labels,pred_val_docking_labels = [],[]
            
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    graph, y = batch[0].to(device), batch[1].long().to(device)
                    val_labels.extend(y.cpu().detach().numpy())
                    feat = graph.ndata['h'].to(device)
                    
                    pred, _ = model(graph, feat, training=False)
                    loss = criterion(pred, y)
                    total_val_loss += loss.item()
                    
                    _, predicted = torch.max(pred, 1)
                    pred_val_docking_labels.extend(predicted.cpu().detach().numpy())
                    val_correct += (predicted == y).sum().item()

            val_accuracy = val_correct / len(val_loader.dataset)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {total_val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")
            info_vals.update({'Val Loss':total_val_loss / len(val_loader),
                              'Val Acc':val_accuracy})
            print('model_train.py True val_docking_labels:', val_labels)
            print('model_train.py Predicted val_docking_labels:', pred_val_docking_labels)
            val_metrics = log_metrics(val_labels, pred_val_docking_labels, 0)
            info_vals.update(val_metrics)
            wandb.log(info_vals)
           
            torch.cuda.empty_cache()
        
    torch.save(model.state_dict(), f'{model_save_path}/{config.arch}_epoch_{epoch + 1}_val_loss_{total_val_loss:.4f}.pt')
    print(f"Model saved to {model_save_path}")
    return model



# def predict_with_graph_model(model, test_loader, cutoff):
#     """
#     Predict labels and uncertainty using the trained PyTorch model, and compute R² for molecules with y_true < cutoff.

#     :param model: Trained PyTorch DockingModel.
#     :param test_loader: DataLoader for test features.
#     :param cutoff: Cutoff value for filtering y_true.
#     :return: y_true, y_pred, uncertainties
#     """
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()
    
#     y_true, y_pred, uncertainties = [], [], []

#     with torch.no_grad():
#         for batch in test_loader:
#             graph, y = batch[0].to(device), batch[1].float().to(device)
#             feat = graph.ndata['h'].to(device)
            
#             preds, _ = model(graph, feat, training=False)
#             preds = preds[:, 0].cpu().numpy()
            
#             y_true.append(y.cpu().numpy())
#             y_pred.append(preds)
#             uncertainties.append(-np.max(preds))  # Approximate uncertainty

#     y_true = np.concatenate(y_true)
#     y_pred = np.concatenate(y_pred)
#     uncertainties = np.array(uncertainties)
#     print('model_train.py below cutoff y_true ',y_true)
#     print('model_train.py below cutoff y_pred ',y_pred)
    
#     # Compute R² for y_true < cutoff
#     mask = y_true < cutoff
#     if np.any(mask):  # Ensure at least one value meets the condition
#         r2_cutoff = r2_score(y_true[mask], y_pred[mask])
#         print(f"R² for y_true < {cutoff}: {r2_cutoff:.4f}")
#     else:
#         print(f"No molecules found with y_true < {cutoff}")
#     torch.cuda.empty_cache()

#     return y_true, y_pred, uncertainties

def predict_with_graph_model(
    model, test_loader, acquisition_function: str = "random", n_iter=3
):
    """
    Predict labels, probabilities, and uncertainty using the trained PyTorch graph model.
    Incorporates support for various active learning acquisition functions.

    :param model: Trained PyTorch Graph Model.
    :param test_loader: DataLoader for test features.
    :param acquisition_function: Acquisition function for prioritizing samples.
                                 Options: "random", "entropy", "least_confidence", 
                                 "margin", "greedy", "ucb", "unc".
    :param n_iter: Number of Monte Carlo iterations for uncertainty estimation.
    :return: y_true, y_pred, uncertainties
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    y_true, y_pred, uncertainties = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            graph, y = batch[0].to(device), batch[1].float().to(device)
            feat = graph.ndata['h'].to(device)

            # Use MC Dropout for uncertainty-based acquisition functions
            if acquisition_function in {"ucb", "unc"}:
                mean_pred, std_dev = model.mc_dropout_forward(graph, feat, n_iter=n_iter)
                probabilities = torch.softmax(mean_pred, dim=1).cpu().numpy()
            else:
                logits, _ = model(graph, feat, training=False)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                mean_pred, std_dev = logits, None

            predicted_proba = probabilities[:, 1]
            predicted_labels = np.argmax(probabilities, axis=1)

            # Compute uncertainty scores
            if acquisition_function == "entropy":
                uncertainty = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
            elif acquisition_function == "least_confidence":
                uncertainty = 1 - np.max(probabilities, axis=1)
            elif acquisition_function == "margin":
                sorted_probs = np.sort(probabilities, axis=1)
                uncertainty = sorted_probs[:, -1] - sorted_probs[:, -2]
            elif acquisition_function == "random":
                uncertainty = np.random.rand(graph.batch_size)
            elif acquisition_function == "greedy":
                uncertainty = mean_pred[:, 1].cpu().numpy()  # Use predicted mean
            elif acquisition_function == "ucb":
                uncertainty = mean_pred[:, 1].cpu().numpy() + 2 * std_dev[:, 1].cpu().numpy()
            elif acquisition_function == "unc":
                uncertainty = std_dev[:, 1].cpu().numpy()
            else:
                raise ValueError(f"Unknown acquisition function: {acquisition_function}")

            y_true.append(y.cpu().numpy())
            y_pred.append(predicted_labels)
            uncertainties.append(uncertainty)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    uncertainties = np.concatenate(uncertainties)

    torch.cuda.empty_cache()

    return y_true, y_pred, uncertainties


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
def weighted_mse_loss(y_pred, y_true, weights):
    return torch.mean(weights * (y_pred - y_true) ** 2)

def train_weighted_graph_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, 
                               weight_decay=0.01, model_save_path= '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL'):
    """
    Train the DockingModel using PyTorch with a custom weighted MSE loss.

    :param model: PyTorch DockingModel instance.
    :param train_loader: DataLoader for training data.
    :param val_loader: DataLoader for validation data.
    :param num_epochs: Number of training epochs.
    :param lr: Learning rate.
    :param weight_decay: Weight decay for the optimizer.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            graph, y = batch[0].to(device), batch[1].float().to(device)
            feat = graph.ndata['h'].to(device)

            optimizer.zero_grad()
            pred, _ = model(graph, feat, training=True)

            # Calculate weights based on percentile
            percentiles = torch.argsort(torch.argsort(y)) / len(y)
            weights = torch.where(percentiles <= 0.01, 10.0, 1.0)  # 10x weight for 1%ile scores

            loss = weighted_mse_loss(pred[:, 0], y, weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                graph, y = batch[0].to(device), batch[1].float().to(device)
                feat = graph.ndata['h'].to(device)
                
                pred, _ = model(graph, feat, training=False)
                val_loss += nn.functional.mse_loss(pred[:, 0], y).item()
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss / len(val_loader):.4f}")
        torch.cuda.empty_cache()

    # Save the model after training
    torch.save(model.state_dict(), f'{model_save_path}/gine_{epoch}_{val_loss}.pt')
    print(f"Model saved to {model_save_path}")
    return model
