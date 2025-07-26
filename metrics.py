from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import numpy as np


def compute_enrichment(pred_scores, true_labels, top_fraction=0.01):
    """
    Compute enrichment factor at a given fraction of the screened library.
    
    Args:
        pred_scores (list or np.array): Predicted probabilities of being active.
        true_labels (list or np.array): Ground truth (1 for active, 0 for inactive).
        top_fraction (float): Fraction of the dataset to consider (e.g., 0.01 for top 1%).
        
    Returns:
        enrichment_factor (float): The computed enrichment factor.
    """
    # Convert to numpy arrays
    pred_scores = np.array(pred_scores)
    true_labels = np.array(true_labels)
    
    # Number of molecules to consider in top X%
    num_molecules = len(pred_scores)
    top_k = int(top_fraction * num_molecules)
    
    # Sort molecules by predicted score (descending)
    sorted_indices = np.argsort(-pred_scores)  # Negative sign for descending order
    top_k_indices = sorted_indices[:top_k]

    # Compute % of actives in the top X%
    actives_in_top_k = np.sum(true_labels[top_k_indices])  # Count actives
    percentage_actives_top_k = actives_in_top_k / top_k

    # Compute % of actives in the entire library
    percentage_actives_total = np.mean(true_labels)  # Mean gives fraction of actives

    # Compute enrichment factor
    enrichment_factor = percentage_actives_top_k / percentage_actives_total
    return enrichment_factor


def log_metrics(true_labels, predictions, al_iteration):
    f1 = f1_score(true_labels, predictions)
    auc = roc_auc_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    # enrichment_1 = compute_enrichment(pred_proba, true_labels)

    
    print(f"AL Iteration {al_iteration}: F1 Score: {f1}, AUC-ROC: {auc}, Precision: {precision}, Recall: {recall}")
    return {'f1':f1, 'auc':auc, 'precision':precision, 'recall':recall, 'tn':tn/len(true_labels),
            'fp':fp//len(true_labels), 'fn':fn/len(true_labels), 'tp':tp/len(true_labels)}


# # Example Usage:
# pred_scores = np.random.rand(2000000)  # Simulated scores for 2M molecules
# true_labels = np.random.choice([0, 1], size=2000000, p=[0.95, 0.05])  # 5% actives

# enrichment_1 = compute_enrichment(pred_scores, true_labels, top_fraction=0.01)  # EF at top 1%
# print(f"Enrichment Factor (Top 1%): {enrichment_1:.2f}")

