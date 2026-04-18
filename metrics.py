"""
metrics.py
----------
Centralised evaluation metrics for both project tasks.
All other modules import from here — never calculate metrics inline.

Why centralise?
  Consistency: everyone uses the same formula.
  Rubric compliance: Precision, Recall, F-measure are required for BOTH tasks.
  Reusability: one fix here fixes it everywhere.

Tasks covered:
  1. Node Classification → Precision, Recall, F1, Accuracy (per-class + macro)
  2. Link Prediction     → Precision, Recall, F1, AUC-ROC, Average Precision
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ──────────────────────────────────────────────────────────────
# TASK 1: NODE CLASSIFICATION
# ──────────────────────────────────────────────────────────────


def calculate_node_metrics(y_true, y_pred, model_name="Model", verbose=True):
    """
    Calculates evaluation metrics for node classification.

    How it works:
      - y_true: the ground truth labels  (e.g. [3, 1, 6, 2, ...])
      - y_pred: what the model predicted (e.g. [3, 0, 6, 2, ...])
      - We compare them element-by-element using sklearn.

    'macro' averaging = calculate metric per class, then take the
    unweighted mean. This treats all 7 Cora topics equally regardless
    of how many papers belong to each topic.

    Args:
        y_true     (array-like): Ground truth node labels.
        y_pred     (array-like): Predicted node labels from the model.
        model_name (str):        Display name for print output.
        verbose    (bool):       If True, print a formatted summary table.

    Returns:
        dict: {
            'accuracy':  float,
            'precision': float,  ← macro-averaged across all 7 classes
            'recall':    float,  ← macro-averaged across all 7 classes
            'f1':        float,  ← macro-averaged across all 7 classes
            'model':     str
        }
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    if verbose:
        print(f"\n{'='*50}")
        print(f"  NODE CLASSIFICATION METRICS — {model_name}")
        print(f"{'='*50}")
        print(f"  Accuracy:            {accuracy:.4f}  ({accuracy*100:.2f}%)")
        print(f"  Precision (macro):   {precision:.4f}")
        print(f"  Recall    (macro):   {recall:.4f}")
        print(f"  F1-Score  (macro):   {f1:.4f}")
        print(f"\n  Per-Class Breakdown:")
        print(classification_report(y_true, y_pred, zero_division=0))

    return {
        "model": model_name,
        "task": "node_classification",
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ──────────────────────────────────────────────────────────────
# TASK 2: LINK PREDICTION
# ──────────────────────────────────────────────────────────────


def calculate_link_metrics(
    y_true,
    y_scores,
    threshold=0.5,
    model_name="Model",
    verbose=True,
    k_values=(10, 50, 100),
):
    """
    Calculates evaluation metrics for link prediction.

    How it works:
      - y_true:   binary labels — 1 = real edge exists, 0 = fake edge
      - y_scores: raw probability scores output by the model (0.0 to 1.0)
      - We apply a threshold (default 0.5) to convert scores → predictions
        e.g. score 0.73 → predicted 1 (link exists)
             score 0.31 → predicted 0 (no link)

    Why AUC-ROC?
      It measures how well the model ranks real edges above fake ones,
      independent of the threshold. Standard metric for link prediction.

    Why Average Precision?
      Also threshold-independent. Better than AUC when the dataset has
      many more negative (fake) edges than positive (real) ones.

        Args:
        y_true    (array-like): Binary ground truth. 1=real edge, 0=fake.
        y_scores  (array-like): Model's predicted probability for each edge.
        threshold (float):      Score cutoff to call an edge real (default 0.5).
        model_name (str):       Display name for print output.
        verbose    (bool):      If True, print a formatted summary.
                k_values (tuple[int]):  K values used for ranking metrics.

        Ranking metrics:
            - prediction@k: precision among top-k scored candidate links.
            - hits@k:       recall-like coverage of true links captured in top-k.

    Returns:
        dict: {
            'precision': float,
            'recall':    float,
            'f1':        float,
            'auc_roc':   float,
            'avg_precision': float,
            'model':     str
        }
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_scores)
    avg_prec = average_precision_score(y_true, y_scores)

    ranking_metrics = {}
    sorted_idx = np.argsort(-y_scores)
    total_true = int(y_true.sum())
    for k in k_values:
        k_eff = min(int(k), int(y_true.size))
        if k_eff <= 0:
            continue
        topk_true = y_true[sorted_idx[:k_eff]]
        tp_at_k = int(topk_true.sum())
        prediction_at_k = tp_at_k / k_eff
        hits_at_k = tp_at_k / max(total_true, 1)
        ranking_metrics[f"prediction_at_{k}"] = round(prediction_at_k, 4)
        ranking_metrics[f"hits_at_{k}"] = round(hits_at_k, 4)

    if verbose:
        print(f"\n{'='*50}")
        print(f"  LINK PREDICTION METRICS — {model_name}")
        print(f"{'='*50}")
        print(f"  Threshold used:      {threshold}")
        print(f"  Precision:           {precision:.4f}")
        print(f"  Recall:              {recall:.4f}")
        print(f"  F1-Score:            {f1:.4f}")
        print(f"  AUC-ROC:             {auc_roc:.4f}  ← threshold-independent")
        print(f"  Average Precision:   {avg_prec:.4f}  ← threshold-independent")
        for k in k_values:
            pred_key = f"prediction_at_{k}"
            hit_key = f"hits_at_{k}"
            if pred_key in ranking_metrics and hit_key in ranking_metrics:
                print(
                    f"  Prediction@{k}:       {ranking_metrics[pred_key]:.4f}"
                    f"  | Hits@{k}: {ranking_metrics[hit_key]:.4f}"
                )
        print(f"{'='*50}\n")

    results = {
        "model": model_name,
        "task": "link_prediction",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc_roc": round(auc_roc, 4),
        "avg_precision": round(avg_prec, 4),
    }
    results.update(ranking_metrics)
    return results


# ──────────────────────────────────────────────────────────────
# QUICK SELF-TEST (runs only when you execute this file directly)
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running metrics.py self-test with synthetic data...\n")

    # --- Node Classification test ---
    # Simulating 10 predictions where 8 are correct
    node_true = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2]
    node_pred = [0, 1, 2, 3, 4, 5, 6, 1, 1, 0]  # positions 7 and 9 are wrong

    node_results = calculate_node_metrics(node_true, node_pred, model_name="Test-GCN")
    print(f"Returned dict: {node_results}\n")

    # --- Link Prediction test ---
    # Simulating 10 edge predictions: 5 real edges, 5 fake edges
    link_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    link_scores = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.4, 0.35]

    link_results = calculate_link_metrics(
        link_true, link_scores, model_name="Test-VGAE"
    )
    print(f"Returned dict: {link_results}")

    print("\n✅ metrics.py self-test complete.")
