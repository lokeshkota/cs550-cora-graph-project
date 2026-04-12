"""
baseline_model.py
-----------------
Two responsibilities:
  1. Create a proper 80/20 stratified train/test split on node indices.
     (Overrides Cora's default which only trains on ~5% of nodes.)

  2. Train a Logistic Regression classifier using ONLY node features
     (word vectors) — completely ignoring the graph edges.
     This is our performance floor. The GCN must beat this score.

Why Logistic Regression as a baseline?
  It answers the question: "How much does the graph structure actually help?"
  If GCN barely beats LR, the graph isn't adding much value.
  If GCN crushes LR, the edges carry rich semantic meaning.
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from data_loader import load_cora


def create_80_20_split(data, random_state=42):
    """
    Creates a stratified 80/20 train/test split on node indices.

    'Stratified' means all 7 Cora topic classes are proportionally
    represented in both splits — no class gets accidentally left out.

    Args:
        data (Data):        PyG Data object from load_cora().
        random_state (int): Seed for reproducibility (same seed = same split).

    Returns:
        train_mask (BoolTensor): Shape [num_nodes], True for training nodes.
        test_mask  (BoolTensor): Shape [num_nodes], True for test nodes.
    """
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)          # [0, 1, 2, ..., 2707]
    labels = data.y.numpy()                 # Class label for each node

    # stratify=labels ensures class balance in both splits
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=random_state,
        stratify=labels,
    )

    # Convert index arrays into boolean masks (same shape as data.y)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    print("\n[Split] 80/20 Stratified Split Applied:")
    print(f"  Train nodes: {train_mask.sum().item():>5}  "
          f"({train_mask.sum().item() / num_nodes * 100:.1f}%)")
    print(f"  Test nodes:  {test_mask.sum().item():>5}  "
          f"({test_mask.sum().item() / num_nodes * 100:.1f}%)")

    return train_mask, test_mask


def run_baseline(data, train_mask, test_mask):
    """
    Trains a Logistic Regression model on raw node features only.

    No edges are used. This deliberately ignores the graph structure
    to establish a lower-bound accuracy that the GCN must surpass.

    Args:
        data (Data):             PyG Data object.
        train_mask (BoolTensor): Boolean mask for training nodes.
        test_mask  (BoolTensor): Boolean mask for test nodes.

    Returns:
        clf:    Trained LogisticRegression model.
        y_pred: Predicted labels for test nodes (numpy array).
        y_test: True labels for test nodes (numpy array).
    """
    # data.x is a [2708 x 1433] tensor — convert to numpy for sklearn
    X = data.x.numpy()
    y = data.y.numpy()

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print("\n[Baseline] Training Logistic Regression (features only, no edges)...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"[Baseline] Test Accuracy: {accuracy * 100:.2f}%")
    print("\n[Baseline] Per-Class Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("[Baseline] ⚠️  The GCN (Member 2) must beat this score!\n")

    return clf, y_pred, y_test


if __name__ == "__main__":
    _, data = load_cora()
    train_mask, test_mask = create_80_20_split(data)

    # Attach our custom 80/20 masks to the data object for use by other modules
    data.custom_train_mask = train_mask
    data.custom_test_mask = test_mask

    run_baseline(data, train_mask, test_mask)
    print("✅ baseline_model.py ran successfully.")
