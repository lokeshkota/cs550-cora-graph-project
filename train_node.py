"""
train_node.py
-------------
Training loop for node classification on the Cora dataset.
Trains the GCN model and evaluates it against the LR baseline.

What this file does, step by step:
  1. Loads Cora data via data_loader.py
  2. Creates a 70% train / 10% val / 20% test split
     (extends the 80/20 split from baseline_model.py)
  3. Initialises the GCN model from models.py
  4. Trains for up to 200 epochs using Adam optimiser + CrossEntropyLoss
  5. Saves the best model weights whenever val accuracy improves
  6. Evaluates on the held-out 20% test set using metrics.py
  7. Saves all results to outputs/results.csv

Why a validation set?
  We split the 80% training data into:
    - 70% actual training  → model learns from this
    - 10% validation       → we watch accuracy here to pick the best epoch
  If we picked the best model based on the TEST set, we'd be cheating
  (the test set would influence our model selection). The val set is
  a clean way to select the best checkpoint without touching test data.

Hyperparameters (tuned for Cora):
  lr           = 0.01   — how fast the model updates its weights each step
  weight_decay = 5e-4   — L2 regularisation, penalises overly large weights
  hidden       = 64     — size of the middle layer in GCN
  dropout      = 0.5    — fraction of neurons disabled each training step
  epochs       = 200    — number of full passes through training data
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import csv
from sklearn.model_selection import train_test_split

from data_loader import load_cora
from baseline_model import create_80_20_split
from models import GCN, GAT
from metrics import calculate_node_metrics


# ── GCN Hyperparameters ───────────────────────────────────────
GCN_LR           = 0.01
GCN_WEIGHT_DECAY = 5e-4
GCN_HIDDEN       = 64
GCN_DROPOUT      = 0.5

# ── GAT Hyperparameters ───────────────────────────────────────
GAT_LR           = 0.005   # GAT converges with a slightly lower lr
GAT_WEIGHT_DECAY = 5e-4
GAT_HEADS        = 8       # 8 parallel attention heads
GAT_HIDDEN_HEAD  = 8       # 8 features per head → 64 total (same as GCN)
GAT_DROPOUT      = 0.6     # GAT paper uses 0.6

# ── Shared settings ───────────────────────────────────────
EPOCHS       = 200
PRINT_EVERY  = 20
RESULTS_PATH = "outputs/results.csv"
# ──────────────────────────────────────────────────────────────


def make_three_way_split(data, random_state=42):
    """
    Creates a 70% train / 10% val / 20% test split.

    Strategy:
      1. Use create_80_20_split() to get 80% train, 20% test (same as LR baseline)
      2. Split the 80% further: 87.5% of 80% = 70% train
                                12.5% of 80% = 10% val

    This gives us a clean val set to pick the best training checkpoint.

    Args:
        data (Data): PyG Data object from load_cora().

    Returns:
        train_mask, val_mask, test_mask: Boolean tensors [num_nodes].
    """
    num_nodes = data.num_nodes

    # Step 1: get the base 80/20 split
    train_mask_80, test_mask = create_80_20_split(data, random_state=random_state)

    # Step 2: extract the 80% train indices, then split into 70/10
    train_indices = torch.where(train_mask_80)[0].numpy()
    labels_train  = data.y[train_mask_80].numpy()

    train_idx, val_idx = train_test_split(
        train_indices,
        test_size=0.125,        # 12.5% of 80% = 10% of total
        random_state=random_state,
        stratify=labels_train,  # keep class balance
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True

    print(f"\n[Split] Final 70 / 10 / 20 Split:")
    print(f"  Train: {train_mask.sum().item():>5}  "
          f"({train_mask.sum().item() / num_nodes * 100:.1f}%)")
    print(f"  Val:   {val_mask.sum().item():>5}  "
          f"({val_mask.sum().item() / num_nodes * 100:.1f}%)")
    print(f"  Test:  {test_mask.sum().item():>5}  "
          f"({test_mask.sum().item() / num_nodes * 100:.1f}%)")

    return train_mask, val_mask, test_mask


def train_one_epoch(model, data, train_mask, optimizer, criterion):
    """
    Runs a single training epoch.

    One epoch = one full pass through all training nodes:
      1. Zero out gradients from the previous step
         (PyTorch accumulates them by default — we reset each epoch)
      2. Forward pass: feed data through GCN, get class scores
      3. Compute loss on TRAINING nodes only
         (val and test nodes are invisible to the model during training)
      4. Backward pass: compute gradients via backpropagation
      5. Optimiser step: nudge weights in the direction that reduces loss

    Args:
        model      (GCN):       The neural network.
        data       (Data):      Full Cora graph.
        train_mask (BoolTensor): Identifies training nodes.
        optimizer:              Adam optimiser instance.
        criterion:              CrossEntropyLoss instance.

    Returns:
        float: Training loss for this epoch.
    """
    model.train()           # enables dropout
    optimizer.zero_grad()   # reset gradients

    # Forward pass — compute class scores for ALL nodes
    out = model(data.x, data.edge_index)

    # Compute loss on training nodes ONLY
    # out[train_mask] → class scores for training nodes
    # data.y[train_mask] → true labels for training nodes
    loss = criterion(out[train_mask], data.y[train_mask])

    loss.backward()         # compute gradients
    optimizer.step()        # update weights

    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    """
    Evaluates the model on a given set of nodes (val or test).
    Gradient computation is disabled — we're not learning, just measuring.

    Args:
        model (GCN):       The trained neural network.
        data  (Data):      Full Cora graph.
        mask  (BoolTensor): Identifies which nodes to evaluate.

    Returns:
        accuracy (float): Fraction of correct predictions.
        y_pred   (np.array): Predicted labels for these nodes.
        y_true   (np.array): True labels for these nodes.
    """
    model.eval()            # disables dropout for evaluation

    out    = model(data.x, data.edge_index)
    preds  = out[mask].argmax(dim=1)   # pick class with highest score

    y_pred = preds.cpu().numpy()
    y_true = data.y[mask].cpu().numpy()

    accuracy = (y_pred == y_true).mean()
    return accuracy, y_pred, y_true


def train_model(model, data, train_mask, val_mask, test_mask,
                lr, weight_decay, model_path, model_name):
    """
    Generic training loop — works for both GCN and GAT.

    Trains for EPOCHS epochs, saving the best checkpoint based on
    validation accuracy. After training, loads best checkpoint and
    evaluates on the test set.

    Args:
        model        (nn.Module):   Initialised model (GCN or GAT).
        data         (Data):        Full Cora graph.
        train_mask   (BoolTensor):  Training nodes.
        val_mask     (BoolTensor):  Validation nodes.
        test_mask    (BoolTensor):  Test nodes.
        lr           (float):       Learning rate.
        weight_decay (float):       L2 regularisation strength.
        model_path   (str):         Where to save the best weights.
        model_name   (str):         Display name (e.g. 'GCN', 'GAT').

    Returns:
        dict: Final metrics from metrics.py.
    """
    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()

    print(f"\n[{model_name}] Starting training — {EPOCHS} epochs")
    print(f"{'─'*55}")
    print(f"  {'Epoch':>6}  {'Train Loss':>11}  "
          f"{'Val Acc':>9}  {'Best Val':>9}")
    print(f"{'─'*55}")

    best_val_acc = 0.0
    best_epoch   = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, data, train_mask, optimizer, criterion)
        val_acc, _, _ = evaluate(model, data, val_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), model_path)

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            marker = " ← best" if epoch == best_epoch else ""
            print(f"  {epoch:>6}  {train_loss:>11.4f}  "
                  f"{val_acc:>9.4f}  {best_val_acc:>9.4f}{marker}")

    print(f"{'─'*55}")
    print(f"\n[{model_name}] Best val accuracy: {best_val_acc:.4f} "
          f"at epoch {best_epoch}")
    print(f"[{model_name}] Saved weights → {model_path}")

    # Load best checkpoint and evaluate on test set
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"\n[{model_name}] Evaluating best checkpoint on test set...")
    _, y_pred, y_true = evaluate(model, data, test_mask)
    results = calculate_node_metrics(
        y_pred=y_pred, y_true=y_true, model_name=model_name
    )

    # Append to results CSV
    results["best_epoch"]   = best_epoch
    results["best_val_acc"] = round(best_val_acc, 4)
    file_exists = os.path.isfile(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

    delta = results['accuracy'] - 0.7657
    symbol = "▲" if delta > 0 else "▼"
    print(f"  Improvement:           {symbol} {abs(delta)*100:.2f}%")
    print(f"{'='*55}\n")

    return results


def train_gcn(data, train_mask, val_mask, test_mask):
    """Initialises and trains the GCN model."""
    model = GCN(
        in_channels=data.num_node_features,
        hidden_channels=GCN_HIDDEN,
        out_channels=int(data.y.max().item()) + 1,
        dropout=GCN_DROPOUT,
    )
    return train_model(
        model, data, train_mask, val_mask, test_mask,
        lr=GCN_LR, weight_decay=GCN_WEIGHT_DECAY,
        model_path="models/gcn_best.pth", model_name="GCN",
    )


def train_gat(data, train_mask, val_mask, test_mask):
    """Initialises and trains the GAT model."""
    model = GAT(
        in_channels=data.num_node_features,
        hidden_per_head=GAT_HIDDEN_HEAD,
        out_channels=int(data.y.max().item()) + 1,
        heads=GAT_HEADS,
        dropout=GAT_DROPOUT,
    )
    return train_model(
        model, data, train_mask, val_mask, test_mask,
        lr=GAT_LR, weight_decay=GAT_WEIGHT_DECAY,
        model_path="models/gat_best.pth", model_name="GAT",
    )


if __name__ == "__main__":

    # Load data
    _, data = load_cora()

    # Create 70/10/20 split (same for both models — fair comparison)
    train_mask, val_mask, test_mask = make_three_way_split(data)

    # ── Train GCN first ──────────────────────────────────────
    gcn_results = train_gcn(data, train_mask, val_mask, test_mask)

    # ── Train GAT second ─────────────────────────────────────
    gat_results = train_gat(data, train_mask, val_mask, test_mask)

    # ── Final side-by-side comparison ────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Model':<12} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1':>10}")
    print(f"  {'-'*52}")
    print(f"  {'LR Baseline':<12} {'76.57%':>10} {'-':>10} "
          f"{'-':>10} {'-':>10}")
    for r in [gcn_results, gat_results]:
        print(f"  {r['model']:<12} "
              f"{r['accuracy']*100:>9.2f}% "
              f"{r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} "
              f"{r['f1']:>10.4f}")
    print(f"{'='*60}")

    print("\n✅ train_node.py complete.")
