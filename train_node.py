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
from models import GCN
from metrics import calculate_node_metrics


# ── Hyperparameters ────────────────────────────────────────────
LR           = 0.01
WEIGHT_DECAY = 5e-4
HIDDEN       = 64
DROPOUT      = 0.5
EPOCHS       = 200
PRINT_EVERY  = 20     # print progress every N epochs
MODEL_PATH   = "models/gcn_best.pth"
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


def train_gcn(data, train_mask, val_mask, test_mask):
    """
    Full GCN training loop.

    Trains for EPOCHS epochs, saving the model state whenever
    validation accuracy improves (best checkpoint strategy).

    After training finishes, loads the best checkpoint and
    evaluates on the test set.

    Args:
        data       (Data):      Full Cora graph.
        train_mask (BoolTensor): Training nodes.
        val_mask   (BoolTensor): Validation nodes (used to select best epoch).
        test_mask  (BoolTensor): Test nodes (evaluated once at the very end).

    Returns:
        dict: Final metrics from metrics.py.
    """
    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Initialise model, optimiser, and loss function
    model = GCN(
        in_channels=data.num_node_features,
        hidden_channels=HIDDEN,
        out_channels=int(data.y.max().item()) + 1,
        dropout=DROPOUT,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion = torch.nn.CrossEntropyLoss()
    # CrossEntropyLoss = Softmax + Negative Log Likelihood in one step.
    # It converts raw class scores → probabilities, then penalises
    # the model for assigning low probability to the correct class.

    print(f"\n[GCN] Starting training — {EPOCHS} epochs")
    print(f"      LR={LR}, Hidden={HIDDEN}, Dropout={DROPOUT}, "
          f"Weight Decay={WEIGHT_DECAY}")
    print(f"{'─'*55}")
    print(f"  {'Epoch':>6}  {'Train Loss':>11}  "
          f"{'Val Acc':>9}  {'Best Val':>9}")
    print(f"{'─'*55}")

    best_val_acc  = 0.0
    best_epoch    = 0
    history       = []   # track loss + accuracy per epoch for plotting later

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_one_epoch(model, data, train_mask, optimizer, criterion)
        val_acc, _, _ = evaluate(model, data, val_mask)

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_acc": val_acc})

        # Save model if this is the best validation accuracy so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), MODEL_PATH)

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            marker = " ← best" if epoch == best_epoch else ""
            print(f"  {epoch:>6}  {train_loss:>11.4f}  "
                  f"{val_acc:>9.4f}  {best_val_acc:>9.4f}{marker}")

    print(f"{'─'*55}")
    print(f"\n[GCN] Training complete.")
    print(f"      Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"      Saved weights → {MODEL_PATH}")

    # ── Final Evaluation on Test Set ──────────────────────────
    # Load the best checkpoint (not the last epoch's weights)
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"\n[GCN] Evaluating best checkpoint on test set...")

    _, y_pred, y_true = evaluate(model, data, test_mask)
    results = calculate_node_metrics(y_pred=y_pred, y_true=y_true,
                                     model_name="GCN")

    # ── Save results to CSV ───────────────────────────────────
    results["best_epoch"] = best_epoch
    results["best_val_acc"] = round(best_val_acc, 4)

    file_exists = os.path.isfile(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

    print(f"\n[GCN] Results saved → {RESULTS_PATH}")
    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"{'='*55}")
    print(f"  LR Baseline accuracy:  76.57%")
    print(f"  GCN Test accuracy:     {results['accuracy']*100:.2f}%")
    delta = results['accuracy'] - 0.7657
    symbol = "▲" if delta > 0 else "▼"
    print(f"  Improvement:           {symbol} {abs(delta)*100:.2f}%")
    print(f"{'='*55}\n")

    return results


if __name__ == "__main__":
    # Load data
    _, data = load_cora()

    # Create 70/10/20 split
    train_mask, val_mask, test_mask = make_three_way_split(data)

    # Train GCN and evaluate
    results = train_gcn(data, train_mask, val_mask, test_mask)

    print("✅ train_node.py complete.")
