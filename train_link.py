"""
train_link.py
-------------
Link prediction training on Cora using:
  1) GAE  (baseline)
  2) VGAE (main model)

Workflow:
  - Load Cora
  - Split edges with RandomLinkSplit (80/10/10)
  - Train each model with reconstruction loss
  - Select best checkpoint by validation AUC-ROC
  - Evaluate on test split and append metrics to outputs/results.csv
  - Save ROC artifacts for plotting
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import GAE, VGAE, GCNConv
from torch_geometric.transforms import RandomLinkSplit

from data_loader import load_cora
from metrics import calculate_link_metrics


# Hyperparameters
HIDDEN_CHANNELS = 64
LATENT_CHANNELS = 32
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 300
PRINT_EVERY = 20
PATIENCE = 40
RANDOM_FEATURE_DIM = 32

# Harder split configuration
DISJOINT_TRAIN_RATIO = float(os.getenv("LP_DISJOINT_RATIO", "0.0"))
USE_RANDOM_FEATURES = os.getenv("LP_USE_RANDOM_FEATURES", "0") == "1"


def _build_experiment_tag():
    if DISJOINT_TRAIN_RATIO > 0.0 or USE_RANDOM_FEATURES:
        tag = f"disjoint_{DISJOINT_TRAIN_RATIO}"
        if USE_RANDOM_FEATURES:
            tag = f"{tag}_randnx{RANDOM_FEATURE_DIM}"
        return tag
    return ""


EXPERIMENT_TAG = _build_experiment_tag()


def _suffix(tag):
    return f"_{tag}" if tag else ""


_TAG_SUFFIX = _suffix(EXPERIMENT_TAG)

# Paths
RESULTS_PATH = f"outputs/results{_TAG_SUFFIX}.csv"
GAE_WEIGHTS_PATH = f"models/gae_best{_TAG_SUFFIX}.pth"
VGAE_WEIGHTS_PATH = f"models/vgae_best{_TAG_SUFFIX}.pth"
GAE_ROC_PATH = f"outputs/gae_roc_data{_TAG_SUFFIX}.npz"
VGAE_ROC_PATH = f"outputs/vgae_roc_data{_TAG_SUFFIX}.npz"


class GAEEncoder(torch.nn.Module):
    """Two-layer GCN encoder that outputs node embeddings for GAE."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


class VGAEEncoder(torch.nn.Module):
    """Variational GCN encoder with separate mean/log-std heads."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


def make_edge_splits(data, seed=42):
    """Create 80/10/10 train/val/test edge splits with negatives for val/test."""
    splitter = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
        split_labels=False,
        disjoint_train_ratio=DISJOINT_TRAIN_RATIO,
    )

    torch.manual_seed(seed)
    train_data, val_data, test_data = splitter(data)
    print("Message edges:", train_data.edge_index.shape[1])  # message passing
    print("Label edges:", train_data.edge_label_index.shape[1])  # supervision

    def _count_labels(split_name, split_data):
        labels = split_data.edge_label
        pos = int((labels == 1).sum().item())
        neg = int((labels == 0).sum().item())
        print(
            f"  {split_name:<5} | message edges: {split_data.edge_index.size(1):>5} | "
            f"label edges: {labels.numel():>5} (pos={pos}, neg={neg})"
        )

    print("\n[Link Split] RandomLinkSplit (80/10/10)")
    print(f"  disjoint_train_ratio={DISJOINT_TRAIN_RATIO}")
    print(
        "  note: message-passing train edges and supervised train edges are partially disjoint"
    )
    _count_labels("train", train_data)
    _count_labels("val", val_data)
    _count_labels("test", test_data)

    return train_data, val_data, test_data


def _positive_edge_index_from_labels(split_data):
    """Extract positive edge indices from edge_label/edge_label_index tensors."""
    pos_mask = split_data.edge_label == 1
    return split_data.edge_label_index[:, pos_mask]


def evaluate_link_model(model, split_data, model_name, threshold=0.5, verbose=False):
    """Run edge scoring on one split and compute rubric metrics."""
    model.eval()
    with torch.no_grad():
        z = model.encode(split_data.x, split_data.edge_index)
        logits = model.decode(z, split_data.edge_label_index)
        y_scores = torch.sigmoid(logits).cpu().numpy()

    y_true = split_data.edge_label.cpu().numpy().astype(int)
    metrics = calculate_link_metrics(
        y_true=y_true,
        y_scores=y_scores,
        threshold=threshold,
        model_name=model_name,
        verbose=verbose,
    )
    return metrics, y_true, y_scores


def find_best_threshold_by_f1(y_true, y_scores, model_name="Model"):
    """Select threshold on validation scores that maximizes F1."""
    threshold_grid = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in threshold_grid:
        metrics = calculate_link_metrics(
            y_true=y_true,
            y_scores=y_scores,
            threshold=float(threshold),
            model_name=model_name,
            verbose=False,
        )
        current_f1 = float(metrics["f1"])
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)

    return round(best_threshold, 4), round(best_f1, 4)


def append_row_to_results_csv(row, csv_path=RESULTS_PATH):
    """Append one metrics row while preserving mixed node/link schemas safely."""
    if os.path.isfile(csv_path):
        existing = pd.read_csv(csv_path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])

    updated.to_csv(csv_path, index=False)


def _save_roc_artifacts(path, y_true, y_scores):
    np.savez(path, y_true=np.asarray(y_true), y_scores=np.asarray(y_scores))
    print(f"  Saved ROC artifacts -> {path}")


def train_single_model(
    model,
    train_data,
    val_data,
    test_data,
    model_name,
    weight_path,
    roc_path,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    epochs=EPOCHS,
    patience=PATIENCE,
):
    """Train one link model (GAE or VGAE) and persist best checkpoint + metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = -1.0
    best_epoch = 0
    stale_epochs = 0
    best_state_dict = None

    print(f"\n[{model_name}] Training for {epochs} epochs")
    print(f"{'-' * 63}")
    print(f"  {'Epoch':>6}  {'Train Loss':>11}  {'Val AUC':>9}  {'Best Val':>9}")
    print(f"{'-' * 63}")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model.encode(train_data.x, train_data.edge_index)
        pos_edge_index = _positive_edge_index_from_labels(train_data)
        loss = model.recon_loss(z, pos_edge_index)

        if isinstance(model, VGAE):
            loss = loss + (1.0 / train_data.num_nodes) * model.kl_loss()

        loss.backward()
        optimizer.step()

        val_metrics, _, _ = evaluate_link_model(
            model=model,
            split_data=val_data,
            model_name=model_name,
            verbose=False,
        )
        val_auc = float(val_metrics["auc_roc"])

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            stale_epochs = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, weight_path)
        else:
            stale_epochs += 1

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            marker = " <- best" if epoch == best_epoch else ""
            print(
                f"  {epoch:>6}  {loss.item():>11.4f}  {val_auc:>9.4f}  "
                f"{best_val_auc:>9.4f}{marker}"
            )

        if stale_epochs >= patience:
            print(
                f"[{model_name}] Early stop: no val AUC improvement for {patience} epochs."
            )
            break

    print(f"{'-' * 63}")
    print(f"[{model_name}] Best val AUC: {best_val_auc:.4f} at epoch {best_epoch}")
    print(f"[{model_name}] Saved weights -> {weight_path}")

    if best_state_dict is None:
        best_state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(best_state_dict)

    val_metrics, val_y_true, val_y_scores = evaluate_link_model(
        model=model,
        split_data=val_data,
        model_name=model_name,
        threshold=0.5,
        verbose=False,
    )
    best_threshold, best_val_f1 = find_best_threshold_by_f1(
        y_true=val_y_true,
        y_scores=val_y_scores,
        model_name=model_name,
    )
    print(
        f"[{model_name}] Selected threshold on validation split: "
        f"{best_threshold} (val F1={best_val_f1}, val AUC={val_metrics['auc_roc']})"
    )

    print(f"[{model_name}] Evaluating best checkpoint on test split...")
    test_metrics, y_true, y_scores = evaluate_link_model(
        model=model,
        split_data=test_data,
        model_name=model_name,
        threshold=best_threshold,
        verbose=True,
    )

    test_metrics["best_epoch"] = int(best_epoch)
    test_metrics["best_val_auc"] = round(best_val_auc, 4)
    test_metrics["selected_threshold"] = best_threshold
    test_metrics["val_f1_at_selected_threshold"] = best_val_f1
    append_row_to_results_csv(test_metrics)
    _save_roc_artifacts(roc_path, y_true, y_scores)

    return test_metrics


def run_link_prediction(seed=42):
    """Main execution for link prediction training and evaluation."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    _, data = load_cora()
    if USE_RANDOM_FEATURES:
        data.x = torch.randn(
            data.num_nodes,
            RANDOM_FEATURE_DIM,
            dtype=data.x.dtype,
            device=data.x.device,
        )
        print(
            "[train_link.py] Feature ablation ON: "
            f"replaced data.x with random Gaussian features of shape "
            f"[{data.num_nodes}, {RANDOM_FEATURE_DIM}]"
        )

    train_data, val_data, test_data = make_edge_splits(data=data, seed=seed)

    in_channels = train_data.num_node_features

    gae = GAE(
        encoder=GAEEncoder(
            in_channels=in_channels,
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=LATENT_CHANNELS,
        )
    )
    vgae = VGAE(
        encoder=VGAEEncoder(
            in_channels=in_channels,
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=LATENT_CHANNELS,
        )
    )

    gae_results = train_single_model(
        model=gae,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        model_name="GAE",
        weight_path=GAE_WEIGHTS_PATH,
        roc_path=GAE_ROC_PATH,
    )

    vgae_results = train_single_model(
        model=vgae,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        model_name="VGAE",
        weight_path=VGAE_WEIGHTS_PATH,
        roc_path=VGAE_ROC_PATH,
    )

    print("\n[train_link.py] Summary")
    print(f"  Experiment tag: {EXPERIMENT_TAG}")
    print(f"  disjoint_train_ratio: {DISJOINT_TRAIN_RATIO}")
    print(f"  use_random_features: {USE_RANDOM_FEATURES}")
    if USE_RANDOM_FEATURES:
        print(f"  random_feature_dim: {RANDOM_FEATURE_DIM}")
    print(f"  GAE  -> AUC-ROC: {gae_results['auc_roc']}, F1: {gae_results['f1']}")
    print(f"  VGAE -> AUC-ROC: {vgae_results['auc_roc']}, F1: {vgae_results['f1']}")
    print(f"  Results appended -> {RESULTS_PATH}")

    return {"GAE": gae_results, "VGAE": vgae_results}


if __name__ == "__main__":
    run_link_prediction(seed=42)
