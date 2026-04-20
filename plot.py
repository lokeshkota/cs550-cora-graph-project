"""
plot.py
-------
Generates report charts for:
  1) Node classification comparison (LR vs GCN vs GAT)
  2) Link prediction comparison (GAE vs VGAE)
  3) ROC curve for link prediction
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.manifold import TSNE

from data_loader import load_cora
from baseline_model import create_80_20_split, run_baseline
from metrics import calculate_node_metrics
from models import GAT


EXPERIMENT_TAG = ""

NODE_MODELS = ["LR", "GCN", "GAT"]
LINK_MODELS = ["GAE", "VGAE"]

NODE_METRICS = ["accuracy", "precision", "recall", "f1"]
LINK_BAR_METRICS = ["precision", "recall", "f1", "auc_roc", "avg_precision"]

LINK_METRIC_GROUPS = {
    "Core Metrics": [
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "avg_precision",
    ],
    "Threshold Tuning": [
        "selected_threshold",
        "val_f1_at_selected_threshold",
    ],
    "Top-K Ranking": [
        "prediction_at_10",
        "hits_at_10",
        "prediction_at_50",
        "hits_at_50",
        "prediction_at_100",
        "hits_at_100",
    ],
}

METRIC_DISPLAY_NAMES = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "auc_roc": "AUC-ROC",
    "avg_precision": "Avg Precision",
    "selected_threshold": "Selected Threshold",
    "val_f1_at_selected_threshold": "Val F1 @ Selected Threshold",
    "prediction_at_10": "Prediction@10",
    "hits_at_10": "Hits@10",
    "prediction_at_50": "Prediction@50",
    "hits_at_50": "Hits@50",
    "prediction_at_100": "Prediction@100",
    "hits_at_100": "Hits@100",
}


def _suffix(tag):
    return f"_{tag}" if tag else ""


_TAG_SUFFIX = _suffix(EXPERIMENT_TAG)


RESULTS_PATH = f"outputs/results{_TAG_SUFFIX}.csv"
OUTPUT_DIR = "outputs"
ACADEMIC_DIR = "selected_plots"

# File basenames for synchronization
NODE_PLOT_FILE = f"node_classification_comparison{_TAG_SUFFIX}.png"
LINK_BAR_FILE = f"gae_vgae_comparison{_TAG_SUFFIX}.png"
LINK_TABLE_FILE = f"gae_vgae_table{_TAG_SUFFIX}.png"
ROC_PLOT_FILE = f"link_prediction_roc{_TAG_SUFFIX}.png"
CM_PLOT_FILE = f"confusion_matrix{_TAG_SUFFIX}.png"
TSNE_PLOT_FILE = f"tsne_clusters{_TAG_SUFFIX}.png"

# Full paths
NODE_PLOT_PATH = os.path.join(OUTPUT_DIR, NODE_PLOT_FILE)
LINK_BAR_PATH = os.path.join(OUTPUT_DIR, LINK_BAR_FILE)
LINK_TABLE_PATH = os.path.join(OUTPUT_DIR, LINK_TABLE_FILE)
ROC_PLOT_PATH = os.path.join(OUTPUT_DIR, ROC_PLOT_FILE)
CM_PLOT_PATH = os.path.join(OUTPUT_DIR, CM_PLOT_FILE)
TSNE_PLOT_PATH = os.path.join(OUTPUT_DIR, TSNE_PLOT_FILE)


def _ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ACADEMIC_DIR, exist_ok=True)


def _save_dual(filename):
    """Saves the current figure to both outputs/ and selected_plots/ for sync."""
    out_path = os.path.join(OUTPUT_DIR, filename)
    acad_path = os.path.join(ACADEMIC_DIR, filename)
    
    plt.savefig(out_path, dpi=220)
    plt.savefig(acad_path, dpi=220)
    print(f"[plot.py] Synced -> {out_path} & {acad_path}")


def _safe_load_results(path=RESULTS_PATH):
    if os.path.isfile(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def _rows_for_models(df, task_name, model_names):
    if df.empty or "task" not in df.columns or "model" not in df.columns:
        return []
    task_df = df[df["task"] == task_name].copy()
    rows = []
    for model_name in model_names:
        row = _latest_row(task_df, model_name)
        if row is not None:
            rows.append(row)
    return rows


def _latest_row(df, model_name):
    if df.empty or "model" not in df.columns:
        return None
    rows = df[df["model"] == model_name]
    if rows.empty:
        return None
    return rows.iloc[-1].to_dict()


def _compute_lr_baseline_row():
    """Fallback to compute LR metrics if not present in outputs/results.csv."""
    _, data = load_cora()
    train_mask, test_mask = create_80_20_split(data)
    _, y_pred, y_true = run_baseline(data, train_mask, test_mask)
    metrics = calculate_node_metrics(
        y_true=y_true,
        y_pred=y_pred,
        model_name="LR",
        verbose=False,
    )
    return metrics


def _add_bar_labels(ax, decimals=4):
    """Annotate grouped bar plots with numeric values."""
    for container in ax.containers:
        labels = []
        for bar in container:
            height = bar.get_height()
            if pd.isna(height):
                labels.append("")
            else:
                labels.append(f"{height:.{decimals}f}")
        ax.bar_label(container, labels=labels, padding=1, fontsize=8, rotation=0)


def plot_node_classification_comparison(df):
    """Bar chart for LR vs GCN vs GAT across Accuracy/Precision/Recall/F1."""
    rows = _rows_for_models(
        df, task_name="node_classification", model_names=NODE_MODELS
    )

    if not any(r.get("model") == "LR" for r in rows):
        print(
            f"[plot.py] LR row not found in {RESULTS_PATH}. Computing baseline metrics now..."
        )
        rows.append(_compute_lr_baseline_row())

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        print(
            "[plot.py] Skipping node chart: no node-classification metrics available."
        )
        return

    missing_cols = [c for c in NODE_METRICS if c not in plot_df.columns]
    if missing_cols:
        print(f"[plot.py] Skipping node chart: missing columns {missing_cols}")
        return

    plot_df = plot_df[["model"] + NODE_METRICS].set_index("model")
    plot_df = plot_df.reindex(NODE_MODELS).dropna(how="all")

    sns.set_theme(style="whitegrid")
    ax = plot_df.plot(kind="bar", figsize=(11, 6), width=0.78)
    ax.set_title("Node Classification Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_ylim(0.0, 1.08)
    ax.legend(title="Metric", loc="lower right")
    _add_bar_labels(ax, decimals=4)
    plt.xticks(rotation=0)
    plt.tight_layout()
    _save_dual(NODE_PLOT_FILE)
    plt.close()
    print(f"[plot.py] Saved -> {NODE_PLOT_PATH}")


def plot_link_metrics_comparison(df):
    """Bar chart for GAE vs VGAE on core link-prediction metrics."""
    rows = _rows_for_models(df, task_name="link_prediction", model_names=LINK_MODELS)
    if not rows:
        print("[plot.py] Skipping link bar chart: no link metrics found.")
        return

    plot_df = pd.DataFrame(rows)
    missing_cols = [c for c in LINK_BAR_METRICS if c not in plot_df.columns]
    if missing_cols:
        print(f"[plot.py] Skipping link bar chart: missing columns {missing_cols}")
        return

    plot_df = plot_df[["model"] + LINK_BAR_METRICS].set_index("model")
    plot_df = plot_df.rename(columns=METRIC_DISPLAY_NAMES)
    plot_df = plot_df.reindex(LINK_MODELS).dropna(how="all")

    sns.set_theme(style="whitegrid")
    ax = plot_df.plot(kind="bar", figsize=(10, 6), width=0.75)
    ax.set_title(
        "Link Prediction: Core Metrics (GAE vs VGAE)", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_ylim(0.0, 1.08)
    ax.legend(title="Metric", loc="lower right")
    _add_bar_labels(ax, decimals=4)
    plt.xticks(rotation=0)
    plt.tight_layout()
    _save_dual(LINK_BAR_FILE)
    plt.close()
    print(f"[plot.py] Saved -> {LINK_BAR_PATH}")


def plot_link_metrics_table(df):
    """Heatmap-style table for fine-grained GAE/VGAE link metric comparison."""
    rows = _rows_for_models(df, task_name="link_prediction", model_names=LINK_MODELS)
    if not rows:
        print("[plot.py] Skipping link table: no link metrics found.")
        return

    plot_df = pd.DataFrame(rows)
    desired_cols = [m for group in LINK_METRIC_GROUPS.values() for m in group]
    available_cols = [c for c in desired_cols if c in plot_df.columns]
    if not available_cols:
        print(
            "[plot.py] Skipping link table: none of the required metric columns found."
        )
        return

    base_table_df = (
        plot_df[["model"] + available_cols].set_index("model").reindex(LINK_MODELS).T
    )

    table_rows = []
    ytick_labels = []
    header_rows = []

    for group_name, cols in LINK_METRIC_GROUPS.items():
        present_cols = [c for c in cols if c in available_cols]
        if not present_cols:
            continue

        header_rows.append((len(table_rows), group_name))
        table_rows.append([np.nan] * len(LINK_MODELS))
        ytick_labels.append("")

        for col in present_cols:
            table_rows.append(base_table_df.loc[col].tolist())
            ytick_labels.append(METRIC_DISPLAY_NAMES.get(col, col))

    if not table_rows:
        print(
            "[plot.py] Skipping link table: no grouped rows available after filtering."
        )
        return

    table_df = pd.DataFrame(table_rows, columns=LINK_MODELS, index=ytick_labels)
    mask = table_df.isna()
    annot = table_df.map(lambda v: "" if pd.isna(v) else f"{v:.4f}")

    # All selected link metrics are in [0, 1], so a fixed color scale is valid.
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9.5, max(6.2, 0.5 * len(table_df.index) + 2)))
    ax = sns.heatmap(
        table_df,
        annot=annot,
        fmt="",
        mask=mask,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.8,
        linecolor="#d9d9d9",
        cbar_kws={"label": "Score"},
    )

    for row_idx, group_name in header_rows:
        ax.axhspan(row_idx, row_idx + 1, color="#f2f2f2", zorder=3)
        ax.hlines(row_idx, *ax.get_xlim(), colors="#2f2f2f", linewidth=2.2)
        ax.text(
            -0.02,
            row_idx + 0.5,
            group_name,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#222222",
        )

    plt.title(
        "Link Prediction Metrics Table (Grouped): GAE vs VGAE",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Model")
    plt.ylabel("Metric Group / Metric")
    plt.yticks(rotation=0)
    plt.tight_layout()
    _save_dual(LINK_TABLE_FILE)
    plt.close()
    print(f"[plot.py] Saved -> {LINK_TABLE_PATH}")


def _load_roc_artifacts():
    candidates = [
        ("GAE", f"outputs/gae_roc_data{_TAG_SUFFIX}.npz"),
        ("VGAE", f"outputs/vgae_roc_data{_TAG_SUFFIX}.npz"),
    ]
    loaded = []
    for model_name, path in candidates:
        if os.path.isfile(path):
            arr = np.load(path)
            loaded.append((model_name, arr["y_true"], arr["y_scores"]))
    return loaded


def plot_link_prediction_roc():
    """Create a ROC curve figure for link prediction from saved score artifacts."""
    roc_inputs = _load_roc_artifacts()
    if not roc_inputs:
        print("[plot.py] Skipping ROC plot: no ROC artifacts found in outputs/.")
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))

    for model_name, y_true, y_scores in roc_inputs:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2.2, label=f"{model_name} (AUC={roc_auc:.4f})")

    plt.plot(
        [0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, label="Random"
    )
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Link Prediction ROC Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    _save_dual(ROC_PLOT_FILE)
    plt.close()
    print(f"[plot.py] Saved -> {ROC_PLOT_PATH}")


def plot_node_diagnostics():
    """Generates T-SNE and Confusion Matrix from the trained GAT model."""
    _ensure_dirs()
    
    # Load data
    _, data = load_cora()
    
    # Load GAT Model
    model = GAT(
        in_channels=data.num_node_features,
        hidden_per_head=8,
        out_channels=int(data.y.max().item()) + 1,
        heads=8,
        dropout=0.6
    )
    
    weights_path = "models/gat_best.pth"
    if not os.path.exists(weights_path):
        print(f"[plot.py] Model weights not found at {weights_path}. Skipping T-SNE/CM.")
        return

    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    # Get predictions and latent embeddings
    with torch.no_grad():
        # Using output of first layer for embeddings
        embeddings = model.conv1(data.x, data.edge_index)
        outputs = model(data.x, data.edge_index)
        preds = outputs.argmax(dim=1)

    # 1. Confusion Matrix
    y_true = data.y[data.test_mask].numpy()
    y_pred = preds[data.test_mask].numpy()
    cm = confusion_matrix(y_true, y_pred)
    classes = ["Case Based", "Genetic Algorithms", "Neural Networks", "Probabilistic Methods", "Reinforcement Learning", "Rule Learning", "Theory"]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title("GAT Confusion Matrix (Cora Test Set)", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    _save_dual(CM_PLOT_FILE)
    plt.close()
    print(f"[plot.py] Saved -> {CM_PLOT_PATH}")

    # 2. T-SNE Clustering
    print("[plot.py] Running T-SNE clustering (this might take a minute)...")
    tsne = TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(embeddings.numpy())
    
    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(classes):
        mask = data.y.numpy() == i
        plt.scatter(z_tsne[mask, 0], z_tsne[mask, 1], label=cls, s=15, alpha=0.7)
    
    plt.title("T-SNE Visualization of GAT Latent Space Clusters", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    _save_dual(TSNE_PLOT_FILE)
    plt.close()
    print(f"[plot.py] Saved -> {TSNE_PLOT_PATH}")


def generate_all_plots():
    _ensure_dirs()
    print(f"[plot.py] Experiment tag: {EXPERIMENT_TAG}")
    df = _safe_load_results()
    plot_node_classification_comparison(df)
    plot_link_metrics_comparison(df)
    plot_link_metrics_table(df)
    plot_link_prediction_roc()
    plot_node_diagnostics()
    print("[plot.py] Finished generating available plots.")


if __name__ == "__main__":
    generate_all_plots()
