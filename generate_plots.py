import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import os

from data_loader import load_cora
from models import GAT

def generate_academic_plots():
    os.makedirs("selected_plots", exist_ok=True)
    
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
        print(f"Weights not found at {weights_path}")
        return

    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    # Get predictions and latent embeddings
    with torch.no_grad():
        # Using the output of the first layer as embeddings for T-SNE
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
    plt.title("GAT Confusion Matrix (Cora Test Set)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("selected_plots/confusion_matrix.png", dpi=200)
    print("Saved confusion_matrix.png")

    # 2. T-SNE Clustering
    print("Running T-SNE (this might take a minute)...")
    tsne = TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(embeddings.numpy())
    
    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(classes):
        mask = data.y.numpy() == i
        plt.scatter(z_tsne[mask, 0], z_tsne[mask, 1], label=cls, s=15, alpha=0.7)
    
    plt.title("T-SNE Visualization of GAT Latent Space Clusters")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("selected_plots/tsne_clusters.png", dpi=200)
    print("Saved tsne_clusters.png")

if __name__ == "__main__":
    generate_academic_plots()
