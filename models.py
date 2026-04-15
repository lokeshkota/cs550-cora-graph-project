"""
models.py
---------
Neural network architecture definitions for the CS550 project.

This file contains ONLY model class definitions — no training, no data loading.
The training logic lives in train_node.py and train_link.py.

Models defined here:
  [x] GCN  — Graph Convolutional Network (node classification baseline)
  [ ] GAT  — Graph Attention Network     (node classification main model)
              → Will be added here after GCN is verified working.

How GCN works (intuition):
  Regular neural networks look at each node's features in isolation.
  GCN propagates information across edges — each node aggregates
  (averages) the feature vectors of all its direct neighbors, then
  passes the combined representation through learned weight matrices.

  Two GCN layers means each node "sees" information from:
    Layer 1: its immediate neighbors (1-hop)
    Layer 2: neighbors of neighbors  (2-hop)

  For Cora: a paper about Reinforcement Learning will be surrounded by
  other RL papers. After 2 hops, the model has strong contextual signal
  to classify it correctly — even if its own word vector is ambiguous.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# ──────────────────────────────────────────────────────────────
#  MODEL 1: GCN — Baseline for Node Classification
# ──────────────────────────────────────────────────────────────

class GCN(torch.nn.Module):
    """
    Two-layer Graph Convolutional Network for node classification.

    Architecture:
        Input (1433 word features)
            ↓  GCNConv Layer 1  [1433 → hidden_channels]
            ↓  ReLU activation
            ↓  Dropout (randomly zeroes neurons to prevent overfitting)
            ↓  GCNConv Layer 2  [hidden_channels → 7 classes]
        Output (raw scores for each of the 7 Cora topics)

    Why two layers?
        One layer only aggregates 1-hop neighbors.
        Two layers gives 2-hop reach — proven sweet spot for Cora.
        More layers cause "oversmoothing" (all nodes become similar).

    Why dropout?
        During training, randomly disables `p` fraction of neurons.
        Forces the network to not rely on any single feature.
        Turned OFF automatically during evaluation (self.training=False).

    Args:
        in_channels     (int): Number of input features per node.
                               For Cora: 1433.
        hidden_channels (int): Size of the intermediate representation.
                               Default 64 — proven to work well on Cora.
        out_channels    (int): Number of output classes.
                               For Cora: 7.
        dropout         (float): Dropout probability. Default 0.5.
    """

    def __init__(self, in_channels, hidden_channels=64,
                 out_channels=7, dropout=0.5):
        super(GCN, self).__init__()
        self.dropout = dropout

        # Layer 1: reduces 1433 raw word features → 64 graph-aware features
        # GCNConv uses both the node's own features AND its neighbors' features
        self.conv1 = GCNConv(in_channels, hidden_channels)

        # Layer 2: maps 64 features → 7 class scores (one per Cora topic)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Defines the forward pass — how data flows through the network.

        Args:
            x          (Tensor): Node feature matrix. Shape [num_nodes, 1433].
            edge_index (Tensor): Graph connectivity in COO format. Shape [2, num_edges].
                                 Row 0 = source nodes, Row 1 = target nodes.

        Returns:
            Tensor: Raw class scores (logits). Shape [num_nodes, 7].
                    NOT probabilities yet — CrossEntropyLoss handles that.
        """
        # Pass through Layer 1 — each node aggregates neighbor features
        x = self.conv1(x, edge_index)

        # ReLU activation — replaces negative values with 0
        # Introduces non-linearity so the model can learn complex patterns
        x = F.relu(x)

        # Dropout — randomly zero out neurons during training only
        # self.training is automatically True during model.train()
        #                           and False during model.eval()
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Pass through Layer 2 — produces final class scores
        x = self.conv2(x, edge_index)

        # Return raw logits — CrossEntropyLoss in train_node.py
        # will convert these to probabilities internally
        return x


# ──────────────────────────────────────────────────────────────
#  MODEL 2: GAT — Main Model for Node Classification
#  STATUS: Not yet implemented — added after GCN is verified.
# ──────────────────────────────────────────────────────────────

# class GAT(torch.nn.Module):
#     """
#     Graph Attention Network — upgrade over GCN.
#
#     Key difference from GCN:
#       GCN averages neighbor features equally (every neighbor counts the same).
#       GAT learns WHICH neighbors to pay more attention to.
#       e.g. A paper citing 10 papers — some citations are more relevant than
#       others for classification. GAT figures that out automatically.
#
#     Multi-head attention:
#       Runs `heads` separate attention mechanisms in parallel,
#       then concatenates their outputs. Gives the model multiple
#       "perspectives" on the neighborhood.
#
#     Will be implemented here after GCN achieves ~81% accuracy.
#     """
#     pass


# ──────────────────────────────────────────────────────────────
#  QUICK SANITY CHECK
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Verifies the GCN model initialises and runs a forward pass
    without errors using dummy data matching Cora's dimensions.
    """
    print("Running models.py sanity check...\n")

    # Dummy data matching Cora's exact dimensions
    num_nodes     = 2708   # number of papers in Cora
    num_features  = 1433   # word-presence features per paper
    num_classes   = 7      # research topic categories
    num_edges     = 10556  # citation links (directed)

    # Random feature matrix — same shape as real Cora data
    x_dummy = torch.randn(num_nodes, num_features)

    # Random edge list — same shape as real Cora edge_index
    edge_index_dummy = torch.randint(0, num_nodes, (2, num_edges))

    # Initialise GCN
    model = GCN(
        in_channels=num_features,
        hidden_channels=64,
        out_channels=num_classes,
        dropout=0.5,
    )
    print(f"Model architecture:\n{model}\n")

    # Run a forward pass
    model.eval()  # disable dropout for the sanity check
    with torch.no_grad():
        out = model(x_dummy, edge_index_dummy)

    print(f"Input shape:  {x_dummy.shape}")
    print(f"Output shape: {out.shape}  ← should be [2708, 7]")
    assert out.shape == (num_nodes, num_classes), \
        f"Shape mismatch! Got {out.shape}, expected ({num_nodes}, {num_classes})"

    print("\n✅ GCN forward pass successful — shapes are correct.")
    print("   Ready to be trained in train_node.py")
