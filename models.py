"""
models.py
---------
Neural network architecture definitions for the CS550 project.

This file contains ONLY model class definitions — no training, no data loading.
The training logic lives in train_node.py and train_link.py.

Models defined here:
  [x] GCN  — Graph Convolutional Network (node classification baseline)
  [x] GAT  — Graph Attention Network     (node classification main model)

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
from torch_geometric.nn import GCNConv, GATConv


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
# ──────────────────────────────────────────────────────────────

class GAT(torch.nn.Module):
    """
    Two-layer Graph Attention Network for node classification.

    Key upgrade over GCN:
      GCN: aggregates neighbor features by taking a SIMPLE AVERAGE.
           Every neighbor counts equally, regardless of relevance.
      GAT: learns an ATTENTION SCORE for each (node, neighbor) pair.
           More relevant neighbors get higher scores and contribute more.
           e.g. A paper about RL citing 10 papers — the 3 that are also
           about RL get high attention; the off-topic ones get low attention.

    Multi-head attention (heads=8):
      Runs 8 independent attention mechanisms in parallel.
      Each head learns to focus on DIFFERENT aspects of the neighborhood.
      Their outputs are concatenated after Layer 1, averaged after Layer 2.
      This is analogous to how Transformers work in NLP.

    Architecture:
        Input (1433 word features)
            ↓  Dropout on raw input features
            ↓  GATConv Layer 1  [1433 → 8 features × 8 heads → 64 features]
            ↓  ELU activation   (smoother than ReLU, standard for GAT)
            ↓  Dropout
            ↓  GATConv Layer 2  [64 → 7 classes, 1 head, averaged]
        Output (raw scores for each of the 7 Cora topics)

    Why ELU instead of ReLU?
      ELU (Exponential Linear Unit) allows small negative values instead of
      clamping at 0. This helps gradient flow and tends to work better with
      attention mechanisms.

    Args:
        in_channels  (int):   Input features per node. For Cora: 1433.
        hidden_per_head (int): Features per attention head in Layer 1.
                               Default 8. Total hidden = hidden_per_head × heads.
        out_channels (int):   Number of output classes. For Cora: 7.
        heads        (int):   Number of parallel attention heads. Default 8.
        dropout      (float): Dropout probability. Default 0.6.
    """

    def __init__(self, in_channels, hidden_per_head=8, out_channels=7,
                 heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.dropout = dropout

        # Layer 1: multi-head attention
        #   Each head produces `hidden_per_head` features.
        #   concat=True → concatenate all heads → total: hidden_per_head * heads
        #   e.g. 8 features × 8 heads = 64 features (same size as GCN hidden layer)
        self.conv1 = GATConv(
            in_channels,
            hidden_per_head,
            heads=heads,
            dropout=dropout,
            concat=True,
        )

        # Layer 2: single-head attention for final classification
        #   Input: hidden_per_head * heads (64)
        #   concat=False → AVERAGE the head outputs → shape: [nodes, out_channels]
        self.conv2 = GATConv(
            hidden_per_head * heads,
            out_channels,
            heads=1,
            dropout=dropout,
            concat=False,
        )

    def forward(self, x, edge_index):
        """
        Defines how data flows through the GAT.

        Note: dropout is applied to the raw INPUT features in GAT
        (unlike GCN where dropout sits between the two conv layers).
        This follows the original GAT paper (Veličković et al., 2018).

        Args:
            x          (Tensor): Node features. Shape [num_nodes, 1433].
            edge_index (Tensor): Graph edges in COO format. Shape [2, num_edges].

        Returns:
            Tensor: Raw class scores (logits). Shape [num_nodes, 7].
        """
        # Dropout on raw input features (GAT paper recommendation)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 1: multi-head attention aggregation
        x = self.conv1(x, edge_index)

        # ELU activation — smoother alternative to ReLU
        x = F.elu(x)

        # Dropout between layers
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2: final classification head
        x = self.conv2(x, edge_index)

        return x  # raw logits → CrossEntropyLoss handles softmax


# ──────────────────────────────────────────────────────────────
#  QUICK SANITY CHECK
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Verifies BOTH models initialise and run forward passes
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

    print(f"\n✅ GCN forward pass successful — shapes are correct.")

    # ── GAT sanity check ──────────────────────────────────────
    gat_model = GAT(
        in_channels=num_features,
        hidden_per_head=8,
        out_channels=num_classes,
        heads=8,
        dropout=0.6,
    )
    print(f"\nGAT architecture:\n{gat_model}\n")

    gat_model.eval()
    with torch.no_grad():
        gat_out = gat_model(x_dummy, edge_index_dummy)

    print(f"GAT Output shape: {gat_out.shape}  ← should be [2708, 7]")
    assert gat_out.shape == (num_nodes, num_classes), \
        f"Shape mismatch! Got {gat_out.shape}"

    print("\n✅ GAT forward pass successful — shapes are correct.")
    print("   Both models ready to be trained in train_node.py")
