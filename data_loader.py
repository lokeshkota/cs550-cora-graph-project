"""
data_loader.py
--------------
Downloads the Cora citation network dataset using PyTorch Geometric
and prints key statistics about the graph.

Cora facts:
  - 2708 nodes  (scientific papers)
  - 5429 edges  (citation links, treated as undirected → 10556 directed)
  - 7 classes   (research topics: Neural_Networks, Rule_Learning, etc.)
  - 1433 features per node (bag-of-words word presence vector)
"""

from torch_geometric.datasets import Planetoid


def load_cora(root="./data"):
    """
    Downloads (on first run) and returns the Cora dataset.

    PyTorch Geometric caches the download inside `root/Cora/`.
    Subsequent calls use the cached version — no re-download needed.

    Args:
        root (str): Directory where the dataset will be stored.

    Returns:
        dataset: The full Planetoid dataset object (length 1 for Cora).
        data:    A PyG Data object — the single Cora graph.
                 Key attributes:
                   data.x            — Node feature matrix [2708 x 1433]
                   data.y            — Node labels         [2708]
                   data.edge_index   — Edge list (COO format) [2 x 10556]
                   data.train_mask   — Default train mask  [2708] bool
                   data.val_mask     — Default val mask    [2708] bool
                   data.test_mask    — Default test mask   [2708] bool
    """
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]  # Cora is a single-graph dataset

    print("=" * 45)
    print("         CORA DATASET STATISTICS")
    print("=" * 45)
    print(f"  Nodes (papers):          {data.num_nodes}")
    print(f"  Edges (citations):       {data.num_edges}")
    print(f"  Node feature dimensions: {dataset.num_node_features}")
    print(f"  Classes (topics):        {dataset.num_classes}")
    print("-" * 45)
    print(f"  Has isolated nodes:      {data.has_isolated_nodes()}")
    print(f"  Has self-loops:          {data.has_self_loops()}")
    print(f"  Is undirected:           {data.is_undirected()}")
    print("-" * 45)
    print("  Default PyG split (NOT used — we apply 80/20):")
    print(f"    Train nodes: {data.train_mask.sum().item():>5}  "
          f"({data.train_mask.sum().item() / data.num_nodes * 100:.1f}%)")
    print(f"    Val nodes:   {data.val_mask.sum().item():>5}  "
          f"({data.val_mask.sum().item() / data.num_nodes * 100:.1f}%)")
    print(f"    Test nodes:  {data.test_mask.sum().item():>5}  "
          f"({data.test_mask.sum().item() / data.num_nodes * 100:.1f}%)")
    print("=" * 45)

    return dataset, data


if __name__ == "__main__":
    dataset, data = load_cora()
    print("\n✅ data_loader.py ran successfully.")
