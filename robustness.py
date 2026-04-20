"""
robustness.py
-------------
Stress-testing module for GNN models. 
Simulates structural (edge) and semantic (feature) attacks.
"""

import torch
import numpy as np
from typing import List, Dict
from copy import deepcopy

def drop_edges(edge_index: torch.Tensor, drop_rate: float, seed: int = 42) -> torch.Tensor:
    """
    Randomly removes a percentage of edges from the graph.
    
    Args:
        edge_index: COO format edges [2, E]
        drop_rate:  Fraction of edges to remove (0.0 to 1.0)
    """
    if drop_rate <= 0:
        return edge_index
        
    num_edges = edge_index.size(1)
    num_to_keep = int(num_edges * (1.0 - drop_rate))
    
    torch.manual_seed(seed)
    indices = torch.randperm(num_edges)[:num_to_keep]
    return edge_index[:, indices]

def flip_features(x: torch.Tensor, flip_rate: float, seed: int = 42) -> torch.Tensor:
    """
    Randomly flips 0/1 bits in the binary feature matrix.
    
    Args:
        x:         Feature matrix [N, F]
        flip_rate: Fraction of bits to flip
    """
    if flip_rate <= 0:
        return x
        
    x_new = deepcopy(x)
    torch.manual_seed(seed)
    
    # Generate mask of bits to flip
    mask = torch.rand(x_new.shape) < flip_rate
    # Flip: 1-1=0, 1-0=1
    x_new[mask] = 1.0 - x_new[mask]
    return x_new

@torch.no_grad()
def run_stress_test(model, data, attack_type="structural", rates=[0.0, 0.05, 0.1, 0.2, 0.3]):
    """
    Evaluates model accuracy across different noise levels.
    """
    model.eval()
    # ── PyG Fix: Ensure explain mode is DISABLED ───────────────────────────
    # If the model was used for xAI earlier, it might have .explain=True
    # and stale edge masks, causing an AssertionError on new edge_index.
    for module in model.modules():
        if hasattr(module, 'explain'):
            module.explain = False
    
    results = []
    
    test_mask = data.test_mask
    y_true = data.y[test_mask]
    
    for rate in rates:
        if attack_type == "structural":
            edge_index_atk = drop_edges(data.edge_index, rate)
            out = model(data.x, edge_index_atk)
        else:
            x_atk = flip_features(data.x, rate)
            out = model(x_atk, data.edge_index)
            
        preds = out[test_mask].argmax(dim=1)
        acc = (preds == y_true).float().mean().item()
        results.append({"rate": rate, "accuracy": acc})
        
    return results

if __name__ == "__main__":
    # Small self-test
    print("Testing robustness logic...")
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
    dropped = drop_edges(edges, 0.5)
    print(f"Original edges: {edges.size(1)}, Dropped: {dropped.size(1)}")
    
    x = torch.zeros((10, 10))
    flipped = flip_features(x, 0.1)
    print(f"Flipped bits: {flipped.sum().item()}")
