"""
main.py
-------
Master pipeline for the CS550 Cora Graph Analysis Project.

Running this single file reproduces ALL project results from scratch:
  1. Downloads the Cora dataset and prints statistics
  2. Trains the LR Baseline (node classification floor)
  3. Trains GCN  — node classification baseline model
  4. Trains GAT  — node classification main model
  5. Trains GAE  — link prediction baseline model
  6. Trains VGAE — link prediction main model
  7. Generates all comparison charts → outputs/
  8. Prints a final summary table of every result

Usage:
  python main.py

Note:
  - First run downloads Cora (~5MB) automatically.
  - Training takes ~2-3 minutes on a CPU.
  - Saved model weights go to  models/
  - Charts and results CSV go to outputs/
  - The Streamlit demo is launched separately:
      streamlit run app.py
"""

import time
import os


def separator(title):
    """Prints a clearly visible section header."""
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}\n")


def main():
    start_time = time.time()

    print("\n" + "█" * 60)
    print("  CS550 — Cora Graph Analysis Pipeline")
    print("  Massive Data Mining & Learning — Course Project")
    print("█" * 60)

    # ── STEP 1: Load Cora ─────────────────────────────────────
    separator("STEP 1 of 6 — Loading Cora Dataset")
    from data_loader import load_cora
    dataset, data = load_cora()

    # ── STEP 2: LR Baseline ───────────────────────────────────
    separator("STEP 2 of 6 — LR Baseline (Node Features Only)")
    print("Why: Establishes the performance floor before using any")
    print("     graph structure. GCN and GAT must both beat this.\n")
    from baseline_model import create_80_20_split, run_baseline
    train_mask_80, test_mask_80 = create_80_20_split(data)
    lr_clf, lr_pred, lr_true = run_baseline(data, train_mask_80, test_mask_80)
    lr_accuracy = (lr_pred == lr_true).mean()

    # ── STEP 3: GCN ───────────────────────────────────────────
    separator("STEP 3 of 6 — GCN Node Classification (Baseline GNN)")
    print("Why: Uses graph structure via equal neighbor averaging.")
    print("     Expected: ~88% accuracy, well above LR baseline.\n")
    from train_node import make_three_way_split, train_gcn, train_gat
    train_mask, val_mask, test_mask = make_three_way_split(data)
    gcn_results = train_gcn(data, train_mask, val_mask, test_mask)

    # ── STEP 4: GAT ───────────────────────────────────────────
    separator("STEP 4 of 6 — GAT Node Classification (Main GNN Model)")
    print("Why: Upgrades GCN with learned attention weights per neighbor.")
    print("     Same training loop as GCN — only the model changes.\n")
    gat_results = train_gat(data, train_mask, val_mask, test_mask)

    # ── STEP 5: Link Prediction ───────────────────────────────
    separator("STEP 5 of 6 — Link Prediction (GAE vs VGAE)")
    print("Why: Predicts missing citation links in the Cora graph.")
    print("     GAE = deterministic baseline | VGAE = probabilistic model.\n")
    from train_link import run_link_prediction
    link_results = run_link_prediction(seed=42)

    # ── STEP 6: Charts ────────────────────────────────────────
    separator("STEP 6 of 6 — Generating Comparison Charts")
    print("Saving charts to outputs/ for the report...\n")
    from plot import generate_all_plots
    generate_all_plots()

    # ── FINAL SUMMARY ─────────────────────────────────────────
    elapsed = time.time() - start_time
    separator("FINAL RESULTS SUMMARY")

    print(f"  {'Model':<18} {'Task':<22} {'Accuracy':>10} {'F1':>8} {'AUC-ROC':>9}")
    print(f"  {'-'*68}")

    # LR Baseline
    print(f"  {'LR Baseline':<18} {'Node Classification':<22} "
          f"{lr_accuracy*100:>9.2f}%  {'—':>8} {'—':>9}")

    # GCN
    print(f"  {'GCN':<18} {'Node Classification':<22} "
          f"{gcn_results['accuracy']*100:>9.2f}%  "
          f"{gcn_results['f1']:>8.4f} {'—':>9}")

    # GAT
    print(f"  {'GAT':<18} {'Node Classification':<22} "
          f"{gat_results['accuracy']*100:>9.2f}%  "
          f"{gat_results['f1']:>8.4f} {'—':>9}")

    # Link prediction results from results.csv
    if link_results:
        for model_name, lres in link_results.items():
            print(f"  {model_name:<18} {'Link Prediction':<22} "
                  f"{'—':>10}  "
                  f"{lres.get('f1', 0):>8.4f} "
                  f"{lres.get('auc_roc', 0):>9.4f}")

    print(f"\n  Total runtime: {elapsed:.1f} seconds")
    print(f"  Saved weights: models/gcn_best.pth, gat_best.pth,")
    print(f"                 models/gae_best.pth, vgae_best.pth")
    print(f"  Charts saved:  outputs/")
    print(f"\n  To launch the interactive demo:")
    print(f"    streamlit run app.py")
    print(f"\n{'█' * 60}")
    print(f"  ✅ Pipeline complete.")
    print(f"{'█' * 60}\n")


if __name__ == "__main__":
    main()
