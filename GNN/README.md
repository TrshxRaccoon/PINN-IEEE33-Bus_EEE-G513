# Physics-Informed Edge-Attributed GCN for Power System State Estimation

This directory contains the Graph Neural Network (GNN) implementation for **Physics-Informed State Estimation on Distribution Grids** under sparse and missing PMU data (IEEE 33-bus system).

---

## Quickstart Commands

All commands should be executed from within the `GNN/` folder:

```powershell
# 1. Run Automated Pre-Flight Contract Checks (< 0.5s)
python check_gnn_contract.py --data ../MLP/data/ieee33

# 2. Train Clean Lambda Sweep (0.0 to 1000.0)
python train_gnn.py --data ../MLP/data/ieee33 --epochs 100 --out results_clean

# 3. Train Augmented Lambda Sweep (with dynamic on-the-fly PMU dropout)
python train_gnn.py --data ../MLP/data/ieee33 --epochs 100 --aug --out results_augmented

# 4. Evaluate Checkpoint on Test Sets (10% to 50% dropout)
python evaluate_gnn.py --checkpoint results_augmented/best_model_Augmented_seed42.pt --data ../MLP/data/ieee33

# 5. Generate Publication Plots (saved in plots/)
python plot_gnn_results.py

# 6. Interactive Single-Snapshot Comparison & Visualization Demo
python compare_models.py --condition Augmented --drop 50 --snapshot 0 --plot
```

For complete architectural details, mathematical formulations, and step-by-step guidance, please consult [GNN_OPERATING_GUIDE.md](file:///d:/BITS%20Pilani%20Goa%20Campus/Academics/ML4EE/ML_Project/GNN/GNN_OPERATING_GUIDE.md).
