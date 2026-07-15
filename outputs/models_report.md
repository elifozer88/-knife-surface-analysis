**Model Run Summary**

- **Cart Regression**: MAE 0.0263 μm, RMSE 0.0400 μm, R² 0.5474
  - Tree plot: [outputs/figures/regression_cart_tree.png](outputs/figures/regression_cart_tree.png)

- **Pruned Cart Regression**: MAE 0.0280 μm, R² 0.5066

- **CART (Gini) Classification**: accuracy 0.61, macro F1 ~0.58
  - Tree plot: [outputs/figures/gini_optimized_tree.png](outputs/figures/gini_optimized_tree.png)

- **C4.5 (Entropy) Classification**: accuracy 0.60, macro F1 ~0.58
  - Tree plot: [outputs/figures/c45_optimized_tree.png](outputs/figures/c45_optimized_tree.png)

- **Pruned Classifiers**: both reached ~0.57 accuracy; pruned confusion matrices and plots saved in `outputs/figures/` (see directory).

Files created:
- [outputs/metrics_summary.csv](outputs/metrics_summary.csv)
- [outputs/models_report.md](outputs/models_report.md)

Next steps: commit results, run additional hyperparameter sweeps, or generate a PDF report. Which would you like me to do?
