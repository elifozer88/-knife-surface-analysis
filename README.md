# Knife Surface Analysis

A small project for analyzing knife surface images and deriving roughness/condition labels using CART decision trees. The repository contains data preparation, model training (classification and regression), and visualization utilities.

**Quick Overview**
- **Data:** [data/raw](data/raw) (original) and [data/processed](data/processed) (cleaned, feature matrix and splits)
- **Code:** [src](src) — model training scripts and utilities
- **Notebooks:** [notebooks](notebooks) — exploratory analysis
- **Outputs:** [outputs](outputs) — figures, confusion matrices, and exported tables

**Requirements**
Install Python packages from `requirements.txt` (recommended in a venv):

```bash
python -m venv venv
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

**How to run the main scripts**
- Train and evaluate CART regression (saves tree plot):

```bash
python src/cart_regressor.py
```

- Train and evaluate CART classifier (Gini):

```bash
python src/cart_model.py
```

- Train and evaluate entropy-based (C4.5-like) classifier:

```bash
python src/c45_model.py
```

**Pruned models and analysis**
- Pruned variants are under `src/pruned/` and can be run directly, e.g.:

```bash
python src/pruned/cart_model_pruned.py
python src/pruned/c45_model_pruned.py
python src/pruned/cart_regressor_pruned.py
```

- A small analysis helper that computes a binned true-vs-predicted matrix for the pruned regressor is at `src/analysis/regression_pruned_matrix.py`.

**Outputs produced**
- Model figures: `outputs/figures/` (tree images, confusion matrices, heatmaps)
- Summary metrics: `outputs/metrics_summary.csv`
- Prediction tables: `outputs/tables/`

**Notes**
- All cross-validation is group-aware (knife groups) to avoid data leakage. Classification uses `StratifiedGroupKFold` where applicable.

If you want the README tailored (Turkish, extra usage examples, or badges), tell me which sections to change.
