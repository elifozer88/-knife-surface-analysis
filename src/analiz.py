import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Prevent pop-ups
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Create Dummy Dataset
cols = ["review_count", "latitude", "longitude", "is_open"]
X_raw, y = make_classification(n_samples=500, n_features=4, n_informative=3, 
                               n_redundant=1, random_state=42)
X = pd.DataFrame(X_raw, columns=cols)

# 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define and Fit Models
cart_tree = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
cart_tree.fit(X_train, y_train)

c45_tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
c45_tree.fit(X_train, y_train)

# 3. Helper Function to Get Splits per Feature
def get_splits_dict(tree, feature_names):
    tree_ = tree.tree_
    splits = {feat: [] for feat in feature_names}
    def recurse(node):
        if tree_.feature[node] != -2: # If not a leaf node
            feat_name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            splits[feat_name].append(threshold)
            recurse(tree_.children_left[node])
            recurse(tree_.children_right[node])
    recurse(0)
    return splits

cart_splits = get_splits_dict(cart_tree, X.columns)
c45_splits = get_splits_dict(c45_tree, X.columns)

# 4. Plotting Histograms with Split Lines
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), dpi=200)
axes = axes.flatten()

for i, feat in enumerate(X.columns):
    ax = axes[i]
    
    # Plot data distribution grouped by target class
    for class_val, color, label in zip([0, 1], ['#e28743', '#1e81b0'], ['Class 0', 'Class 1']):
        subset = X_train[y_train == class_val][feat]
        ax.hist(subset, bins=25, alpha=0.6, color=color, label=label, edgecolor='white')
    
    # Draw CART splits (Red dashed lines)
    for split_val in cart_splits[feat]:
        cart_line = ax.axvline(split_val, color='#d9534f', linestyle='--', linewidth=2, 
                               label='CART Split' if 'CART Split' not in ax.get_legend_handles_labels()[1] else "")
        
    # Draw C4.5 splits (Blue dotted lines)
    for split_val in c45_splits[feat]:
        c45_line = ax.axvline(split_val, color='#0275d8', linestyle=':', linewidth=2.5, 
                               label='C4.5 Split' if 'C4.5 Split' not in ax.get_legend_handles_labels()[1] else "")
        
    ax.set_title(f"Distribution & Splits for {feat.upper()}", fontsize=12, fontweight='bold')
    ax.set_xlabel(feat)
    ax.set_ylabel("Count")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

# Create 'figures' directory if it doesn't exist
output_dir = "figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save figure
output_path = os.path.join(output_dir, "features_splits_histogram.png")
plt.savefig(output_path, bbox_inches='tight')
print(f"\n[INFO] Histogram plot successfully saved to: {output_path}\n")