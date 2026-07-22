import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

data_path = "data/processed/split_data.npz"
raw_data_path = "data/raw/Kochmesser_ohne_prozessdaten.xlsx"

if not os.path.exists(data_path):
    raise FileNotFoundError("Processed split data file missing.")

data = np.load(data_path, allow_pickle=True)
X_train = data["X_train"]
y_train = data["y_clf_train"]

if os.path.exists(raw_data_path):
    df_raw = pd.read_excel(raw_data_path)
else:
    csv_path = "Kochmesser_ohne_prozessdaten.xlsx - Kochmesser_ohne_prozessdaten.csv"
    df_raw = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

model = DecisionTreeClassifier(criterion="entropy", ccp_alpha=0.001, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
importances = model.feature_importances_

if df_raw is not None:
    feature_names = [c for c in df_raw.columns if c not in ['Name', 'Linie', 'Class', 'Sınıf', 'Knife_ID', 'ID']]
    feature_names = feature_names[:X_train.shape[1]]
else:
    feature_names = [f"Parameter {i}" for i in range(X_train.shape[1])]

feat_imp_df = pd.DataFrame({
    'Parameter': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
bars = ax.barh(feat_imp_df['Parameter'][::-1], feat_imp_df['Importance'][::-1], color='#2c5282', edgecolor='#1a365d', height=0.55, zorder=3)

for bar in bars:
    width = bar.get_width()
    if width > 0:
        ax.annotate(f'{width*100:.1f}%',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=9, fontweight='bold', color='#1a202c')

ax.set_title("Quantitative Hierarchy of Surface Roughness Parameters in Decision Tree Classification", fontsize=11, fontweight='bold', pad=18)
ax.set_xlabel("Relative Feature Importance Score (Information Gain Weight)", fontsize=9, fontweight='bold', labelpad=10)
ax.set_ylabel("Surface Texture Parameters (3D Roughness Descriptors)", fontsize=9, fontweight='bold', labelpad=10)

ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

output_fig_dir = os.path.join("outputs", "figures", "4_sensitivity_analysis")
os.makedirs(output_fig_dir, exist_ok=True)

plt.savefig(os.path.join(output_fig_dir, "model_feature_importance_histogram.png"), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(output_fig_dir, "model_feature_importance_histogram.eps"), format='eps', bbox_inches='tight')
plt.close()