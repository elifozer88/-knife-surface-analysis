# =============================================================================
# plot_ultimate_intersection_heatmap.py (EPS SUPPORT)
# Hochschule München / KIM — Erasmus+ Internship Project
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

print("=" * 100)
print("🎯 GENERATING BOTH DIAGNOSTIC HEATMAPS IN EPS FORMAT...")
print("=" * 100)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

# Paths
data_path = "data/processed/split_data.npz"
raw_data_path = "data/raw/Kochmesser_ohne_prozessdaten.xlsx"
output_fig_dir = os.path.join("outputs", "figures", "4_sensitivity_analysis")
os.makedirs(output_fig_dir, exist_ok=True)

if not os.path.exists(data_path):
    csv_fallback = "Kochmesser_ohne_prozessdaten.xlsx - Kochmesser_ohne_prozessdaten.csv"
    if os.path.exists(csv_fallback):
        raw_data_path = csv_fallback
    else:
        raise FileNotFoundError("Data files not found!")

# Load Data
data = np.load(data_path, allow_pickle=True)
df_raw = pd.read_excel(raw_data_path) if raw_data_path.endswith(".xlsx") else pd.read_csv(raw_data_path)

X_all = np.vstack([data["X_train"], data["X_test"]])
y_all = np.concatenate([data["y_clf_train"], data["y_clf_test"]])
knife_names = df_raw['Name'].values
lines = df_raw['Linie'].values

# =============================================================================
# MAP 1: ORIGINAL INTERSECTION (EPS)
# =============================================================================
splits = ["70_30_split", "80_20_split", "90_10_split"]
split_dfs = {}

for split in splits:
    file_path = os.path.join("outputs", split, "pruned", "test_knives_behavior_analysis.csv")
    if os.path.exists(file_path):
        df_sp = pd.read_csv(file_path)
        df_sp['Shared_Failure'] = (~df_sp['CART_Correct'] & ~df_sp['C45_Correct']).astype(int)
        split_dfs[split] = df_sp

if len(split_dfs) == 3:
    failed_knives_70_30 = set(split_dfs["70_30_split"][split_dfs["70_30_split"]['Shared_Failure'] == 1]['Knife_ID'].unique())
    failed_knives_80_20 = set(split_dfs["80_20_split"][split_dfs["80_20_split"]['Shared_Failure'] == 1]['Knife_ID'].unique())
    failed_knives_90_10 = set(split_dfs["90_10_split"][split_dfs["90_10_split"]['Shared_Failure'] == 1]['Knife_ID'].unique())

    shared_failed_knives = failed_knives_70_30.intersection(failed_knives_80_20).intersection(failed_knives_90_10)

    combined_list = []
    for split in splits:
        df_split = split_dfs[split]
        filtered_df = df_split[df_split['Knife_ID'].isin(shared_failed_knives)]
        combined_list.append(filtered_df)
        
    combined_df = pd.concat(combined_list)
    pivot_intersection = combined_df.pivot_table(index='Knife_ID', columns='Linie_Angle', values='Shared_Failure', aggfunc='max')

    fig, ax = plt.subplots(figsize=(10, max(5, len(pivot_intersection) * 0.35)), dpi=300)
    sns.heatmap(pivot_intersection, cmap=sns.color_palette(["#e0f2f1", "#d9534f"]), cbar=False, linewidths=0.5, linecolor='white', ax=ax)
    ax.set_title("Teeth-level Error Distribution: Shared Failures in All Splits (At least 1 failure)\n[Red (1) = Shared Failure | Light (0) = Correct/Partial Match]", fontsize=10, fontweight='bold', pad=15)
    ax.set_xlabel("Linie (Measurement Angle / Image Perspective)", fontsize=9, fontweight='bold')
    ax.set_ylabel("Knife ID (Intersection of 70/30, 80/20, 90/10)", fontsize=9, fontweight='bold')
    plt.tight_layout()
    
    # EPS Formatında Kaydetme (.eps uzantısı ve format parametresi) 🎯
    save_path_1 = os.path.join(output_fig_dir, "knife_angles_error_heatmap_intersection_any.eps")
    plt.savefig(save_path_1, format='eps', bbox_inches='tight')
    plt.close()
    print(f"✓ Map 1 Saved as EPS: {save_path_1}")

# =============================================================================
# MAP 2: ULTIMATE ABSOLUTE 10/10 FAILURES (EPS)
# =============================================================================
model = DecisionTreeClassifier(criterion="entropy", ccp_alpha=0.001, class_weight="balanced", random_state=42)
model.fit(X_all, y_all)
preds = model.predict(X_all)

results = pd.DataFrame({'Knife_ID': knife_names, 'Linie_Angle': lines, 'Actual': y_all, 'Pred': preds})
results['Shared_Failure'] = (results['Actual'] != results['Pred']).astype(int)

absolute_failed_knives = [
    "Koch-A0-T0-05_cropped_Messpunkt", "Koch-A11-T11-14_cropped_Messpunkt", "Koch-A19-T19-15_cropped_Messpunkt",
    "Koch-A2-T2-07_cropped_Messpunkt", "Koch-A2-T2-18_cropped_Messpunkt", "Koch-A21-T21-20_cropped_Messpunkt",
    "Koch-A22-T22-07_cropped_Messpunkt", "Koch-A24-T24-12_cropped_Messpunkt", "Koch-A24-T24-15_cropped_Messpunkt",
    "Koch-A26-T26-20_cropped_Messpunkt", "Koch-F1-08_cropped_Messpunkt", "Koch-VVS10-09_cropped_Messpunkt",
    "Koch-VVS16-11_cropped_Messpunkt", "Koch-VVS16-19_cropped_Messpunkt", "Koch-VVS19-04_cropped_Messpunkt",
    "Koch-VVS7-16_cropped_Messpunkt", "Koch-VVS7-18_cropped_Messpunkt"
]

pivot_absolute = results[results['Knife_ID'].isin(absolute_failed_knives)].pivot(index='Knife_ID', columns='Linie_Angle', values='Shared_Failure')

fig, ax = plt.subplots(figsize=(10, 7.5), dpi=300)
sns.heatmap(pivot_absolute, cmap=sns.color_palette(["#d9534f"]), cbar=False, linewidths=0.75, linecolor='white', ax=ax)
ax.set_title("Teeth-level Error Distribution: Absolute 10/10 Failures in Entire Dataset\n[Red (1) = Shared Failure | These 17 Knives Failed on Every Single Angle]", fontsize=10, fontweight='bold', pad=15)
ax.set_xlabel("Linie (Measurement Angle / Image Perspective)", fontsize=9, fontweight='bold')
ax.set_ylabel("Absolute Failed Knife ID (10/10 Errors)", fontsize=9, fontweight='bold')
plt.tight_layout()

# EPS Formatında Kaydetme 🎯
save_path_2 = os.path.join(output_fig_dir, "knife_angles_ultimate_absolute_10_10_heatmap.eps")
plt.savefig(save_path_2, format='eps', bbox_inches='tight')
plt.close()
