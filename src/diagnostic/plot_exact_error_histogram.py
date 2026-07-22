# =============================================================================
# plot_exact_error_histogram.py (EPS SUPPORT)
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 100)
print("📊 GENERATING ACADEMIC HISTOGRAM IN EPS FORMAT...")
print("=" * 100)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

splits = ["70_30_split", "80_20_split", "90_10_split"]
split_dfs = {}

for split in splits:
    file_path = os.path.join("outputs", split, "pruned", "test_knives_behavior_analysis.csv")
    if os.path.exists(file_path):
        df_sp = pd.read_csv(file_path)
        df_sp['Shared_Failure'] = (~df_sp['CART_Correct'] & ~df_sp['C45_Correct']).astype(int)
        split_dfs[split] = df_sp

if len(split_dfs) == 3:
    failed_70 = set(split_dfs["70_30_split"][split_dfs["70_30_split"]['Shared_Failure'] == 1]['Knife_ID'].unique())
    failed_80 = set(split_dfs["80_20_split"][split_dfs["80_20_split"]['Shared_Failure'] == 1]['Knife_ID'].unique())
    failed_90 = set(split_dfs["90_10_split"][split_dfs["90_10_split"]['Shared_Failure'] == 1]['Knife_ID'].unique())
    shared_failed_knives = failed_70.intersection(failed_80).intersection(failed_90)

    combined_df = pd.concat([split_dfs[split][split_dfs[split]['Knife_ID'].isin(shared_failed_knives)] for split in splits])
    pivot = combined_df.pivot_table(index='Knife_ID', columns='Linie_Angle', values='Shared_Failure', aggfunc='max')

    columns = [f"Col {int(c)}\n(Linie {int(c)+1})" for c in pivot.columns]
    error_counts = [int(pivot[c].fillna(0).sum()) for c in pivot.columns]
    percentages = [(count / len(pivot)) * 100 for count in error_counts]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.bar(columns, error_counts, color="#d9534f", edgecolor="#b52b27", width=0.6, zorder=3)
    
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(f'{height}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')

    ax.set_title("Error Distribution Across Measurement Perspectives (N = 18 Failed Knives)", fontsize=12, fontweight='bold', pad=20)
    ax.set_xlabel("Linie (Measurement Angle / Image Perspective)", fontsize=10, fontweight='bold', labelpad=10)
    ax.set_ylabel("Number of Failures (Total: 18 Knives)", fontsize=10, fontweight='bold', labelpad=10)
    
    ax.set_ylim(0, 14)
    ax.set_yticks(range(0, 15, 2))
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    output_fig_dir = os.path.join("outputs", "figures", "4_sensitivity_analysis")
    os.makedirs(output_fig_dir, exist_ok=True)
    
    # EPS Formatında Kaydetme 🎯
    save_path = os.path.join(output_fig_dir, "knife_angles_error_histogram.eps")
    plt.savefig(save_path, format='eps', bbox_inches='tight')
    plt.close()
