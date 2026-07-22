# =============================================================================
# plot_angle_heatmap.py (MULTI-SPLIT GENERATOR)
# Generates Heatmaps for 70/30, 80/20, and 90/10 scenarios
# Hochschule München / KIM — Erasmus+ Internship Project
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Prevent GUI pop-ups
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 100)
print("📊 GENERATING ACADEMIC HEATMAPS FOR ALL SPLIT CONFIGURATIONS...")
print("=" * 100)

# --- Academic Font Config (Times New Roman) ---
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

# Target splits we want to generate plots for
splits_to_plot = [
    {"dir_name": "70_30_split", "label": "70/30 Split"},
    {"dir_name": "80_20_split", "label": "80/20 Split"},
    {"dir_name": "90_10_split", "label": "90/10 Split"}
]

output_fig_dir = os.path.join("outputs", "figures", "4_sensitivity_analysis")
os.makedirs(output_fig_dir, exist_ok=True)

# Loop over each split scenario and generate its unique heatmap
for split in splits_to_plot:
    analysis_dir = os.path.join("outputs", split["dir_name"], "pruned")
    behavior_file = os.path.join(analysis_dir, "test_knives_behavior_analysis.csv")
    
    if not os.path.exists(behavior_file):
        print(f"[WARNING] Skipping {split['label']} - behavior file not found at: {behavior_file}")
        continue
        
    # 1. Load data
    df = pd.read_csv(behavior_file)
    df['Shared_Failure'] = (~df['CART_Correct'] & ~df['C45_Correct']).astype(int)
    
    # 2. Pivot
    pivot_df = df.pivot(index='Knife_ID', columns='Linie_Angle', values='Shared_Failure')
    
    # Filter only failed knives to keep plots clean and focused
    failed_knives_mask = pivot_df.sum(axis=1) > 0
    pivot_df_filtered = pivot_df[failed_knives_mask]
    
    if len(pivot_df_filtered) == 0:
        print(f"[INFO] No shared failures found for {split['label']}.")
        continue
        
    # 3. Plotting
    fig_height = max(6, len(pivot_df_filtered) * 0.3)
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=300)
    
    cmap = sns.color_palette(["#e0f2f1", "#d9534f"]) # Teal (0) and Coral Red (1)
    
    sns.heatmap(
        pivot_df_filtered,
        cmap=cmap,
        cbar=False,
        linewidths=0.5,
        linecolor='white',
        yticklabels=True,
        xticklabels=True,
        ax=ax
    )
    
    # Customize titles with specific split labels
    ax.set_title(f"Teeth-level Error Distribution: Shared Failures across Knife Angles (Linie) - {split['label']}\n[Red (1) = Shared Failure | Light (0) = Correct/Partial Match]", 
                 fontsize=11, fontweight='bold', pad=15)
    ax.set_xlabel("Linie (Measurement Angle / Image Perspective)", fontsize=10, labelpad=10, fontweight='bold')
    ax.set_ylabel("Knife ID (Only showing knives with failures)", fontsize=10, labelpad=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save with unique filename matching the split
    save_filename = f"knife_angles_error_heatmap_{split['dir_name']}.png"
    save_path = os.path.join(output_fig_dir, save_filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Heatmap for {split['label']} saved to: '{save_path}'")

print("=" * 100 + "\n")