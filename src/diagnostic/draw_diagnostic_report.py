# =============================================================================
# draw_diagnostic_report.py
# Real Data — Champion vs. Cursed Boundary Proximity Diagnostics
# Hochschule München / KIM — Erasmus+ Internship Project
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Securely import seaborn for publication-quality statistical plots
try:
    import seaborn as sns
except ImportError:
    raise ImportError("Please install seaborn via: pip install seaborn")

print("=" * 60)
print("STARTING BOUNDARY PROXIMITY DIAGNOSTIC ANALYSIS...")
print("=" * 60)

# 1. ACQUIRE INSTANCE-LEVEL EVALUATION REPORT
report_path = "data/processed/knife_instance_accuracy_report.csv"
if not os.path.exists(report_path):
    raise FileNotFoundError(
        f"CRITICAL ERROR: '{report_path}' not found. "
        f"Please run the instance evaluation script before executing diagnostics."
    )

df_knives = pd.read_csv(report_path)

# 2. CATEGORIZE SPECIMENS BASED ON PREDICTION ACCURACY
# Champion: Models achieved 100% classification accuracy on all profiles of the knife.
# Cursed: Models failed on every single profile (0% accuracy), indicating persistent boundary errors.
df_knives['Category'] = 'Transition/Borderline Knives'
df_knives.loc[df_knives['Accuracy_Rate'] == 1.0, 'Category'] = 'Champion (100% Accurate)'
df_knives.loc[df_knives['Accuracy_Rate'] == 0.0, 'Category'] = 'Cursed (0% Accurate)'

# Isolate the extreme performance cohorts for comparative boundary testing
df_plot = df_knives[df_knives['Category'].isin(['Champion (100% Accurate)', 'Cursed (0% Accurate)'])].copy()

# 3. CONSTRUCT PUBLICATION-QUALITY PLOTS
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 11), dpi=300)

# Palette mapping matching the model evaluation standards (Green for accurate, Red for errors)
category_palette = {
    'Champion (100% Accurate)': '#2ecc71', 
    'Cursed (0% Accurate)': '#e74c3c'
}

# -----------------------------------------------------------------------------
# PLOT 1: HISTOGRAM WITH KERNEL DENSITY ESTIMATION (KDE)
# -----------------------------------------------------------------------------
sns.histplot(
    data=df_plot, 
    x='True_Ra', 
    hue='Category', 
    kde=True, 
    bins=30, 
    palette=category_palette, 
    alpha=0.6, 
    ax=ax1
)

# Plot academic specification limits from the paper (Table 1 boundaries)
ax1.axvline(x=0.11, color='#34495e', linestyle='--', linewidth=1.5, label='Class 0/1 Boundary (0.11 μm)')
ax1.axvline(x=0.45, color='#34495e', linestyle='-.', linewidth=1.5, label='Class 1/2 Boundary (0.45 μm)')

ax1.set_title("True Ra Distribution of Champion vs. Cursed Specimens (Boundary Effect Diagnostics)", fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel("Physical Surface Roughness Ra (μm)", fontsize=11, labelpad=8)
ax1.set_ylabel("Profile Sample Count", fontsize=11, labelpad=8)
ax1.legend(loc='upper right', frameon=True, shadow=True)

# -----------------------------------------------------------------------------
# PLOT 2: PROXIMITY BOXPLOT (DISTANCE TO NEAREST DECISION BOUNDARY)
# -----------------------------------------------------------------------------
# Calculate absolute distance of each knife's true Ra to the closest decision boundary (0.11 or 0.45)
df_plot['Distance_to_Boundary'] = df_plot['True_Ra'].apply(lambda x: min(abs(x - 0.11), abs(x - 0.45)))

sns.boxplot(
    data=df_plot, 
    x='Category', 
    y='Distance_to_Boundary', 
    palette=category_palette, 
    width=0.4, 
    ax=ax2
)

ax2.set_title("Specimen Distance to Decision Boundaries (0.11 μm & 0.45 μm Limit Zones)", fontsize=13, fontweight='bold', pad=12)
ax2.set_ylabel("Distance to Nearest Decision Boundary (μm)", fontsize=11, labelpad=8)
ax2.set_xlabel("")
ax2.set_xticklabels(['Champion (100% Accurate)', 'Cursed (0% Accurate)'], fontsize=11, fontweight='semibold')

# 4. SERIALIZE GRAPHICAL DIAGNOSTICS TO DISK
plt.tight_layout()

# Establish target output directory in a systematic manner
output_dir = os.path.join("outputs", "figures", "1_data_diagnostics")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "Fig1_boundary_diagnostics.png")
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close(fig)

print(f"✓ Boundary proximity diagnostics saved successfully to: '{output_path}'")