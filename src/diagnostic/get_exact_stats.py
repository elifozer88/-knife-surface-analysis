import os
import pandas as pd

# Load the test behavior file (Intersection analysis relies on these)
splits = ["70_30_split", "80_20_split", "90_10_split"]
split_dfs = {}

for split in splits:
    file_path = os.path.join("outputs", split, "pruned", "test_knives_behavior_analysis.csv")
    if os.path.exists(file_path):
        df_sp = pd.read_csv(file_path)
        df_sp['Shared_Failure'] = (~df_sp['CART_Correct'] & ~df_sp['C45_Correct']).astype(int)
        split_dfs[split] = df_sp

if len(split_dfs) == 3:
    # 1. Find the 18 shared failed knives
    failed_70 = set(split_dfs["70_30_split"][split_dfs["70_30_split"]['Shared_Failure'] == 1]['Knife_ID'].unique())
    failed_80 = set(split_dfs["80_20_split"][split_dfs["80_20_split"]['Shared_Failure'] == 1]['Knife_ID'].unique())
    failed_90 = set(split_dfs["90_10_split"][split_dfs["90_10_split"]['Shared_Failure'] == 1]['Knife_ID'].unique())
    shared_failed_knives = failed_70.intersection(failed_80).intersection(failed_90)

    # 2. Combine and pivot
    combined_df = pd.concat([split_dfs[split][split_dfs[split]['Knife_ID'].isin(shared_failed_knives)] for split in splits])
    pivot = combined_df.pivot_table(index='Knife_ID', columns='Linie_Angle', values='Shared_Failure', aggfunc='max')

    # 3. Calculate exact error sum for each column (0-9) (FIXED FORMATTING! 🎯)
    print("=" * 80)
    print("📊 EXACT ERROR COUNTS PER COLUMN FOR THE 18 KNIVES:")
    print("=" * 80)
    for col in pivot.columns:
        # Sum errors and fill NaN with 0, then convert to int to prevent ValueError
        error_count = int(pivot[col].fillna(0).sum())
        percentage = (error_count / len(pivot)) * 100
        # Convert col to int for formatting safety
        col_int = int(col)
        print(f"Column (Alt Sayı) {col_int:2d} (Linie {col_int+1:2d}) -> {error_count:2d} Errors / {len(pivot)} Knives ({percentage:.1f}%)")
    print("=" * 80)
else:
    print("Missing some split files to compute exact statistics.")