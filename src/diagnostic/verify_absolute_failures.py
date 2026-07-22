import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Load everything cleanly
data_path = "data/processed/split_data.npz"
raw_data_path = "data/raw/Kochmesser_ohne_prozessdaten.xlsx" # adjust if CSV

if os.path.exists(data_path) and os.path.exists(raw_data_path):
    data = np.load(data_path, allow_pickle=True)
    df_raw = pd.read_excel(raw_data_path) if raw_data_path.endswith(".xlsx") else pd.read_csv(raw_data_path)
    
    X_all = np.vstack([data["X_train"], data["X_test"]])
    y_all = np.concatenate([data["y_clf_train"], data["y_clf_test"]])
    knife_names = df_raw['Name'].values
    lines = df_raw['Linie'].values
    
    # Train champion model on ALL data to find absolute, unresolvable failures
    model = DecisionTreeClassifier(criterion="entropy", ccp_alpha=0.001, class_weight="balanced", random_state=42)
    model.fit(X_all, y_all)
    preds = model.predict(X_all)
    
    results = pd.DataFrame({
        'Knife_ID': knife_names,
        'Linie': lines,
        'Actual': y_all,
        'Pred': preds
    })
    results['Correct'] = results['Actual'] == results['Pred']
    
    # Group by knife and count total errors out of 10 lines
    knife_summary = results.groupby('Knife_ID').agg(
        Total_Lines=('Linie', 'count'),
        Error_Count=('Correct', lambda x: (x == False).sum())
    ).reset_index()
    
    # Find knives with 10/10 errors
    absolute_failures = knife_summary[knife_summary['Error_Count'] == 10]
    
    print("="*80)
    print("🚨 ABSOLUTE 10/10 ERROR KNIVES IN ENTIRE DATASET:")
    print("="*80)
    if len(absolute_failures) > 0:
        print(absolute_failures.to_string(index=False))
    else:
        print("No knife has 10/10 absolute errors across the whole dataset.")
    print("="*80)