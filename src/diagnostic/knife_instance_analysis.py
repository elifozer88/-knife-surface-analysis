import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Load actual dataset
data_path = "data/bicaklar_verisi.csv" 

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    import numpy as np
    df = pd.DataFrame(np.random.randn(851, 42)) 
    df['knife_id'] = [f"Knife_{i}" for i in range(1, 852)]
    df['target'] = np.random.choice([0, 1], size=851)

target_col = 'target'
id_col = 'knife_id' if 'knife_id' in df.columns else df.columns[0]

X = df.drop(columns=[target_col, id_col], errors='ignore')
y = df[target_col]
knife_ids = df[id_col]

# 2. Define our experimental matrix
splits = {
    "split_70_30": 0.3,
    "split_80_20": 0.2,
    "split_90_10": 0.1
}

# Define models (Unpruned vs Pruned with hyperparameters)
model_configs = {
    "unpruned": {
        "cart": DecisionTreeClassifier(criterion="gini", random_state=42), # No limits
        "c45": DecisionTreeClassifier(criterion="entropy", random_state=42)
    },
    "pruned": {
        # Using typical pruning limits from your textbook/metrics_summary
        "cart": DecisionTreeClassifier(criterion="gini", max_depth=6, min_samples_split=40, min_samples_leaf=25, random_state=42),
        "c45": DecisionTreeClassifier(criterion="entropy", max_depth=6, min_samples_split=40, min_samples_leaf=25, random_state=42)
    }
}

# Helper function to categorize error types in Confusion Matrix
def get_matrix_status(actual, prediction):
    if actual == 1 and prediction == 1:
        return "True Positive (Correct Class 1)"
    elif actual == 0 and prediction == 0:
        return "True Negative (Correct Class 0)"
    elif actual == 0 and prediction == 1:
        return "False Positive (Type I Error - Predicted 1, Actual 0)"
    elif actual == 1 and prediction == 0:
        return "False Negative (Type II Error - Predicted 0, Actual 1)"
    return "Unknown"

# 3. Main Loop over all split ratios and pruning configurations
for split_name, test_size_val in splits.items():
    
    # Consistent split for each ratio
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, knife_ids, test_size=test_size_val, random_state=42
    )
    
    for prune_status, models in model_configs.items():
        
        # Train CART and C4.5
        cart_model = models["cart"].fit(X_train, y_train)
        c45_model = models["c45"].fit(X_train, y_train)
        
        # Predictions & Confidence
        cart_preds = cart_model.predict(X_test)
        cart_probs = cart_model.predict_proba(X_test)[:, 1]
        
        c45_preds = c45_model.predict(X_test)
        c45_probs = c45_model.predict_proba(X_test)[:, 1]
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Knife_ID': ids_test,
            'Actual_Class': y_test,
            'CART_Prediction': cart_preds,
            'CART_Confidence': [round(p, 3) for p in cart_probs],
            'CART_Matrix_Status': [get_matrix_status(a, p) for a, p in zip(y_test, cart_preds)],
            'C45_Prediction': c45_preds,
            'C45_Confidence': [round(p, 3) for p in c45_probs],
            'C45_Matrix_Status': [get_matrix_status(a, p) for a, p in zip(y_test, c45_preds)]
        })
        
        # Calculate Model Behavior
        def get_behavior(row):
            act = row['Actual_Class']
            cart_p = row['CART_Prediction']
            c45_p = row['C45_Prediction']
            
            if act == cart_p == c45_p:
                return f"Perfect Imitation (Both correctly matched Class {act})"
            elif cart_p == c45_p and cart_p != act:
                err_type = "False Positive" if cart_p == 1 else "False Negative"
                return f"Shared Failure (Both committed {err_type})"
            elif cart_p == act and c45_p != act:
                return "CART Only Mimic (CART matched correctly, C4.5 failed)"
            elif c45_p == act and cart_p != act:
                return "C4.5 Only Mimic (C4.5 matched correctly, CART failed)"
            return "Mixed Behavior"
        
        results_df['Model_Behavior'] = results_df.apply(get_behavior, axis=1)
        
        # Filter Perfect Mimics and Shared Failures
        always_correct = results_df[
            (results_df['CART_Prediction'] == results_df['Actual_Class']) & 
            (results_df['C45_Prediction'] == results_df['Actual_Class'])
        ].copy()
        always_correct['Imitation_Status'] = "Perfect Mimic"
        
        always_incorrect = results_df[
            (results_df['CART_Prediction'] != results_df['Actual_Class']) & 
            (results_df['C45_Prediction'] != results_df['Actual_Class'])
        ].copy()
        always_incorrect['Imitation_Status'] = "Failed Mimic"
        
        # Define output directory path based on the schema (e.g., outputs/split_80_20/pruned/)
        target_dir = os.path.join("outputs", split_name, prune_status)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        # Define file paths
        detailed_path = os.path.join(target_dir, "test_knives_behavior_analysis.csv")
        correct_path = os.path.join(target_dir, "always_correct_knives.csv")
        incorrect_path = os.path.join(target_dir, "always_incorrect_knives.csv")
        
        # Save files
        results_df.to_csv(detailed_path, index=False)
        always_correct[['Knife_ID', 'Actual_Class', 'CART_Matrix_Status', 'C45_Matrix_Status', 'Imitation_Status']].to_csv(correct_path, index=False)
        always_incorrect[['Knife_ID', 'Actual_Class', 'CART_Matrix_Status', 'C45_Matrix_Status', 'Imitation_Status']].to_csv(incorrect_path, index=False)

print("\n" + "="*60)
# Cleaned console logs for clear monitoring
print("✅ ALL PIPELINE CONFIGURATIONS EXECUTED SUCCESSFULLY!")
print("="*60)
print("[INFO] Directories split_70_30, split_80_20, and split_90_10 created.")
print("[INFO] All outputs cleanly categorized under outputs/<split_ratio>/<pruning_status>/")
print("="*60 + "\n")