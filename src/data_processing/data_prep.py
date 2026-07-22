# =============================================================================
# data_prep.py
# Knife Surface Analysis — Data Preprocessing & Partitioning Pipeline
# Hochschule München / KIM — Erasmus+ Internship Project
# =============================================================================

# -----------------------------------------------------------------------------
# LIBRARIES
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import os

# =============================================================================
# STEP 1: DATA ACQUISITION & LOADING
# =============================================================================
print("=" * 60)
print("STEP 1: Loading Raw Experimental Dataset...")
print("=" * 60)

FILE_PATH = "data/raw/Kochmesser_ohne_prozessdaten.xlsx"

# Validate the existence of the raw data file before initiating read operations
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        f"CRITICAL ERROR: '{FILE_PATH}' not found. "
        f"Please verify that the raw Excel file is placed in the 'data/raw/' directory."
    )

# Load the spreadsheet utilizing the openpyxl engine
df = pd.read_excel(FILE_PATH, engine="openpyxl")

print(f"✓ Dataset loaded successfully.")
print(f"  Total records (rows)    : {df.shape[0]}")
print(f"  Total attributes (cols) : {df.shape[1]}")


# =============================================================================
# STEP 2: ATTRIBUTE & TARGET DEFINITIONS
# =============================================================================
# The experimental dataset is structured into three primary segments:
#
#   1. IDENTIFICATION COLUMNS:
#      - Spalte1 : Row sequence index (excluded from modeling)
#      - Name    : Knife Identifier (mapped as 'knife_id' for group tracking)
#      - Linie   : Measurement line profile index (0 to 9)
#
#   2. FEATURE COLUMNS (42 variables, Columns D to AT / indices 3 to 44):
#      - Indices [3:15]  : 12 vertical roughness/depth parameters (e.g., Ra, Rq, Rz)
#      - Indices [15:45] : 30 horizontal, frequency, and textural parameters
#
#   3. TARGET VARIABLES:
#      - AT Column (Ra3) : Discrete classification target (Classes 0, 1, 2)
#      - AU Column (Ra)  : Continuous regression target (roughness value in μm)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Defining Model Features and Target Variables...")
print("=" * 60)

# Establish unique key for Group-based splitting
KNIFE_ID_COL = "Name"

# Slice feature matrix: Column index 3 to 45 represents the 42 physical attributes
FEATURE_COLS = df.columns[3:45].tolist()

# Define experimental targets
TARGET_CLF = "Ra3"   # Classification target (Predictive quality classes)
TARGET_REG = "Ra"    # Regression target (Continuous surface roughness prediction)

print(f"✓ Group Identifier Column : {KNIFE_ID_COL}")
print(f"✓ Total Selected Features : {len(FEATURE_COLS)}")
print(f"✓ Classification Target   : {TARGET_CLF}")
print(f"✓ Regression Target       : {TARGET_REG}")
print(f"\n  Attribute Feature List:")
for i, col in enumerate(FEATURE_COLS, 1):
    print(f"    [{i:2d}] {col}")


# =============================================================================
# STEP 3: DATA QUALITY ASSURANCE & DESCRIPTIVE STATISTICAL ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Data Quality & Integrity Checks")
print("=" * 60)

# Audit feature matrix for missing values (Null/NaN)
missing_features = df[FEATURE_COLS].isnull().sum()
missing_features = missing_features[missing_features > 0]

if len(missing_features) == 0:
    print("✓ Data Integrity: No missing values detected across all 42 feature columns.")
else:
    print("⚠ WARNING: Missing values identified in features:")
    print(missing_features)

# Audit target vectors for missing values
missing_targets = df[[TARGET_CLF, TARGET_REG]].isnull().sum()
print(f"\n✓ Target Integrity Audit:")
print(f"  {TARGET_CLF} (Class)      : {missing_targets[TARGET_CLF]} missing value(s)")
print(f"  {TARGET_REG} (Regression) : {missing_targets[TARGET_REG]} missing value(s)")

# Analyze profile distribution per specimen (Strictly 10 lines per knife)
knife_counts = df[KNIFE_ID_COL].value_counts()
print(f"\n✓ Measurement Line Distribution per Specimen:")
print(f"  Unique Knife Specimens : {knife_counts.nunique() if hasattr(knife_counts, 'nunique') else len(knife_counts)}")
print(f"  Profile Counts/Knife   : min={knife_counts.min()}, max={knife_counts.max()}, mean={knife_counts.mean():.1f}")

# Descriptive distribution of target classes (Class Imbalance Diagnostics)
print(f"\n✓ Class Distribution for target '{TARGET_CLF}':")
class_dist = df[TARGET_CLF].value_counts().sort_index()
for cls, count in class_dist.items():
    pct = count / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"  Class {cls}: {count:5d} profiles ({pct:.1f}%) {bar}")

print("\n  ⚠ METHODOLOGICAL NOTE: Highly skewed class distribution (Class Imbalance) observed.")
print("    → Imbalance mitigation: Use 'class_weight=\"balanced\"' in tree models.")

# Continuous target metrics
print(f"\n✓ Continuous Target '{TARGET_REG}' Descriptive Statistics:")
print(df[TARGET_REG].describe().to_string())


# =============================================================================
# STEP 4: FEATURE MATRIX & TARGET VECTOR GENERATION
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Generating Input Matrices and Target Vectors...")
print("=" * 60)

X = df[FEATURE_COLS].values          # Feature Matrix (Shape: 8510, 42)
y_clf = df[TARGET_CLF].values        # Classification Vector (Shape: 8510,)
y_reg = df[TARGET_REG].values        # Regression Vector (Shape: 8510,)
groups = df[KNIFE_ID_COL].values     # Group Tracker (Shape: 8510,)

print(f"✓ X Matrix Shape      : {X.shape} (Samples x Features)")
print(f"✓ y_clf Vector Shape  : {y_clf.shape}")
print(f"✓ y_reg Vector Shape  : {y_reg.shape}")
print(f"✓ Group Vector Shape  : {groups.shape}")


# =============================================================================
# STEP 5: GROUP-BASED TRAIN/TEST PARTITIONING (DATA LEAKAGE PREVENTION)
# =============================================================================
# To evaluate generalized model performance accurately on unseen physical objects:
#
#   - Traditional random splitting is NOT applicable due to multiple profiles (10)
#     originating from the same physical specimen.
#   - Placing profiles of the same knife in both train and test partitions
#     induces severe "Data Leakage," leading to overoptimistic performance.
#   - GroupShuffleSplit isolates entire knife specimens.
#     All 10 profiles of a single specimen are assigned strictly to either 
#     the training or testing set.
#
# Partition ratio: 80% Training (~681 knives), 20% Testing (~170 knives)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Executing Group-Based Train/Test Split")
print("=" * 60)
print("  (GroupShuffleSplit configured to prevent cross-profile data leakage)")

gss = GroupShuffleSplit(
    n_splits=1,       # Single partition generation
    test_size=0.2,    # 20% Holdout evaluation set
    random_state=42   # Deterministic seed for reproducible partitioning
)

# Extract partition indices
train_idx, test_idx = next(gss.split(X, y_clf, groups=groups))

# Split matrices based on generated indices
X_train, X_test         = X[train_idx],     X[test_idx]
y_clf_train, y_clf_test = y_clf[train_idx], y_clf[test_idx]
y_reg_train, y_reg_test = y_reg[train_idx], y_reg[test_idx]

# Calculate unique specimen allocations
train_knives = len(set(groups[train_idx]))
test_knives  = len(set(groups[test_idx]))

print(f"\n✓ Training Partition:")
print(f"  Samples: {X_train.shape[0]} ({train_knives} unique knives x 10 profiles)")
print(f"✓ Testing Partition (Holdout):")
print(f"  Samples: {X_test.shape[0]} ({test_knives} unique knives x 10 profiles)")

# Verification audit to confirm strict partitioning (No overlapping groups)
train_knife_set = set(groups[train_idx])
test_knife_set  = set(groups[test_idx])
overlap = train_knife_set & test_knife_set
print(f"\n✓ Overlap Audit: {len(overlap)} shared knife specimens detected.")
if len(overlap) == 0:
    print("  ✓ Verification Pass: Strict specimen isolation achieved. No data leakage.")
else:
    print("  ⚠ CRITICAL WARNING: Overlap detected. Partitioning logic compromised.")

print(f"\n✓ Training Partition Class Balance ({TARGET_CLF}):")
train_dist = pd.Series(y_clf_train).value_counts().sort_index()
for cls, count in train_dist.items():
    pct = count / len(y_clf_train) * 100
    print(f"  Class {cls}: {count:5d} ({pct:.1f}%)")


# =============================================================================
# STEP 6: CROSS-VALIDATION SCHEME SETUP (GROUPKFOLD)
# =============================================================================
# To avoid overfitting during parameter optimization, we define a 5-fold 
# cross-validation process restricted strictly inside the training partition.
#
# GroupKFold ensures that no knife specimen is split across folds.
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Initializing Group-Based Cross-Validation Setup")
print("=" * 60)

cv = GroupKFold(n_splits=5)
train_groups = groups[train_idx]

print(f"✓ GroupKFold(n_splits=5) initialized.")
print(f"  Specimens/fold: ~{train_knives // 5} unique knives in validation split.")
print(f"  Usage Syntax  : cv.split(X_train, y_clf_train, groups=train_groups)")
print(f"\n  Descriptive Fold Dimensions:")
for fold_num, (tr, val) in enumerate(cv.split(X_train, y_clf_train, groups=train_groups), 1):
    print(f"    Fold {fold_num}: Train={len(tr)} samples, Validation={len(val)} samples "
          f"({len(set(train_groups[val]))} unique knives)")


# =============================================================================
# STEP 7: SERIALIZING PROCESSED DATASETS
# =============================================================================
# The preprocessed features, targets, and group mappings are serialized
# into a compressed NumPy format (.npz) to enable modular modeling scripts
# to consume processed data directly without re-reading the Excel file.
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Serializing Processed Partitions to Disk...")
print("=" * 60)

# Ensure processed directory exists physically
processed_data_dir = os.path.join("data", "processed")
os.makedirs(processed_data_dir, exist_ok=True)

# Serialize arrays into a single compressed package
npz_path = os.path.join(processed_data_dir, "split_data.npz")
np.savez(
    npz_path,
    X_train=X_train,
    X_test=X_test,
    y_clf_train=y_clf_train,
    y_clf_test=y_clf_test,
    y_reg_train=y_reg_train,
    y_reg_test=y_reg_test,
    train_groups=train_groups,
    test_groups=groups[test_idx]
)

# Export feature column names for diagnostic model interpretability
csv_path = os.path.join(processed_data_dir, "feature_names.csv")
pd.Series(FEATURE_COLS).to_csv(csv_path, index=False)

print(f"✓ Partition arrays saved successfully to: '{npz_path}'")
print(f"✓ Feature index labels saved successfully to : '{csv_path}'")