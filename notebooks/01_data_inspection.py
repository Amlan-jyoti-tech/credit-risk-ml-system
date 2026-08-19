"""
01_data_inspection.py
====================
P0.2: Initial Data Inspection of the Lending Club Dataset.

Source: OpenML Dataset ID 43729 (Lending Club Loan Data, 2007-2015 era)
License: CC0 Public Domain

This script inspects the raw dataset WITHOUT any cleaning or transformation.
Its purpose is to understand what we're working with before P0.3.
"""

import pandas as pd
import numpy as np
import os

# ============================================================
# 1. Load the raw dataset
# ============================================================

DATA_PATH = os.path.join("data", "raw", "lending_club_loans.csv")

print("=" * 60)
print("LENDING CLUB LOAN DATA — INITIAL INSPECTION")
print("=" * 60)
print(f"\nData source: OpenML Dataset ID 43729")
print(f"File path: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print(f"\n{'=' * 60}")
print("1. DATASET SHAPE")
print(f"{'=' * 60}")
print(f"Rows:    {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

# ============================================================
# 2. Column names and dtypes
# ============================================================

print(f"\n{'=' * 60}")
print("2. COLUMNS AND DATA TYPES")
print(f"{'=' * 60}")
print(f"\n{'Column':<25} {'Dtype':<15}")
print("-" * 40)
for col in df.columns:
    print(f"{col:<25} {str(df[col].dtype):<15}")

# ============================================================
# 3. Missing values
# ============================================================

print(f"\n{'=' * 60}")
print("3. MISSING VALUES")
print(f"{'=' * 60}")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": missing_pct
})
print(f"\n{missing_df.to_string()}")
print(f"\nTotal cells with missing values: {df.isnull().sum().sum()}")

# ============================================================
# 4. Target variable: not.fully.paid
# ============================================================

print(f"\n{'=' * 60}")
print("4. TARGET VARIABLE: not.fully.paid")
print(f"{'=' * 60}")
if "not.fully.paid" in df.columns:
    target_counts = df["not.fully.paid"].value_counts().sort_index()
    target_pcts = df["not.fully.paid"].value_counts(normalize=True).sort_index() * 100
    print(f"\nValue counts:")
    for val in target_counts.index:
        label = "Fully Paid" if val == 0 else "Not Fully Paid (Default)"
        print(f"  {val} ({label}): {target_counts[val]:,} ({target_pcts[val]:.1f}%)")
    print(f"\nClass imbalance ratio (majority / minority): "
          f"{target_counts.max() / target_counts.min():.2f}")
else:
    print("WARNING: 'not.fully.paid' column not found!")
    print("Available columns:", list(df.columns))

# ============================================================
# 5. Check for loan_status column (common in full LC data)
# ============================================================

print(f"\n{'=' * 60}")
print("5. LOAN STATUS CHECK")
print(f"{'=' * 60}")
if "loan_status" in df.columns:
    print(f"\nloan_status values:")
    print(df["loan_status"].value_counts().to_string())
else:
    print("\nNo 'loan_status' column found.")
    print("This dataset uses 'not.fully.paid' as the pre-computed binary target.")
    print("  0 = Fully Paid")
    print("  1 = Not Fully Paid (Default/Charged Off)")

# ============================================================
# 6. Date columns check
# ============================================================

print(f"\n{'=' * 60}")
print("6. DATE COLUMNS CHECK")
print(f"{'=' * 60}")
date_cols = [col for col in df.columns if any(
    keyword in col.lower() for keyword in ["date", "issue", "earliest", "last"]
)]
if date_cols:
    print(f"\nPotential date columns found: {date_cols}")
    for col in date_cols:
        print(f"\n  {col}:")
        print(f"    Sample values: {df[col].head().tolist()}")
else:
    print("\nNo date columns found in this dataset.")
    print("This means we CANNOT do temporal train/test split by loan issue date.")
    print("We will need an alternative strategy for P0.8 (discussed below).")

# ============================================================
# 7. Numerical summary
# ============================================================

print(f"\n{'=' * 60}")
print("7. NUMERICAL SUMMARY")
print(f"{'=' * 60}")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumeric columns ({len(numeric_cols)}):")
print(df[numeric_cols].describe().round(2).to_string())

# ============================================================
# 8. Categorical columns
# ============================================================

print(f"\n{'=' * 60}")
print("8. CATEGORICAL COLUMNS")
print(f"{'=' * 60}")
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
if cat_cols:
    for col in cat_cols:
        print(f"\n  {col}:")
        print(f"    Unique values ({df[col].nunique()}): {df[col].unique().tolist()}")
else:
    print("No categorical (object) columns found.")

# ============================================================
# 9. credit.policy column inspection
# ============================================================

print(f"\n{'=' * 60}")
print("9. CREDIT POLICY COLUMN")
print(f"{'=' * 60}")
if "credit.policy" in df.columns:
    print(f"\ncredit.policy value counts:")
    cp_counts = df["credit.policy"].value_counts().sort_index()
    for val, count in cp_counts.items():
        pct = count / len(df) * 100
        label = "Meets credit underwriting" if val == 1 else "Does NOT meet criteria"
        print(f"  {val} ({label}): {count:,} ({pct:.1f}%)")

# ============================================================
# 10. Key feature distributions (quick stats)
# ============================================================

print(f"\n{'=' * 60}")
print("10. KEY FEATURE QUICK STATS")
print(f"{'=' * 60}")
key_features = {
    "int.rate": "Interest Rate",
    "installment": "Monthly Installment ($)",
    "log.annual.inc": "Log Annual Income",
    "dti": "Debt-to-Income Ratio",
    "fico": "FICO Score",
    "days.with.cr.line": "Days with Credit Line",
    "revol.bal": "Revolving Balance ($)",
    "revol.util": "Revolving Utilization (%)",
}
for col, desc in key_features.items():
    if col in df.columns:
        print(f"\n  {desc} ({col}):")
        print(f"    Min: {df[col].min():.2f}  |  "
              f"Median: {df[col].median():.2f}  |  "
              f"Max: {df[col].max():.2f}  |  "
              f"Mean: {df[col].mean():.2f}")

# ============================================================
# 11. Summary of implications for our project
# ============================================================

print(f"\n{'=' * 60}")
print("11. IMPLICATIONS FOR PROJECT PIPELINE")
print(f"{'=' * 60}")
print("""
KEY FINDINGS:
  1. Dataset has 9,578 rows and 14 columns (simplified LC dataset)
  2. Target is 'not.fully.paid' (already binary: 0=paid, 1=default)
  3. No 'loan_status' column — target is pre-computed
  4. No date columns — cannot do calendar-based temporal split
  5. No member_id — cannot build per-borrower history features
  6. Annual income is log-transformed ('log.annual.inc')
  7. One categorical column: 'purpose' (loan purpose)
  8. FICO score is available directly

IMPACT ON PROJECT PLAN:
  - P0.3 (Target): Already defined as 'not.fully.paid', minimal work needed
  - P0.4 (SQLite): We can still create a relational schema
  - P0.5 (SQL features): We can create synthetic transactions table
                          for SQL feature extraction demonstration
  - P0.6 (Leakage audit): Still critical — verify no post-origination info
  - P0.8 (Temporal split): Must use random or stratified split instead
                            of date-based split (no date column)
  - P0.9/P0.10 (Models): Proceed normally with baseline + XGBoost
""")

print("=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)
