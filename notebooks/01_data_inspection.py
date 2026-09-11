"""
01_data_inspection.py
====================
P0.2R: Initial Data Inspection of the Full Lending Club Accepted Loans Dataset.

Source: Kaggle - wordsforthewise/lending-club
File: accepted_2007_to_2018Q4.csv
License: CC0 Public Domain
Original source: LendingClub.com (2007-2018)

This script inspects the raw dataset WITHOUT any cleaning or transformation.
Its purpose is to understand the data before P0.3R (target definition).

NOTE: This replaces the original P0.2 inspection which used the simplified
      OpenML dataset (9,578 rows, 14 columns). The full dataset has
      ~2.26 million rows and 151 columns.
"""

import pandas as pd
import numpy as np
import os
import sys

# ============================================================
# Configuration
# ============================================================

DATA_PATH = os.path.join("data", "raw", "accepted_2007_to_2018Q4.csv")

print("=" * 70)
print("LENDING CLUB ACCEPTED LOANS -- FULL DATASET INSPECTION (P0.2R)")
print("=" * 70)
print(f"\nData source: Kaggle (wordsforthewise/lending-club)")
print(f"File path: {DATA_PATH}")
print(f"File size: {os.path.getsize(DATA_PATH) / (1024**2):.1f} MB")

# ============================================================
# 1. Load the raw dataset
# ============================================================

print(f"\n{'=' * 70}")
print("1. LOADING DATASET")
print(f"{'=' * 70}")
print("Loading (this takes 30-60 seconds for ~2.3M rows)...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Rows:    {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

# ============================================================
# 2. Column inventory with dtypes
# ============================================================

print(f"\n{'=' * 70}")
print("2. COLUMN INVENTORY")
print(f"{'=' * 70}")
print(f"\n{'#':<4} {'Column':<40} {'Dtype':<12} {'Non-Null':>10} {'Null%':>7}")
print("-" * 80)
for i, col in enumerate(df.columns):
    dtype = str(df[col].dtype)
    non_null = df[col].notna().sum()
    null_pct = df[col].isnull().sum() / len(df) * 100
    print(f"{i:<4} {col:<40} {dtype:<12} {non_null:>10,} {null_pct:>6.1f}%")

# ============================================================
# 3. Target variable: loan_status
# ============================================================

print(f"\n{'=' * 70}")
print("3. TARGET VARIABLE: loan_status")
print(f"{'=' * 70}")
if "loan_status" in df.columns:
    status_counts = df["loan_status"].value_counts()
    print(f"\nUnique statuses: {df['loan_status'].nunique()}")
    print(f"Null count: {df['loan_status'].isnull().sum()}")
    print(f"\nDistribution:")
    for status, count in status_counts.items():
        pct = count / len(df) * 100
        print(f"  {status:<55} {count:>10,}  ({pct:>5.2f}%)")

    # Categorize for P0.3R reference
    print("\n  Preliminary categorization for P0.3R:")
    definitive = ["Fully Paid", "Charged Off", "Default"]
    unresolved = ["Current", "Late (31-120 days)", "In Grace Period", "Late (16-30 days)"]
    policy = [s for s in status_counts.index if "credit policy" in s.lower()]

    for status in status_counts.index:
        count = status_counts[status]
        if status in definitive:
            cat = "DEFINITIVE"
        elif status in unresolved:
            cat = "UNRESOLVED (exclude)"
        elif status in policy:
            cat = "POLICY (review)"
        else:
            cat = "UNKNOWN (review)"
        print(f"    [{cat:<25}] {status}")
else:
    print("WARNING: 'loan_status' column not found!")

# ============================================================
# 4. Issue date: temporal range
# ============================================================

print(f"\n{'=' * 70}")
print("4. ISSUE DATE (issue_d)")
print(f"{'=' * 70}")
if "issue_d" in df.columns:
    print(f"\nNull count: {df['issue_d'].isnull().sum()}")
    print(f"Unique months: {df['issue_d'].nunique()}")
    print(f"Sample values: {df['issue_d'].dropna().head(5).tolist()}")

    # Parse to datetime
    df["issue_date_parsed"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    parse_failures = df["issue_date_parsed"].isnull().sum() - df["issue_d"].isnull().sum()
    print(f"Parse failures: {parse_failures}")
    print(f"Earliest: {df['issue_date_parsed'].min()}")
    print(f"Latest:   {df['issue_date_parsed'].max()}")
    print(f"Span:     {(df['issue_date_parsed'].max() - df['issue_date_parsed'].min()).days} days")

    # Distribution by year
    print("\nLoans issued by year:")
    year_counts = df["issue_date_parsed"].dt.year.value_counts().sort_index()
    for year, count in year_counts.items():
        pct = count / len(df) * 100
        bar = "#" * int(pct)
        print(f"  {int(year)}: {count:>8,}  ({pct:>5.2f}%)  {bar}")
else:
    print("No 'issue_d' column found.")

# ============================================================
# 5. Borrower identification
# ============================================================

print(f"\n{'=' * 70}")
print("5. BORROWER IDENTIFICATION")
print(f"{'=' * 70}")

# Check member_id
if "member_id" in df.columns:
    null_count = df["member_id"].isnull().sum()
    print(f"\nmember_id:")
    print(f"  Null: {null_count:,} / {len(df):,} ({null_count/len(df)*100:.1f}%)")
    if null_count < len(df):
        print(f"  Unique: {df['member_id'].nunique()}")
    else:
        print("  STATUS: ALL NULL -- cannot identify borrowers")
        print("  IMPACT: No multi-loan-per-borrower features possible")

# Check loan id
if "id" in df.columns:
    null_count = df["id"].isnull().sum()
    unique = df["id"].nunique()
    print(f"\nid (loan ID):")
    print(f"  Null: {null_count:,}")
    print(f"  Unique: {unique:,} / {len(df):,}")
    print(f"  All unique: {unique == len(df)}")

# ============================================================
# 6. Missing values summary
# ============================================================

print(f"\n{'=' * 70}")
print("6. MISSING VALUES SUMMARY")
print(f"{'=' * 70}")

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)

# Categorize by missingness level
full = (missing == 0).sum()
low = ((missing > 0) & (missing_pct < 5)).sum()
medium = ((missing_pct >= 5) & (missing_pct < 50)).sum()
high = ((missing_pct >= 50) & (missing_pct < 95)).sum()
near_empty = (missing_pct >= 95).sum()

print(f"\n  Columns with 0% missing:    {full}")
print(f"  Columns with <5% missing:   {low}")
print(f"  Columns with 5-50% missing: {medium}")
print(f"  Columns with 50-95% missing:{high}")
print(f"  Columns with >95% missing:  {near_empty}")

print(f"\n  Top 20 most-missing columns:")
missing_sorted = missing_pct[missing_pct > 0].sort_values(ascending=False).head(20)
for col, pct in missing_sorted.items():
    print(f"    {col:<45} {pct:>6.1f}%")

# ============================================================
# 7. Key origination-time features
# ============================================================

print(f"\n{'=' * 70}")
print("7. KEY ORIGINATION-TIME FEATURES")
print(f"{'=' * 70}")

origination_features = {
    # Loan terms
    "loan_amnt": "Loan amount ($)",
    "term": "Loan term",
    "int_rate": "Interest rate (%)",
    "installment": "Monthly installment ($)",
    "grade": "LC risk grade",
    "sub_grade": "LC sub-grade",
    # Borrower info
    "emp_length": "Employment length",
    "home_ownership": "Home ownership",
    "annual_inc": "Annual income ($)",
    "verification_status": "Income verification",
    "purpose": "Loan purpose",
    "title": "Loan title (free text)",
    "addr_state": "Borrower state",
    "dti": "Debt-to-income ratio",
    # Credit history
    "fico_range_low": "FICO score (low)",
    "fico_range_high": "FICO score (high)",
    "earliest_cr_line": "Earliest credit line",
    "open_acc": "Open credit accounts",
    "total_acc": "Total credit accounts",
    "revol_bal": "Revolving balance ($)",
    "revol_util": "Revolving utilization (%)",
    "pub_rec": "Public derogatory records",
    "delinq_2yrs": "Delinquencies (last 2 yrs)",
    "inq_last_6mths": "Inquiries (last 6 months)",
    "mort_acc": "Mortgage accounts",
}

for col, desc in origination_features.items():
    if col not in df.columns:
        print(f"\n  [MISSING] {desc} ({col})")
        continue

    nulls = df[col].isnull().sum()
    null_pct = nulls / len(df) * 100

    if df[col].dtype == "object":
        nunique = df[col].nunique()
        top_vals = df[col].value_counts().head(5)
        print(f"\n  {desc} ({col}):  [categorical, {nunique} unique, {null_pct:.1f}% null]")
        for val, count in top_vals.items():
            pct = count / len(df) * 100
            print(f"    {str(val):<30} {count:>10,}  ({pct:.1f}%)")
    else:
        s = df[col].dropna()
        print(f"\n  {desc} ({col}):  [numeric, {null_pct:.1f}% null]")
        print(f"    min={s.min():.2f}  median={s.median():.2f}  "
              f"mean={s.mean():.2f}  max={s.max():.2f}")

# ============================================================
# 8. Potential post-origination columns (leakage risk)
# ============================================================

print(f"\n{'=' * 70}")
print("8. POST-ORIGINATION COLUMNS (DATA LEAKAGE RISK)")
print(f"{'=' * 70}")
print("""
These columns contain information that would NOT be available at the time
of loan origination. They MUST be excluded from model features in P0.4.
""")

post_origination = {
    # Payment data
    "total_pymnt": "Total payments received",
    "total_pymnt_inv": "Total payments by investors",
    "total_rec_prncp": "Principal received",
    "total_rec_int": "Interest received",
    "total_rec_late_fee": "Late fees received",
    "last_pymnt_d": "Last payment date",
    "last_pymnt_amnt": "Last payment amount",
    # Recovery data
    "recoveries": "Post-charge-off recoveries",
    "collection_recovery_fee": "Collection recovery fees",
    # Current status data
    "out_prncp": "Outstanding principal",
    "out_prncp_inv": "Outstanding principal (investor)",
    "last_credit_pull_d": "Last credit pull date",
    "last_fico_range_high": "Last FICO high (updated)",
    "last_fico_range_low": "Last FICO low (updated)",
    # Post-origination flags
    "hardship_flag": "Hardship program flag",
    "debt_settlement_flag": "Debt settlement flag",
    "settlement_status": "Settlement status",
    "settlement_amount": "Settlement amount",
    "settlement_date": "Settlement date",
}

for col, desc in post_origination.items():
    if col in df.columns:
        nulls = df[col].isnull().sum()
        sample = df[col].dropna().iloc[0] if nulls < len(df) else "ALL NULL"
        print(f"  [LEAKY] {col:<35} ({desc})")
    else:
        print(f"  [N/A]   {col:<35} ({desc})")

# ============================================================
# 9. Summary and implications
# ============================================================

print(f"\n{'=' * 70}")
print("9. SUMMARY AND IMPLICATIONS FOR PROJECT")
print(f"{'=' * 70}")

# Count definitive vs unresolved loans
if "loan_status" in df.columns:
    definitive_statuses = ["Fully Paid", "Charged Off", "Default"]
    policy_paid = "Does not meet the credit policy. Status:Fully Paid"
    policy_charged = "Does not meet the credit policy. Status:Charged Off"

    definitive_mask = df["loan_status"].isin(definitive_statuses)
    unresolved_mask = ~definitive_mask
    # Policy statuses
    policy_mask = df["loan_status"].isin([policy_paid, policy_charged])

    print(f"""
  DATASET OVERVIEW:
    Total rows:           {len(df):>12,}
    Total columns:        {df.shape[1]:>12}
    Date range:           Jun 2007 -- Dec 2018

  TARGET (loan_status):
    Definitive outcomes:  {definitive_mask.sum():>12,}  ({definitive_mask.sum()/len(df)*100:.1f}%)
      - Fully Paid:       {(df['loan_status']=='Fully Paid').sum():>12,}
      - Charged Off:      {(df['loan_status']=='Charged Off').sum():>12,}
      - Default:          {(df['loan_status']=='Default').sum():>12,}
    Policy exceptions:    {policy_mask.sum():>12,}  ({policy_mask.sum()/len(df)*100:.1f}%)
    Unresolved/active:    {(unresolved_mask & ~policy_mask).sum():>12,}  ({(unresolved_mask & ~policy_mask).sum()/len(df)*100:.1f}%)

  TEMPORAL SPLIT:
    issue_d available:    YES (Jun 2007 -- Dec 2018)
    Enables genuine time-based train/val/test split

  BORROWER HISTORY:
    member_id:            ALL NULL -- no multi-borrower linkage possible

  FEATURE RICHNESS:
    Origination features: ~50+ columns available at loan decision time
    Post-origination:     ~20+ columns that MUST be excluded (leakage)
    Near-empty (>95%):    ~{near_empty} columns (mostly hardship/settlement/joint)

  IMPACT ON PROJECT PHASES:
    P0.3R: Map loan_status to default_flag (exclude Current, Late, Grace)
    P0.4:  Leakage audit -- critical with this many post-origination columns
    P0.5:  Rich feature engineering from 50+ origination-time columns
    P0.6:  Temporal train/val/test split using issue_d
    P0.7:  Logistic Regression baseline
    P0.8:  XGBoost model
    P0.9:  Class imbalance handling
    P0.10: Full evaluation suite
""")

# Clean up temporary column
if "issue_date_parsed" in df.columns:
    df.drop(columns=["issue_date_parsed"], inplace=True)

print("=" * 70)
print("P0.2R INSPECTION COMPLETE")
print("=" * 70)
