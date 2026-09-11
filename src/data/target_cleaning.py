"""
target_cleaning.py
==================
P0.3R: Target Definition & Label Cleaning (Full Dataset).

This module creates a clean, validated binary target column (`default_flag`)
for loan default prediction from the full Lending Club dataset.

Dataset: Kaggle `accepted_2007_to_2018Q4.csv` (~2.26M rows)
Source target column: `loan_status` (Multi-class string)

Key design decisions:
    - We map definitive final statuses (Fully Paid, Charged Off, Default) to 0/1.
    - We exclude active/unresolved statuses (Current, Late, In Grace Period).
    - We map explicit policy exception statuses.
    - We drop the original `loan_status` column after mapping to prevent leakage.
    - Output is saved as Parquet to dramatically reduce file size and load times.
"""

import pandas as pd
import os
import sys

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "accepted_2007_to_2018Q4.csv")
# Saving as parquet to save space (1GB+ -> ~200MB) and load faster
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "lending_club_target_cleaned.parquet")

SOURCE_TARGET_COLUMN = "loan_status"
TARGET_COLUMN = "default_flag"

# Definitive mappings
TARGET_MAPPING = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
}

# Statuses indicating the loan outcome is not yet known
EXCLUDED_STATUSES = [
    "Current",
    "Late (31-120 days)",
    "Late (16-30 days)",
    "In Grace Period",
]


# ============================================================
# Functions
# ============================================================

def load_raw_dataset(data_path=None):
    """Load the raw Lending Club dataset from CSV."""
    path = data_path or RAW_DATA_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw dataset not found at: {path}\n"
            f"Please download accepted_2007_to_2018Q4.csv from Kaggle."
        )

    print("Loading raw dataset (this may take 30-60 seconds)...")
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded raw dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def inspect_target_column(df):
    """Inspect the source target column and report its distribution."""
    print(f"\n{'=' * 60}")
    print("TARGET COLUMN INSPECTION")
    print(f"{'=' * 60}")

    if SOURCE_TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{SOURCE_TARGET_COLUMN}' not found!")

    status_counts = df[SOURCE_TARGET_COLUMN].value_counts(dropna=False)
    for status, count in status_counts.items():
        pct = count / len(df) * 100
        print(f"  {str(status):<55} {count:>10,} ({pct:>5.2f}%)")

    null_count = df[SOURCE_TARGET_COLUMN].isnull().sum()
    print(f"\n  Null values in target: {null_count:,}")
    return {"value_counts": status_counts.to_dict(), "null_count": null_count}


def create_default_flag(df):
    """
    Create the binary `default_flag` column and drop unresolved statuses.
    """
    print(f"\n{'=' * 60}")
    print("CREATING default_flag TARGET COLUMN")
    print(f"{'=' * 60}")

    # 1. Drop rows with NaN in loan_status
    df_out = df.dropna(subset=[SOURCE_TARGET_COLUMN]).copy()
    initial_rows = len(df)
    nulls_dropped = initial_rows - len(df_out)
    if nulls_dropped > 0:
        print(f"  Dropped {nulls_dropped:,} rows with NaN in '{SOURCE_TARGET_COLUMN}'")

    # 2. Identify and drop excluded statuses
    excluded_mask = df_out[SOURCE_TARGET_COLUMN].isin(EXCLUDED_STATUSES)
    excluded_count = excluded_mask.sum()
    
    print(f"\n  Excluding {excluded_count:,} rows with unresolved statuses:")
    for status in EXCLUDED_STATUSES:
        count = (df_out[SOURCE_TARGET_COLUMN] == status).sum()
        print(f"    - '{status}': {count:>10,} rows")

    df_out = df_out[~excluded_mask].copy()

    # 3. Apply definitive mapping
    unmapped_before = df_out[SOURCE_TARGET_COLUMN].isnull().sum()
    df_out[TARGET_COLUMN] = df_out[SOURCE_TARGET_COLUMN].map(TARGET_MAPPING)

    unmapped_after = df_out[TARGET_COLUMN].isnull().sum()
    if unmapped_after > 0:
        unmapped_vals = df_out[df_out[TARGET_COLUMN].isnull()][SOURCE_TARGET_COLUMN].unique()
        raise ValueError(f"Found unmapped values in '{SOURCE_TARGET_COLUMN}': {unmapped_vals}")

    df_out[TARGET_COLUMN] = df_out[TARGET_COLUMN].astype(int)

    # 4. Drop the original loan_status column to prevent data leakage
    df_out = df_out.drop(columns=[SOURCE_TARGET_COLUMN])

    print(f"\n  Output column '{TARGET_COLUMN}' created successfully.")
    print(f"  Original column '{SOURCE_TARGET_COLUMN}' dropped from processed data.")
    
    total_excluded = nulls_dropped + excluded_count
    return df_out, total_excluded


def validate_target(df, original_row_count, excluded_count):
    """Run validation checks on the cleaned dataset."""
    print(f"\n{'=' * 60}")
    print("VALIDATION CHECKS")
    print(f"{'=' * 60}")

    assert TARGET_COLUMN in df.columns, f"'{TARGET_COLUMN}' not found."
    print(f"  [OK] '{TARGET_COLUMN}' column exists")

    assert SOURCE_TARGET_COLUMN not in df.columns, f"'{SOURCE_TARGET_COLUMN}' should be dropped."
    print(f"  [OK] '{SOURCE_TARGET_COLUMN}' dropped")

    assert "issue_d" in df.columns, "'issue_d' not found."
    print(f"  [OK] 'issue_d' preserved for temporal splitting")

    valid_values = {0, 1}
    actual_values = set(df[TARGET_COLUMN].unique())
    assert actual_values.issubset(valid_values), f"Unexpected values: {actual_values - valid_values}"
    print(f"  [OK] '{TARGET_COLUMN}' contains only {{0, 1}}")

    assert df[TARGET_COLUMN].isnull().sum() == 0, "Found null values in target."
    print(f"  [OK] No null values in '{TARGET_COLUMN}'")

    retained = len(df)
    expected_retained = original_row_count - excluded_count
    assert retained == expected_retained, f"Row count mismatch: {retained} != {expected_retained}"
    print(f"  [OK] Row counts reconcile: {original_row_count:,} - {excluded_count:,} = {retained:,}")

    print(f"\n  All validation checks passed.")


def generate_summary(df_original, df_cleaned, excluded_count):
    """Generate a concise summary of the target-cleaning process."""
    original_rows = len(df_original)
    retained_rows = len(df_cleaned)
    default_count = int((df_cleaned[TARGET_COLUMN] == 1).sum())
    non_default_count = int((df_cleaned[TARGET_COLUMN] == 0).sum())
    default_rate = default_count / retained_rows if retained_rows > 0 else 0.0

    summary = {
        "original_row_count": original_rows,
        "retained_row_count": retained_rows,
        "excluded_row_count": excluded_count,
        "default_count": default_count,
        "non_default_count": non_default_count,
        "default_rate": default_rate,
    }

    print(f"\n{'=' * 60}")
    print("TARGET CLEANING SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Original rows:      {summary['original_row_count']:>12,}")
    print(f"  Retained rows:      {summary['retained_row_count']:>12,}")
    print(f"  Excluded rows:      {summary['excluded_row_count']:>12,}")
    print(f"  Default (1):        {summary['default_count']:>12,}")
    print(f"  Non-default (0):    {summary['non_default_count']:>12,}")
    print(f"  Default rate:       {summary['default_rate'] * 100:>11.2f}%")
    
    return summary


def save_processed_dataset(df, output_path=None):
    """Save the cleaned modeling-ready dataset to Parquet."""
    path = output_path or PROCESSED_DATA_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)

    print(f"\nSaving to Parquet format for optimized storage/loading...")
    df.to_parquet(path, index=False)

    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print("DATASET SAVED")
    print(f"{'=' * 60}")
    print(f"  Path: {path}")
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  File size: {file_size_mb:.2f} MB")

    return path


def run_target_cleaning(raw_data_path=None, output_path=None):
    """Execute the full target-cleaning pipeline."""
    print("=" * 60)
    print("P0.3R: TARGET DEFINITION & LABEL CLEANING")
    print("=" * 60)

    df_raw = load_raw_dataset(raw_data_path)
    original_row_count = len(df_raw)

    inspect_target_column(df_raw)

    df_cleaned, excluded_count = create_default_flag(df_raw)

    validate_target(df_cleaned, original_row_count, excluded_count)

    summary = generate_summary(df_raw, df_cleaned, excluded_count)

    saved_path = save_processed_dataset(df_cleaned, output_path)

    print(f"\n{'=' * 60}")
    print("P0.3R COMPLETE")
    print(f"{'=' * 60}")

    return df_cleaned, summary, saved_path
