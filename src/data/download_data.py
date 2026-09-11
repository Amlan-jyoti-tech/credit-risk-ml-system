"""
Download Lending Club Loan Data.

PRIMARY DATASET (full accepted loans):
    Source: Kaggle - wordsforthewise/lending-club
    File: accepted_2007_to_2018Q4.csv (~1.6 GB)
    Rows: ~2.26 million
    Columns: 151
    License: CC0 Public Domain
    Original source: LendingClub.com

LEGACY DATASET (simplified):
    Source: OpenML Dataset ID 43729
    File: lending_club_loans.csv (~0.8 MB)
    Rows: 9,578
    Columns: 14

This project uses the FULL dataset for all phases from P0.2R onward.
The simplified dataset was used in early P0.1-P0.3 phases and is retained
for reference but is not used in the modeling pipeline.

Download instructions (full dataset):
    1. Go to https://www.kaggle.com/datasets/wordsforthewise/lending-club
    2. Download accepted_2007_to_2018Q4.csv.gz
    3. Extract and place in data/raw/accepted_2007_to_2018Q4.csv
"""

import os

# === Configuration ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# Primary dataset (full accepted loans)
PRIMARY_DATASET = os.path.join(RAW_DIR, "accepted_2007_to_2018Q4.csv")

# Legacy dataset (simplified, from early phases)
LEGACY_DATASET = os.path.join(RAW_DIR, "lending_club_loans.csv")


def check_data():
    """Check which datasets are available."""
    print("=" * 60)
    print("DATASET STATUS CHECK")
    print("=" * 60)

    primary_exists = os.path.exists(PRIMARY_DATASET)
    legacy_exists = os.path.exists(LEGACY_DATASET)

    if primary_exists:
        size_mb = os.path.getsize(PRIMARY_DATASET) / (1024 * 1024)
        print(f"\n  [OK] Primary dataset found: {PRIMARY_DATASET}")
        print(f"       Size: {size_mb:.1f} MB")
    else:
        print(f"\n  [MISSING] Primary dataset NOT found: {PRIMARY_DATASET}")
        print(f"       Download from: https://www.kaggle.com/datasets/wordsforthewise/lending-club")
        print(f"       File needed: accepted_2007_to_2018Q4.csv")

    if legacy_exists:
        size_mb = os.path.getsize(LEGACY_DATASET) / (1024 * 1024)
        print(f"\n  [OK] Legacy dataset found: {LEGACY_DATASET}")
        print(f"       Size: {size_mb:.1f} MB")
        print(f"       (Not used in current pipeline -- retained for reference)")

    return primary_exists


if __name__ == "__main__":
    if not check_data():
        print("\n\nPlease download the primary dataset before proceeding.")
        print("See docstring at top of this file for instructions.")
