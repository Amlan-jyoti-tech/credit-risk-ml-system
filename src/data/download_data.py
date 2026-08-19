"""
Download Lending Club Loan Data from OpenML.

Source: OpenML Dataset ID 43729 (Lending-Club-Loan-Data, 2007-2015)
License: CC0 Public Domain
Original source: LendingClub.com

This script downloads the full dataset, then samples ~100K rows
for our project (keeping it manageable for development).
"""

import pandas as pd
import os
import sys

# === Configuration ===
# Go up from src/data/ to project root, then into data/raw/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PARQUET_URL = "https://data.openml.org/datasets/0004/43729/dataset_43729.pq"
OUTPUT_FILE = os.path.join(RAW_DIR, "lending_club_loans.csv")
SAMPLE_SIZE = 100_000
RANDOM_SEED = 42


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # Step 1: Download the dataset
    print(f"Downloading Lending Club dataset from OpenML...")
    print(f"URL: {PARQUET_URL}")
    print("This may take a minute depending on your connection...")
    
    try:
        df = pd.read_parquet(PARQUET_URL)
    except Exception as e:
        print(f"\nERROR: Failed to download from OpenML: {e}")
        print("Possible causes:")
        print("  - No internet connection")
        print("  - OpenML server is down")
        print("  - pyarrow not installed (run: pip install pyarrow)")
        sys.exit(1)
    
    print(f"\nFull dataset shape: {df.shape}")
    print(f"Full dataset rows: {df.shape[0]:,}")
    print(f"Full dataset columns: {df.shape[1]}")
    
    # Step 2: Sample if dataset is larger than SAMPLE_SIZE
    if len(df) > SAMPLE_SIZE:
        print(f"\nSampling {SAMPLE_SIZE:,} rows (random_seed={RANDOM_SEED})...")
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    else:
        print(f"\nDataset has {len(df):,} rows, keeping all (under {SAMPLE_SIZE:,} threshold).")
        df_sample = df
    
    # Step 3: Save to CSV
    print(f"Saving to: {OUTPUT_FILE}")
    df_sample.to_csv(OUTPUT_FILE, index=False)
    
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"Saved! File size: {file_size_mb:.1f} MB")
    print(f"Sample shape: {df_sample.shape}")
    print("\nDone. Dataset is ready for inspection.")


if __name__ == "__main__":
    main()
