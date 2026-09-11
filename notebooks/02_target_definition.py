"""
02_target_definition.py
=======================
P0.3R: Target Definition & Label Cleaning Execution

This script executes the target cleaning pipeline defined in
src/data/target_cleaning.py.

It performs the following:
1. Loads the full raw Lending Club dataset (~2.26M rows).
2. Maps the multi-class `loan_status` to a binary `default_flag` (0/1).
3. Excludes unresolved statuses (e.g., Current, In Grace Period).
4. Saves the resulting curated dataset to `data/processed/lending_club_target_cleaned.parquet`.
"""

import sys
import os

# Add the project root to the Python path to allow importing from src
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.target_cleaning import run_target_cleaning

if __name__ == "__main__":
    # Execute the pipeline using default paths configured in the module
    df_cleaned, summary, saved_path = run_target_cleaning()
