# Credit Risk / Loan Default Prediction System

> **Status:** Work in Progress

## Overview

Binary classification system that predicts whether a loan will **default or be charged off** at the point of loan origination.

Built using the public **Lending Club Accepted Loans** dataset.

## Key Concepts

- **Prediction point:** Loan origination (no future information used as features)
- **Temporal train/validation/test split** to prevent data leakage
- **SQL-based feature extraction** from a relational schema (customers, loans, transactions)
- **Baseline:** Logistic Regression → **Main model:** XGBoost
- **Evaluation:** Precision, Recall, F1, ROC-AUC, PR-AUC with threshold analysis

## Tech Stack

- Python, pandas, NumPy
- scikit-learn, XGBoost
- SQLite (relational feature extraction)
- Matplotlib (visualization)
- Jupyter (exploration)

## Project Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/            # Original Lending Club CSV
│   └── processed/      # Cleaned data, SQLite DB
├── notebooks/          # Exploration & analysis notebooks
├── src/
│   ├── data/           # Data loading and cleaning
│   ├── features/       # Feature engineering & SQL extraction
│   ├── models/         # Model training
│   ├── evaluation/     # Metrics and evaluation
│   └── train.py        # Reproducible training pipeline
├── results/            # Metrics, plots, model artifacts
└── docs/               # Documentation (leakage audit, etc.)
```

## Setup

```bash
pip install -r requirements.txt
```

## Note

This is a **portfolio/simulation project** using a public dataset. The transactions table is synthetically generated to demonstrate relational SQL feature extraction and is clearly documented as such. This project does not use real banking transactions or production data.
