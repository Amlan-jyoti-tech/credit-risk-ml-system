# Model Card: P0.7 XGBoost Baseline (P0.10 Inference Release)

## Model Details
* **Model Name:** `xgboost_baseline`
* **Version:** P0.7 (Trained), P0.9 (Calibrated), P0.10 (Deployed)
* **Model Type:** XGBoost Classifier with Isotonic Regression Post-hoc Calibration
* **Purpose:** ML engineering portfolio demonstration for credit risk / loan default prediction.
* **Disclaimer:** NOT a production lending system. For educational and portfolio purposes only.

## Intended Use
* **Primary Use Case:** Predict the probability that a loan application will default (charge off) based on information available at origination.
* **Out of Scope:** Making real credit decisions, pricing loans, handling data post-origination.

## Data & Features
* **Training Data:** Lending Club accepted loans (Issue years 2007–2014), temporal split design (P0.6).
* **Feature Engineering:** Learned missing value imputation, capping, and missingness flags fit exclusively on the training set. (135 features total).
* **Leakage Prevention:** 37 post-origination fields explicitly dropped (e.g., total payments, collection records, hardship flags).

## Performance (Historical Baseline P0.7/P0.9)
* **Validation (2015 cohort, 36m):** Evaluated strictly on out-of-time cohort.
* **Raw Brier Score (XGB):** 0.1799
* **Calibrated Brier Score:** 0.1176
* **AUC:** 0.724 (Historical P0.7 baseline)

## Inference Architecture
* **Framework:** FastAPI
* **Pipeline:** Pydantic Validation $\to$ Reconstructed FeatureEngineer $\to$ Frozen XGBoost $\to$ Frozen Isotonic Calibrator $\to$ Policy Layer (Mock).
* **Safety:** Pydantic `extra="forbid"` drops any requests containing data leakage fields. 

## Limitations
* Isotonic calibration curves may introduce slight non-smoothness in mapping raw scores to probabilities.
* The model assumes feature distributions match the historical 2007-2014 Lending Club context.
