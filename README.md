# Credit Risk & Loan Default Prediction System

> **Disclaimer:** This is a portfolio and ML engineering demonstration project built on a public dataset.
> It is NOT a production lending system, does NOT make real credit decisions, and must NOT be used
> to evaluate real loan applications.

---

## Overview

End-to-end binary classification system that predicts whether a loan will **default or be charged off**,
based exclusively on information available at the moment of loan origination.

The project covers the full ML lifecycle: dataset acquisition, target definition, leakage auditing,
feature engineering, temporal model training, explainability analysis, probability calibration,
and a production-style FastAPI inference service.

---

## Problem Statement

Given the loan application data available at origination (borrower characteristics, credit bureau
signals, loan terms), predict the probability that the loan will eventually be charged off or default.

**Prediction point:** Loan origination only. No post-origination repayment performance data is
used as input features. This constraint is strictly enforced throughout the entire pipeline.

---

## Dataset

| Attribute | Detail |
|:---|:---|
| Source | Public Lending Club Accepted Loans dataset |
| Loans retained after target definition | 1,348,099 |
| Defaults (Charged Off / Default) | 269,360 |
| Non-defaults (Fully Paid) | 1,078,739 |
| Overall default rate | 19.98% |

Loans with indeterminate outcomes (Current, Late, In Grace Period) are excluded from the
dataset because their final outcome is unknown at the time of analysis.

---

## Target Definition (P0.3)

The binary target `default_flag` is derived from `loan_status`:

| `loan_status` value | `default_flag` |
|:---|:---|
| Fully Paid | 0 |
| Does not meet the credit policy. Status:Fully Paid | 0 |
| Charged Off | 1 |
| Default | 1 |
| Does not meet the credit policy. Status:Charged Off | 1 |
| Current, Late (any), In Grace Period | Excluded |

---

## Leakage Prevention (P0.4)

A complete audit of all 151 columns in the dataset classifies every field against the
origination-time prediction point:

| Category | Count |
|:---|:---|
| Post-origination leakage fields (excluded) | 37 |
| Core safe origination features | 66 |
| High-null safe origination features | 16 |
| Identifiers excluded | 3 |

Examples of excluded leakage fields: `total_pymnt`, `out_prncp`, `recoveries`,
`hardship_flag`, `debt_settlement_flag`, `last_pymnt_d`, `collection_recovery_fee`.

The FastAPI inference service enforces this contract at request time: any request
containing a post-origination field is rejected with HTTP 422.

---

## Feature Engineering (P0.5 / P0.6)

The `FeatureEngineer` class (`src/features/build_features.py`) is fit **exclusively on
the training split** and applied identically to validation and test splits.

**Transformations applied:**

| Technique | Details |
|:---|:---|
| Outlier capping | `annual_inc` and `dti` capped using training-derived 99.5th-percentile thresholds |
| Numeric imputation | Median imputation for 35 features (train-derived medians) |
| Informative missingness | 6 binary flags for delinquency-related fields (e.g., `has_prior_delinq`) |
| Sparse imputation | 5 delinquency timing fields imputed with `max(train) + 1` (LR) or `-1` (XGB) |
| Log transform | `annual_inc` log1p-transformed for Logistic Regression |
| Standard scaling | Applied to all numeric features for Logistic Regression |
| One-hot encoding | `addr_state` (50 dummies), `sub_grade` (34 dummies for LR; ordinal for XGB), `purpose`, `home_ownership`, and other categoricals |

**Output dimensions:**

| Model | Features |
|:---|:---|
| Logistic Regression | 168 |
| XGBoost | 135 |

---

## Temporal Train / Validation / Test Strategy (P0.6)

Random splitting is not used. Splits are strictly time-ordered to simulate real deployment:

| Split | Criteria | Rows | Default Rate |
|:---|:---|:---|:---|
| **Train** | Issue year <= 2014 | 453,809 | 17.03% |
| **Validation** | Issue year 2015, 36-month term | 283,026 | 14.89% |
| **Test** | Issue year 2016, 36-month term | 232,361 | 19.85% |

- No ID overlap between any split (verified: 0 overlapping loan IDs).
- All post-origination leakage features are absent from all splits (verified: 0 leakage columns).
- The test set was evaluated **exactly once** after all model and threshold decisions were frozen.

---

## Models (P0.7)

### Logistic Regression (interpretable baseline)

| Hyperparameter | Value |
|:---|:---|
| `class_weight` | `balanced` |
| `penalty` | `l2` (C=1.0) |
| `max_iter` | 1000 |
| `random_state` | 42 |

### XGBoost (performance benchmark)

| Hyperparameter | Value |
|:---|:---|
| `scale_pos_weight` | 4.87 (train neg/pos ratio) |
| `n_estimators` | 300 |
| `max_depth` | 5 |
| `learning_rate` | 0.1 |
| `subsample` / `colsample_bytree` | 0.8 / 0.8 |
| `min_child_weight` | 10 |
| `eval_metric` | `logloss` |
| `random_state` | 42 |

---

## Model Evaluation Results (P0.7)

Threshold selection was performed on the validation set only (maximizing F1). The selected
threshold for each model was frozen and applied exactly once to the test set.

| Metric | Logistic Regression (threshold 0.50) | XGBoost (threshold 0.45) |
|:---|:---|:---|
| **ROC-AUC** | 0.6998 | **0.7083** |
| **PR-AUC (Average Precision)** | 0.3480 | **0.3654** |
| **F1 Score** | 0.4117 | **0.4238** |
| **Precision** | 0.3384 | 0.3227 |
| **Recall** | 0.5257 | **0.6173** |

**Test confusion matrix — XGBoost:**

| | Predicted Non-Default | Predicted Default |
|:---|:---|:---|
| Actual Non-Default | 126,490 (TN) | 59,751 (FP) |
| Actual Default | 17,651 (FN) | 28,469 (TP) |

The XGBoost model achieved a test ROC-AUC of 0.7083 under the project's temporal evaluation setup.

> **Note:** Threshold selection based on F1 assumes equal cost for false positives and false
> negatives. In a real lending context, the financial loss from a default (principal loss) is
> typically much higher than the cost of a rejected application. A production threshold would
> optimize a profit/loss function rather than F1.

---

## Explainability (P0.8)

Explainability is based on **model behavior analysis** on the held-out validation set.
These findings reflect correlations learned by the models, **not causal relationships**.

**Logistic Regression — Top features by |coefficient|:**

Sub-grade dummies dominate (G1, G5, G4, F4...) with the expected monotonic risk gradient:
coefficients increase from low-risk grades (A2) to high-risk grades (G4-G5), confirming
the model has learned Lending Club's internal risk-grading signal.

Other key features: `int_rate` (higher rate -> higher default log-odds), `annual_inc` (higher income -> lower),
`dti` (higher debt-to-income -> higher), `fico_score` (higher -> lower).

**XGBoost — Feature importance:**

Tree-based feature importances computed on the validation set. The same core signals dominate:
credit grade, FICO, interest rate, income, and delinquency history.

> All feature importance reflects model behavior, not causal mechanisms.
> A feature being important to the model does not imply it causes default.

---

## Probability Calibration (P0.9)

Raw model probability outputs from both models are severely miscalibrated due to class-imbalance
corrections (`class_weight='balanced'` / `scale_pos_weight=4.87`). Raw model probabilities showed
substantial calibration error on the evaluation data.

**Post-hoc Isotonic Regression calibration** is applied to correct this:

| Metric | XGB Base | XGB Calibrated |
|:---|:---|:---|
| Brier Score (cal-eval subset) | 0.179 | **0.118** |
| ECE (quantile, n=10) | 0.234 | **0.002** |

After calibration:
- On the held-out calibration-evaluation subset, isotonic calibration reduced XGBoost Brier Score from approximately 0.179 to 0.118 and ECE from approximately 0.234 to 0.002.
- ROC-AUC is unchanged (isotonic calibration is a monotonic transformation; it preserves rank order).

**Implementation:**
- Calibrator class: `IsotonicRegression(out_of_bounds='clip')` from `sklearn.isotonic`
- Fit on: 70% random subset of the validation set (cal-train, ~198k rows)
- Evaluated on: disjoint 30% subset (cal-eval, ~85k rows)
- Test set: not used during calibration
- Calibration artifact: `results/calibration/p09_calibrated_xgb_isotonic.joblib`

---

## FastAPI Inference Service (P0.10)

A production-style REST API that orchestrates the full inference pipeline using the frozen
serialized artifacts.

### Inference pipeline

```
POST /predict
    |
    v
Pydantic schema validation (LoanApplicationRequest, extra='forbid')
    |  -- HTTP 422 on any leakage field, unknown field, or invalid value
    v
Serialized FeatureEngineer (artifacts/feature_engineer_xgb.joblib)
    |  -- Exact training-time transforms applied; no refitting
    v
Frozen XGBoost model (src/models/xgboost_baseline.joblib)
    |  -- predict_proba() -> raw default probability
    v
Frozen Isotonic Calibrator (results/calibration/p09_calibrated_xgb_isotonic.joblib)
    |  -- .predict() -> calibrated default probability
    v
Policy layer (app/policy.py)
    |  -- risk band mapping (LOW / MODERATE / HIGH / VERY_HIGH)
    v
PredictionResponse JSON
```

### Endpoints

| Endpoint | Method | Description |
|:---|:---|:---|
| `/predict` | POST | Runs full inference on a single loan application |
| `/health` | GET | Returns 200 / `{"status": "ok"}` only when all artifacts are loaded |
| `/model-info` | GET | Returns model metadata (name, version, environment) |

### Input validation

The API enforces strict input validation via Pydantic:

- Post-origination/leakage fields are not part of the accepted request schema; because extra fields are forbidden, attempts to submit them are rejected with HTTP 422
- Required numeric fields validated for positive values and correct ranges
- `fico_range_low` must be <= `fico_range_high`; both in [300, 850]
- Date fields (`issue_d`, `earliest_cr_line`) must match `Mon-YYYY` format
- `earliest_cr_line` must not be after `issue_d`
- Categorical fields validated against exact allowed values (matching the FeatureEngineer)
- Optional fields that are omitted are treated as `NaN` and deterministically imputed
  by the FeatureEngineer using training-set-learned statistics

### Artifact safety

- The `FeatureEngineer` is **never refit** at inference. Only `.transform()` is called.
- The XGBoost model is **never refit** at inference. Only `.predict_proba()` is called.
- The Isotonic calibrator is **never refit** at inference. Only `.predict()` is called.
- Artifacts are loaded once at startup and cached for the lifetime of the process.

---

## Docker Deployment (P0.10)

```bash
# Build
docker build -t credit-risk-api .

# Run
docker run -p 8000:8000 credit-risk-api
```

The Docker image includes only production-required artifacts:
- `app/` (API code)
- `src/` (source modules including the FeatureEngineer class definition)
- `artifacts/feature_engineer_xgb.joblib` (serialized FeatureEngineer)
- `src/models/xgboost_baseline.joblib` (frozen XGBoost)
- `results/calibration/p09_calibrated_xgb_isotonic.joblib` (frozen calibrator)
- `artifacts/model_metadata.json` (traceability record)

The raw dataset, training parquet files, notebooks, and all other experiment artifacts
are excluded from the Docker build context.

---

## Project Structure

```
CREDIT_RISK_PREDIction-SYSTEM/
|
+-- app/                              # FastAPI inference service
|   +-- config.py                     # Centralized artifact path constants
|   +-- inference.py                  # Core inference pipeline (FE + XGB + calibration)
|   +-- main.py                       # FastAPI app, endpoints, lifespan startup
|   +-- policy.py                     # Risk band mapping (LOW/MODERATE/HIGH/VERY_HIGH)
|   +-- schemas.py                    # Pydantic request/response models
|
+-- src/
|   +-- features/
|   |   +-- build_features.py         # FeatureEngineer class (frozen P0.5 implementation)
|   +-- models/
|   |   +-- xgboost_baseline.joblib   # Frozen trained XGBoost (P0.7)
|   |   +-- logistic_regression_baseline.joblib  # Frozen trained LR (P0.7)
|   +-- data/                         # Data loading utilities
|
+-- scripts/
|   +-- p08_explainability.py         # P0.8 explainability analysis
|   +-- p08_completion.py             # P0.8 supplementary analysis
|   +-- p09_calibration.py            # P0.9 calibration experiment
|   +-- serialize_feature_pipeline.py # P0.10 Phase 1 FeatureEngineer serialization
|
+-- artifacts/
|   +-- feature_engineer_xgb.joblib   # Serialized FeatureEngineer (P0.10)
|   +-- model_metadata.json           # Model traceability record
|
+-- results/
|   +-- calibration/
|   |   +-- p09_calibrated_xgb_isotonic.joblib  # Frozen isotonic calibrator (P0.9)
|   |   +-- p09_results.json          # P0.9 calibration metrics
|   +-- p07_results.json              # P0.7 model evaluation metrics
|
+-- tests/
|   +-- test_api.py                   # API integration tests (12 tests)
|   +-- test_validation.py            # Schema validation tests (11 tests)
|
+-- docs/
|   +-- P0.3_target_definition.md
|   +-- P0.4_leakage_audit.md
|   +-- P0.5_feature_engineering_design.md
|   +-- P0.6_temporal_dataset_design.md
|   +-- P0.7_model_results.md
|   +-- P0.8_explainability_report.md
|   +-- P0.9_calibration_report.md
|   +-- P0.10_inference_design.md
|   +-- inference_contract.md
|   +-- model_card.md
|
+-- data/
|   +-- raw/                          # Original Lending Club CSV (not in repo)
|   +-- processed/                    # Cleaned parquet, temporal split parquets
|
+-- notebooks/                        # Exploration notebooks
+-- Dockerfile
+-- requirements.txt
+-- .dockerignore
+-- .gitignore
```

---

## Tech Stack

| Category | Tools |
|:---|:---|
| Language | Python 3.11 |
| Data | pandas 2.2.2, NumPy |
| ML | scikit-learn 1.8.0, XGBoost 3.2.0, joblib 1.5.3 |
| API | FastAPI, Pydantic v2, Uvicorn |
| Containerization | Docker |
| Visualization | Matplotlib |
| Exploration | Jupyter |

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the API locally

```bash
# From project root
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

### Run tests

```bash
# From project root
PYTHONPATH=. pytest tests/ -v
```



---

## Example API Request and Response

The following is an **example inference result only**. It is not a real lending decision.

**Request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 10000.0,
    "term": " 36 months",
    "int_rate": 10.99,
    "installment": 327.34,
    "sub_grade": "B4",
    "home_ownership": "MORTGAGE",
    "annual_inc": 75000.0,
    "verification_status": "Verified",
    "dti": 15.0,
    "addr_state": "CA",
    "purpose": "debt_consolidation",
    "application_type": "Individual",
    "fico_range_low": 700.0,
    "fico_range_high": 704.0,
    "earliest_cr_line": "Jan-2005",
    "issue_d": "Dec-2015",
    "total_acc": 20,
    "open_acc": 10,
    "revol_bal": 5000.0,
    "delinq_2yrs": 0.0,
    "acc_now_delinq": 0.0,
    "delinq_amnt": 0.0,
    "pub_rec": 0.0,
    "inq_last_6mths": 1.0
  }'
```

**Response:**

```json
{
  "raw_default_probability": 0.3667903244495392,
  "calibrated_default_probability": 0.1306661069393158,
  "model_name": "xgboost_baseline",
  "model_version": "P0.7",
  "calibration_method": "p09_calibrated_xgb_isotonic.joblib",
  "risk_band": "MODERATE",
  "disclaimer": "ML engineering portfolio implementation. Not a production lending system."
}
```

**Risk band thresholds** (for demonstration only):

| Calibrated Probability | Risk Band |
|:---|:---|
| < 0.10 | LOW |
| 0.10 - 0.25 | MODERATE |
| 0.25 - 0.50 | HIGH |
| >= 0.50 | VERY_HIGH |

---

## Limitations

- **Not production ready.** This is a Tier-1 baseline. High-null schema features, zip-code-level
  features, and additional bureau tradeline features are deferred.
- **Not a causal model.** The model identifies statistical correlations in historical Lending Club
  data. A feature being predictive of default does not imply it causes default.
- **Temporal stability not guaranteed.** The model is trained on 2007-2014 data. Credit market
  conditions change; this model will degrade without recalibration as economic conditions evolve.
- **Fairness not verified.** While direct sensitive attributes are absent from Lending Club data,
  proxies may exist (e.g., `addr_state`, `annual_inc`). A rigorous disparate impact analysis is
  required before applying this or any derivative model to real applicants.
- **Threshold is not cost-optimized.** The operating threshold (0.45 for XGBoost) was selected by
  maximizing F1, which assumes equal cost for false positives and false negatives. A real
  lending deployment requires a cost-sensitive threshold calibrated to actual financial outcomes.
- **Dataset vintage.** The Lending Club dataset reflects 2007-2019 lending behavior. It does not
  represent current credit market dynamics.

---

## Disclaimer

This project is a **portfolio and ML engineering demonstration**. It uses a public historical
dataset from Lending Club. It does not use real banking transactions or production borrower data.
It is not affiliated with Lending Club or any lending institution.

**This system must NOT be used to evaluate real loan applications or make real credit decisions.**
