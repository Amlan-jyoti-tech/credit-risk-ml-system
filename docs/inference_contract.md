# Inference Contract (P0.10)

## Overview
This document defines the inference interface for the Credit Risk Prediction API (P0.10).

## Endpoints

### 1. `POST /predict`
Evaluates a single loan application and returns the predicted probability of default.

**Request Body (JSON):**
*Requires origination fields only. Request must NOT contain post-origination fields (e.g., `total_pymnt`, `default_flag`, `loan_status`, `debt_settlement_flag`).*

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `loan_amnt` | Float (>0) | Yes | Requested loan amount |
| `term` | String | Yes | Loan term (e.g., " 36 months") |
| `int_rate` | Float (>0) | Yes | Interest rate |
| `installment` | Float (>0) | Yes | Monthly payment |
| `sub_grade` | String | Yes | Lending Club assigned subgrade |
| `home_ownership` | String | Yes | MORTGAGE, RENT, OWN, OTHER |
| `annual_inc` | Float ($\ge 0$) | Yes | Annual income |
| `verification_status`| String | Yes | Verified, Source Verified, Not Verified |
| `dti` | Float ($\ge 0$) | Yes | Debt-to-income ratio |
| `addr_state` | String | Yes | US State abbreviation |
| `purpose` | String | Yes | Loan purpose |
| `application_type` | String | Yes | Individual or Joint App |
| `fico_range_low` | Float (300-850) | Yes | FICO lower boundary |
| `fico_range_high` | Float (300-850) | Yes | FICO upper boundary ($\ge$ low) |
| `earliest_cr_line` | String (Mon-YYYY) | Yes | Earliest credit line |
| `issue_d` | String (Mon-YYYY) | Yes | Issue date |
| `total_acc` | Float ($\ge 0$) | Yes | Total credit lines |
| `open_acc` | Float ($\ge 0$) | Yes | Open credit lines |
| `revol_bal` | Float ($\ge 0$) | Yes | Revolving balance |
| `delinq_2yrs` | Float ($\ge 0$) | Yes | Delinquencies in last 2 years |
| `acc_now_delinq` | Float ($\ge 0$) | Yes | Accounts currently delinquent |
| `delinq_amnt` | Float ($\ge 0$) | Yes | Delinquent amount |
| `pub_rec` | Float ($\ge 0$) | Yes | Derogatory public records |
| `inq_last_6mths` | Float ($\ge 0$) | Yes | Inquiries in last 6 months |

*Note: Various other credit history fields (`mths_since_last_delinq`, `mort_acc`, etc.) are optional and will be deterministically imputed if missing.*

**Response (JSON):**
```json
{
  "raw_default_probability": 0.154,
  "calibrated_default_probability": 0.125,
  "model_name": "xgboost_baseline",
  "model_version": "P0.7",
  "calibration_method": "p09_calibrated_xgb_isotonic.joblib",
  "risk_band": "MODERATE",
  "disclaimer": "ML engineering portfolio implementation. Not a production lending system."
}
```

### 2. `GET /health`
Returns `{"status": "ok"}` if the API is running.

### 3. `GET /model-info`
Returns metadata regarding the loaded model, artifacts, and training phase.
