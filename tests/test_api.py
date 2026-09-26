import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
import copy
from unittest.mock import patch

from app.main import app
import app.inference as inf

valid_payload = {
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
    "revol_bal": 5000.0,
    "revol_util": 50.0,
    "open_acc": 10,
    "total_acc": 20,
    "delinq_2yrs": 0.0,
    "acc_now_delinq": 0.0,
    "delinq_amnt": 0.0,
    "pub_rec": 0.0,
    "inq_last_6mths": 1.0
}

minimal_payload = {
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
}

# 11. /health
def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

# 12. /model-info
def test_model_info():
    with TestClient(app) as client:
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "xgboost_baseline"

# 1. complete valid payload
# 9. calibrated probability is within [0,1]
# 13. /predict
def test_predict_valid_input():
    with TestClient(app) as client:
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200, response.text
        data = response.json()
        assert "raw_default_probability" in data
        assert "calibrated_default_probability" in data
        assert "risk_band" in data
        assert "disclaimer" in data
        assert 0 <= data["raw_default_probability"] <= 1
        assert 0 <= data["calibrated_default_probability"] <= 1

# 2. minimal valid payload with optional fields omitted
def test_predict_minimal_payload():
    with TestClient(app) as client:
        # Ensures no 500 internal server error occurs (fixing B2)
        response = client.post("/predict", json=minimal_payload)
        assert response.status_code == 200, response.text
        
# 3. unknown field rejection
def test_unknown_field_rejection():
    with TestClient(app) as client:
        payload = copy.deepcopy(valid_payload)
        payload["fake_field"] = 123
        response = client.post("/predict", json=payload)
        assert response.status_code == 422 # Unprocessable Entity
        
# 4. leakage field rejection
def test_leakage_field_rejection():
    with TestClient(app) as client:
        payload = copy.deepcopy(valid_payload)
        payload["default_flag"] = 1
        payload["loan_status"] = "Charged Off"
        payload["total_pymnt"] = 1000.0
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert "Extra inputs are not permitted" in response.text

# 5. invalid categorical value rejection
def test_invalid_categorical():
    with TestClient(app) as client:
        payload = copy.deepcopy(valid_payload)
        payload["term"] = " 72 months"
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        
# 6. missing required field rejection
# 14. API error response for invalid request
def test_missing_required_field():
    with TestClient(app) as client:
        payload = copy.deepcopy(valid_payload)
        del payload["loan_amnt"]
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert "loan_amnt" in response.text

# 7. malformed date rejection
def test_malformed_date():
    with TestClient(app) as client:
        payload = copy.deepcopy(valid_payload)
        payload["issue_d"] = "2015-12-01"
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert "Dates must be in 'Mon-YYYY' format" in response.text

# 10. deterministic repeated prediction
def test_deterministic_repeated_prediction():
    with TestClient(app) as client:
        res1 = client.post("/predict", json=valid_payload).json()
        res2 = client.post("/predict", json=valid_payload).json()
        assert res1["calibrated_default_probability"] == res2["calibrated_default_probability"]
    
# 8. inference produces exactly 135 features and minimal payload NaN handling
def test_inference_internals():
    with TestClient(app) as client:
        # Client context loads the artifacts via lifespan
        import app.inference as inf
        
        # Test 1: Full payload
        df1 = pd.DataFrame([valid_payload])
        df1.fillna(value=np.nan, inplace=True)
        df1['id'] = 0
        df1 = df1.reindex(columns=inf.RAW_SELECTED)
        
        df_transformed1 = inf._fe.transform(df1, model_type='xgb')
        expected_features = list(inf._xgb_model.feature_names_in_)
        features1 = df_transformed1[expected_features]
        assert features1.shape[1] == 135
        assert list(features1.columns) == expected_features
        assert not features1.isnull().any().any()

        # Test 2: Minimal payload, explicitly verifying omitted optionals become NaN internally
        df2 = pd.DataFrame([minimal_payload])
        df2.fillna(value=np.nan, inplace=True)
        df2['id'] = 0
        df2 = df2.reindex(columns=inf.RAW_SELECTED)
        
        # verify that omitted fields like 'revol_util' are NaN in df2
        assert pd.isna(df2['revol_util'].iloc[0])
        assert pd.isna(df2['emp_length'].iloc[0])
        
        df_transformed2 = inf._fe.transform(df2, model_type='xgb')
        features2 = df_transformed2[expected_features]
        
        assert features2.shape[1] == 135
        assert list(features2.columns) == expected_features
        assert not features2.isnull().any().any()
        
# 15. artifact loading failure behavior
def test_artifact_loading_failure():
    # We patch the load_artifacts function to simulate failure
    with patch("app.main.load_artifacts", side_effect=Exception("Simulated load failure")):
        # Ensure we use a fresh state
        import app.main as am
        am._artifacts_loaded = False
        
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 503
            
            response = client.post("/predict", json=valid_payload)
            assert response.status_code == 503
