import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.inference import predict_single_application, _fe, _xgb_model

client = TestClient(app)

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

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "xgboost_baseline"

def test_predict_valid_input():
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert "raw_default_probability" in data
    assert "calibrated_default_probability" in data
    assert "risk_band" in data
    assert 0 <= data["raw_default_probability"] <= 1
    assert 0 <= data["calibrated_default_probability"] <= 1

def test_missing_required_field():
    payload = valid_payload.copy()
    del payload["loan_amnt"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "loan_amnt" in response.text

def test_missing_optional_field():
    payload = valid_payload.copy()
    if "revol_util" in payload:
        del payload["revol_util"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text

def test_invalid_numeric_input():
    payload = valid_payload.copy()
    payload["loan_amnt"] = "not_a_number"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_invalid_range():
    payload = valid_payload.copy()
    payload["loan_amnt"] = -100 # gt=0 is required
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_leakage_field_rejection():
    payload = valid_payload.copy()
    payload["default_flag"] = 1
    payload["loan_status"] = "Charged Off"
    payload["total_pymnt"] = 1000.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "Extra inputs are not permitted" in response.text

def test_deterministic_repeated_prediction():
    res1 = client.post("/predict", json=valid_payload).json()
    res2 = client.post("/predict", json=valid_payload).json()
    assert res1["calibrated_default_probability"] == res2["calibrated_default_probability"]
    
def test_inference_internals():
    # Test no NaN, exact feature count, exact feature order
    from app.inference import load_artifacts, _fe, _xgb_model
    load_artifacts()
    
    from app.schemas import LoanApplicationRequest
    df = pd.DataFrame([LoanApplicationRequest(**valid_payload).model_dump(exclude_none=False)])
    df.fillna(value=np.nan, inplace=True)
    df.replace({None: np.nan}, inplace=True)
    df['id'] = 0
    df_transformed = _fe.transform(df, model_type='xgb')
    
    expected_features = list(_xgb_model.feature_names_in_)
    features = df_transformed[expected_features]
    
    assert features.shape[1] == 135
    assert list(features.columns) == expected_features
    assert not features.isnull().any().any()
