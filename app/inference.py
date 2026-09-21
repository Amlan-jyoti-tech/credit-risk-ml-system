import os
import joblib
import pandas as pd
from typing import Tuple

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FE_ARTIFACT = os.path.join(PROJECT_ROOT, "artifacts", "feature_engineer_xgb.joblib")
XGB_MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "models", "xgboost_baseline.joblib")
CALIBRATOR_PATH = os.path.join(PROJECT_ROOT, "results", "calibration", "p09_calibrated_xgb_isotonic.joblib")

# Global caches for loaded artifacts
_fe = None
_xgb_model = None
_calibrator = None

def load_artifacts():
    global _fe, _xgb_model, _calibrator
    if _fe is None:
        _fe = joblib.load(FE_ARTIFACT)
    if _xgb_model is None:
        _xgb_model = joblib.load(XGB_MODEL_PATH)
    if _calibrator is None:
        _calibrator = joblib.load(CALIBRATOR_PATH)

def predict_single_application(application_data: dict) -> Tuple[float, float]:
    """
    Executes the inference flow on a single application.
    Returns (raw_probability, calibrated_probability).
    """
    load_artifacts()
    
    # 1. Convert to DataFrame and replace None with np.nan
    import numpy as np
    df = pd.DataFrame([application_data])
    df.fillna(value=np.nan, inplace=True)
    # Ensure any remaining Nones are replaced (pandas sometimes keeps None for object dtype)
    df.replace({None: np.nan}, inplace=True)
    
    # 2. Add dummy id if missing (required by raw_selected in FE)
    if 'id' not in df.columns:
        df['id'] = 0
        
    # 3. Feature Engineering Transform
    df_transformed = _fe.transform(df, model_type='xgb')
    
    # 4. Enforce exact feature alignment
    expected_features = list(_xgb_model.feature_names_in_)
    if len(df_transformed.columns) < len(expected_features):
        missing = set(expected_features) - set(df_transformed.columns)
        raise ValueError(f"Transformed features are missing expected columns: {missing}")
        
    X_inference = df_transformed[expected_features].astype(float)
    
    # 5. Model Inference
    raw_prob = float(_xgb_model.predict_proba(X_inference)[:, 1][0])
    
    # 6. Isotonic Calibration
    calibrated_prob = float(_calibrator.predict([raw_prob])[0])
    
    return raw_prob, calibrated_prob
