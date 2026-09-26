import joblib
import pandas as pd
import numpy as np
from typing import Tuple

from app.config import FE_ARTIFACT, XGB_MODEL_PATH, CALIBRATOR_PATH

# Global caches for loaded artifacts
_fe = None
_xgb_model = None
_calibrator = None

RAW_SELECTED = [
    'id', 'issue_d', 'loan_amnt', 'term', 'int_rate', 'installment', 'sub_grade', 'initial_list_status', 'emp_length', 
    'home_ownership', 'annual_inc', 'verification_status', 'dti', 'addr_state', 'purpose', 'application_type', 
    'total_acc', 'open_acc', 'revol_bal', 'revol_util', 'tot_cur_bal', 'total_rev_hi_lim', 'tot_hi_cred_lim', 
    'avg_cur_bal', 'delinq_2yrs', 'mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog', 
    'mths_since_recent_bc_dlq', 'mths_since_recent_revol_delinq', 'acc_now_delinq', 'delinq_amnt', 'pub_rec', 
    'pub_rec_bankruptcies', 'tax_liens', 'inq_last_6mths', 'mths_since_recent_inq', 'mort_acc', 'num_bc_tl', 
    'num_il_tl', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_rev_accts', 'num_tl_op_past_12m', 'acc_open_past_24mths', 
    'collections_12_mths_ex_med', 'chargeoff_within_12_mths', 'num_accts_ever_120_pd', 'num_tl_90g_dpd_24m', 
    'num_tl_30dpd', 'num_tl_120dpd_2m', 'pct_tl_nvr_dlq', 'tot_coll_amt', 'bc_util', 'bc_open_to_buy', 
    'percent_bc_gt_75', 'total_bc_limit', 'total_bal_ex_mort', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 
    'mo_sin_rcnt_tl', 'mths_since_recent_bc', 'total_il_high_credit_limit', 'num_rev_tl_bal_gt_0', 'num_bc_sats',
    'fico_range_low', 'fico_range_high', 'earliest_cr_line'
]

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
    df = pd.DataFrame([application_data])
    df.fillna(value=np.nan, inplace=True)
    df.replace({None: np.nan}, inplace=True)
    
    # 2. Add dummy id if missing (required by raw_selected in FE)
    if 'id' not in df.columns:
        df['id'] = 0
        
    # 3. Defensive Re-index to ensure exact 68 expected raw columns are present
    # Missing columns will become NaN. Extra columns will be dropped.
    df = df.reindex(columns=RAW_SELECTED)
        
    # 4. Feature Engineering Transform
    df_transformed = _fe.transform(df, model_type='xgb')
    
    # 5. Enforce exact feature alignment
    expected_features = list(_xgb_model.feature_names_in_)
    missing = set(expected_features) - set(df_transformed.columns)
    if missing:
        raise ValueError(f"Transformed features mismatch. Missing: {missing}")
        
    X_inference = df_transformed[expected_features].astype(float)
    
    # 6. Model Inference
    raw_prob = float(_xgb_model.predict_proba(X_inference)[:, 1][0])
    
    # 7. Isotonic Calibration
    calibrated_prob = float(_calibrator.predict([raw_prob])[0])
    
    return raw_prob, calibrated_prob
