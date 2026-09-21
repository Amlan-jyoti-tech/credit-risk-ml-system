import os
import pandas as pd
import joblib
import sys

# Add project root to path so we can import src
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.features.build_features import FeatureEngineer

def serialize_feature_pipeline():
    print("=" * 60)
    print("PHASE 1: FEATURE PIPELINE RECONSTRUCTION & SERIALIZATION")
    print("=" * 60)
    
    # 1. Paths
    RAW_PARQUET = os.path.join(PROJECT_ROOT, "data", "processed", "lending_club_target_cleaned.parquet")
    XGB_MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "models", "xgboost_baseline.joblib")
    ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")
    FE_ARTIFACT_PATH = os.path.join(ARTIFACT_DIR, "feature_engineer_xgb.joblib")
    METADATA_PATH = os.path.join(ARTIFACT_DIR, "model_metadata.json")
    
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
    # 2. Load the raw dataset
    print(f"Loading raw dataset from {RAW_PARQUET}...")
    df = pd.read_parquet(RAW_PARQUET)
    print(f"Raw dataset shape: {df.shape}")
    
    # 3. Reconstruct P0.6 Train Split (Issue Years 2007-2014)
    print("Reconstructing exact P0.6 Train split (issue_d <= 2014)...")
    issue_date = pd.to_datetime(df['issue_d'], format='%b-%Y')
    train_mask = issue_date.dt.year <= 2014
    df_train = df[train_mask].copy()
    print(f"Train split shape: {df_train.shape}")
    
    # 4. Filter columns strictly to P0.5 predictors (raw_selected from validate_features.py)
    raw_selected = [
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
    
    missing_cols = [c for c in raw_selected if c not in df_train.columns]
    if missing_cols:
        raise ValueError(f"Missing required raw columns in dataset: {missing_cols}")
        
    X_train_raw = df_train[raw_selected].copy()
    
    # Ensure no leakage or target is passed
    if 'default_flag' in X_train_raw.columns:
        X_train_raw = X_train_raw.drop(columns=['default_flag'])
        
    # 5. Fit FeatureEngineer on exact train split
    print("Fitting FeatureEngineer on train split...")
    fe = FeatureEngineer()
    fe.fit(X_train_raw)
    
    # 6. Transform a small batch to check the final schema
    print("Transforming sample to verify schema...")
    X_trans_sample = fe.transform(X_train_raw.head(100), model_type='xgb')
    
    # Exclude metadata to get only features passed to XGBoost
    reconstructed_features = [c for c in X_trans_sample.columns if c not in ['id', 'issue_d']]
    
    # 7. MANDATORY REPRODUCIBILITY GATE
    print("Loading frozen XGBoost baseline to verify expected features...")
    xgb_model = joblib.load(XGB_MODEL_PATH)
    frozen_features = list(xgb_model.feature_names_in_)
    
    print(f"Reconstructed feature count: {len(reconstructed_features)}")
    print(f"Frozen model feature count: {len(frozen_features)}")
    
    if len(reconstructed_features) != 135:
        print("FAIL: Reconstructed feature count is not 135!")
        sys.exit(1)
        
    if reconstructed_features == frozen_features:
        print("\nSUCCESS: Reconstructed FeatureEngineer produces EXACTLY the same features (count, names, order)!")
    else:
        print("\nFAIL: Reconstructed features do NOT match frozen XGBoost expectations.")
        
        # Detailed mismatch analysis
        set_recon = set(reconstructed_features)
        set_frozen = set(frozen_features)
        
        missing_in_recon = set_frozen - set_recon
        extra_in_recon = set_recon - set_frozen
        
        if missing_in_recon:
            print(f"Missing in reconstruction: {missing_in_recon}")
        if extra_in_recon:
            print(f"Extra in reconstruction: {extra_in_recon}")
            
        if not missing_in_recon and not extra_in_recon:
            print("Feature sets match, but ORDER is different!")
            for i, (f_recon, f_frozen) in enumerate(zip(reconstructed_features, frozen_features)):
                if f_recon != f_frozen:
                    print(f"Mismatch at index {i}: Recon='{f_recon}' != Frozen='{f_frozen}'")
                    break
                    
        sys.exit(1)
        
    # 8. Verification of transformation parameters (sanity checks)
    assert fe.capping_thresholds.get('annual_inc') is not None, "annual_inc cap missing"
    assert fe.capping_thresholds.get('dti') is not None, "dti cap missing"
    assert 'term' in X_trans_sample.columns, "Categorical encoding (binary) missing"
    
    # 9. Serialize the successfully verified FeatureEngineer
    print(f"\nSerializing reconstructed FeatureEngineer to {FE_ARTIFACT_PATH}...")
    joblib.dump(fe, FE_ARTIFACT_PATH)
    
    # 10. Generate and save Metadata
    import json
    metadata = {
        "model_name": "xgboost_baseline",
        "model_type": "XGBClassifier",
        "training_phase": "P0.7",
        "calibration_artifact": "p09_calibrated_xgb_isotonic.joblib",
        "feature_pipeline_version": "P0.10",
        "python_version": sys.version,
        "scikit_learn_version": joblib.__version__ # Simplified for joblib/sklearn version check
    }
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print("Metadata saved.")
    print("=" * 60)
    print("PHASE 1 COMPLETE.")
    print("=" * 60)

if __name__ == "__main__":
    serialize_feature_pipeline()
