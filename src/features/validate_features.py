import pandas as pd
from src.features.build_features import FeatureEngineer

def run_validation():
    print("--- Feature Engineering Validation ---")
    PARQUET_PATH = r"d:\CREDIT_RISK_PREDIction-SYSTEM\data\processed\lending_club_target_cleaned.parquet"
    
    # Load a tiny subset just for validation
    df = pd.read_parquet(PARQUET_PATH).head(1000)
    
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
    
    # Select only approved predictors + metadata
    X = df[raw_selected].copy()
    
    fe = FeatureEngineer()
    
    print("\n[LR Preprocessing]")
    # We must fit first to learn LR dummies (and we need to fit XGB dummies too, so we'll do this carefully)
    # The current implementation fits dummies for whatever model_type is passed, but it actually needs to fit 
    # dummies for BOTH if we intend to reuse the same object. 
    # However, the normal usage is creating one object per model or passing data.
    # Let's instantiate two just to be safe.
    
    fe_lr = FeatureEngineer()
    X_lr = fe_lr.fit_transform(X, model_type='lr')
    print(f"LR Output Shape: {X_lr.shape}")
    
    fe_xgb = FeatureEngineer()
    X_xgb = fe_xgb.fit_transform(X, model_type='xgb')
    print(f"XGB Output Shape: {X_xgb.shape}")
    
    # Note: shape includes id and issue_d metadata columns.
    # We expect LR features = 168, plus 2 metadata = 170.
    # We expect XGB features = 135, plus 2 metadata = 137.
    print(f"\nLR expected dimensions (incl metadata): 170. Actual: {X_lr.shape[1]}")
    print(f"XGB expected dimensions (incl metadata): 137. Actual: {X_xgb.shape[1]}")
    
    # Check for target or leakage
    leakage = ['last_fico_range_high', 'debt_settlement_flag', 'default_flag']
    found = [c for c in leakage if c in X_lr.columns or c in X_xgb.columns]
    if found:
        print(f"FAILED: Leakage/Target columns found in output: {found}")
    else:
        print("SUCCESS: No leakage or target columns in output.")
        
    dup_lr = [c for c in X_lr.columns if list(X_lr.columns).count(c) > 1]
    if dup_lr:
        print("FAILED: Duplicate columns in LR")
    else:
        print("SUCCESS: No duplicate columns in LR")

if __name__ == "__main__":
    run_validation()
