import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    """
    P0.5 Approved Feature Engineering Pipeline.
    
    Handles:
    - Derivations (fico_score, credit_history_months, missingness flags)
    - Missing value imputation (learned from train)
    - Outlier capping (learned from train)
    - Log transforms for skewed dollars
    - Categorical encoding (one-hot vs ordinal)
    - Scaling for Logistic Regression
    
    Usage:
        fe = FeatureEngineer()
        # Ensure to drop 'default_flag' before passing to fit_transform!
        X_train_lr = fe.fit_transform(X_train, model_type='lr')
        X_val_lr = fe.transform(X_val, model_type='lr')
    """
    
    def __init__(self):
        self.is_fitted = False
        
        # Parameters learned during fit()
        self.imputation_medians = {}
        self.imputation_max_plus_one = {}
        self.capping_thresholds = {}
        self.scalers = {}  # One scaler per log-transformed/numeric column
        
        # Categorical mappings
        self.sub_grade_map = {
            f"{grade}{num}": i 
            for i, (grade, num) in enumerate(
                [(g, n) for g in 'ABCDEFG' for n in '12345']
            )
        }
        
        self.emp_length_map = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, 
            '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7, 
            '8 years': 8, '9 years': 9, '10+ years': 10
        }
        
        # Column definitions from P0.5 contract
        self.informative_missing = [
            'mths_since_last_delinq', 'mths_since_last_record', 
            'mths_since_last_major_derog', 'mths_since_recent_bc_dlq', 
            'mths_since_recent_revol_delinq'
        ]
        
        self.median_impute_cols = [
            'annual_inc', 'dti', 'revol_util', 'tot_cur_bal', 'total_rev_hi_lim',
            'tot_hi_cred_lim', 'avg_cur_bal', 'mths_since_recent_inq', 'mort_acc',
            'num_bc_tl', 'num_il_tl', 'num_actv_bc_tl', 'num_actv_rev_tl',
            'num_rev_accts', 'num_tl_op_past_12m', 'acc_open_past_24mths',
            'collections_12_mths_ex_med', 'chargeoff_within_12_mths', 
            'num_accts_ever_120_pd', 'num_tl_90g_dpd_24m', 'num_tl_30dpd', 
            'num_tl_120dpd_2m', 'pct_tl_nvr_dlq', 'bc_util', 'bc_open_to_buy', 
            'percent_bc_gt_75', 'total_bc_limit', 'total_bal_ex_mort', 
            'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_tl', 
            'mths_since_recent_bc', 'total_il_high_credit_limit', 
            'num_rev_tl_bal_gt_0', 'num_bc_sats'
        ]
        
        self.log_transform_cols = [
            'annual_inc', 'revol_bal', 'tot_cur_bal', 'avg_cur_bal',
            'bc_open_to_buy', 'total_bc_limit', 'total_bal_ex_mort',
            'tot_hi_cred_lim', 'total_il_high_credit_limit',
            'total_rev_hi_lim', 'tot_coll_amt'
        ]
        
        # Explicit categories for robust one-hot encoding
        self.categories = {
            'home_ownership': ['MORTGAGE', 'RENT', 'OWN', 'OTHER'],
            'verification_status': ['Source Verified', 'Verified', 'Not Verified'],
            'purpose': [
                'debt_consolidation', 'credit_card', 'home_improvement', 'other', 
                'major_purchase', 'small_business', 'car', 'medical', 'moving', 
                'vacation', 'house', 'wedding', 'renewable_energy', 'educational'
            ],
            'addr_state': [
                'CA', 'NY', 'TX', 'FL', 'IL', 'NJ', 'PA', 'OH', 'GA', 'VA', 
                'NC', 'MI', 'MD', 'MA', 'WA', 'CO', 'IN', 'TN', 'AZ', 'MO', 
                'MN', 'NV', 'SC', 'WI', 'OR', 'AL', 'CT', 'LA', 'UT', 'KY', 
                'OK', 'KS', 'AR', 'HI', 'NM', 'WV', 'NH', 'RI', 'MS', 'NE', 
                'MT', 'DE', 'AK', 'DC', 'WY', 'SD', 'VT', 'ME', 'ID', 'ND', 'IA'
            ],
            'sub_grade': [f"{g}{n}" for g in 'ABCDEFG' for n in '12345']
        }
        
        self.dummy_columns_lr = []
        self.dummy_columns_xgb = []
        
    def _validate_input(self, X):
        # Ensure no target or leakage in input
        leakage = ['debt_settlement_flag', 'last_fico_range_high'] # Representative leakage check
        if 'default_flag' in X.columns:
            raise ValueError("Target 'default_flag' found in features. Drop it before passing to FeatureEngineer.")
        for col in leakage:
            if col in X.columns:
                raise ValueError(f"Leakage column '{col}' found in features. Drop leakage columns before passing to FeatureEngineer.")

    def fit(self, X):
        self._validate_input(X)
        
        # 1. Learn median imputation values
        for col in self.median_impute_cols:
            if col in X.columns:
                self.imputation_medians[col] = X[col].median()
                
        # 2. Learn max+1 for informative missingness
        for col in self.informative_missing:
            if col in X.columns:
                self.imputation_max_plus_one[col] = X[col].max() + 1
                
        # 3. Learn capping thresholds (99.5th percentile)
        for col in ['annual_inc', 'dti']:
            if col in X.columns:
                self.capping_thresholds[col] = X[col].quantile(0.995)
                
        # 4. We need to fit scalers. We must simulate the transform on train
        # to properly fit the scalers on the transformed data.
        X_trans_lr = self._apply_transformations(X.copy(), is_fit=True, model_type='lr')
        self._apply_transformations(X.copy(), is_fit=True, model_type='xgb')
        
        # Fit scaler on all numeric columns (excluding binary/one-hot/ordinal/metadata)
        self.numeric_features = [
            c for c in X_trans_lr.columns 
            if X_trans_lr[c].nunique() > 2 and c not in ['sub_grade', 'emp_length', 'id', 'issue_d']
        ]
        
        for col in self.numeric_features:
            scaler = StandardScaler()
            # Handle potential NaNs in the extremely rare case they slip through
            vals = X_trans_lr[col].fillna(0).values.reshape(-1, 1)
            scaler.fit(vals)
            self.scalers[col] = scaler
            
        self.is_fitted = True
        return self

    def _apply_transformations(self, X, is_fit, model_type):
        df = X.copy()
        
        # 1. Derivations
        if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
            df['fico_score'] = (df['fico_range_low'] + df['fico_range_high']) / 2
            df = df.drop(columns=['fico_range_low', 'fico_range_high'])
            
        if 'earliest_cr_line' in df.columns and 'issue_d' in df.columns:
            issue_date = pd.to_datetime(df['issue_d'], format='%b-%Y')
            earliest_cr = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y')
            df['credit_history_months'] = ((issue_date - earliest_cr).dt.days / 30.44).astype(float)
            df = df.drop(columns=['earliest_cr_line'])
            
        # 2. Missingness Indicators
        if 'emp_length' in df.columns:
            df['emp_length_missing'] = df['emp_length'].isnull().astype(int)
        
        flag_names = {
            'mths_since_last_delinq': 'has_prior_delinq',
            'mths_since_last_record': 'has_public_record_history',
            'mths_since_last_major_derog': 'has_major_derog',
            'mths_since_recent_bc_dlq': 'has_bc_delinq',
            'mths_since_recent_revol_delinq': 'has_revol_delinq'
        }
        
        for col, flag_name in flag_names.items():
            if col in df.columns:
                # 1 if null (missing). The doc says:
                # has_prior_delinq (1 if non-null, 0 if null)
                # "YES — null means no delinquency."
                # I will adhere to: 1 if non-null (has delinquency), 0 if null (no delinquency)
                df[flag_name] = df[col].notnull().astype(int)
                
        # 3. Informative Imputation
        for col in self.informative_missing:
            if col in df.columns:
                impute_val = self.imputation_max_plus_one.get(col, 999) if model_type == 'lr' else -1
                df[col] = df[col].fillna(impute_val)
                
        # 4. Median / Zero Imputation
        for col in self.median_impute_cols:
            if col in df.columns:
                df[col] = df[col].fillna(self.imputation_medians.get(col, 0))
                
        if 'pub_rec_bankruptcies' in df.columns:
            df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].fillna(0)
        if 'tax_liens' in df.columns:
            df['tax_liens'] = df['tax_liens'].fillna(0)
        if 'tot_coll_amt' in df.columns:
            df['tot_coll_amt'] = df['tot_coll_amt'].fillna(0)
            
        # 5. Outlier Capping
        if 'annual_inc' in df.columns:
            cap = self.capping_thresholds.get('annual_inc', 1e6)
            df['annual_inc'] = df['annual_inc'].clip(upper=cap)
            
        if 'dti' in df.columns:
            cap = self.capping_thresholds.get('dti', 45.0)
            df['dti'] = df['dti'].clip(upper=cap)
            
        if 'revol_util' in df.columns:
            df['revol_util'] = df['revol_util'].clip(upper=100.0)
        if 'bc_util' in df.columns:
            df['bc_util'] = df['bc_util'].clip(upper=100.0)
            
        # 6. Log Transforms (Apply after capping/imputation)
        for col in self.log_transform_cols:
            if col in df.columns:
                df[col] = np.log1p(np.maximum(df[col], 0))
                
        # 7. Categorical Encoding - Binary
        if 'term' in df.columns:
            df['term'] = df['term'].str.strip().map({'36 months': 0, '60 months': 1}).fillna(0)
        if 'application_type' in df.columns:
            df['application_type'] = df['application_type'].map({'Individual': 0, 'Joint App': 1}).fillna(0)
        if 'initial_list_status' in df.columns:
            df['initial_list_status'] = df['initial_list_status'].map({'w': 0, 'f': 1}).fillna(0)
            
        # 8. Categorical Encoding - Ordinal
        if 'emp_length' in df.columns:
            df['emp_length'] = df['emp_length'].map(self.emp_length_map).fillna(5) # Median fallback
            
        if 'home_ownership' in df.columns:
            df['home_ownership'] = df['home_ownership'].replace(['ANY', 'NONE', 'OTHER'], 'OTHER')

        if model_type == 'xgb':
            if 'sub_grade' in df.columns:
                df['sub_grade'] = df['sub_grade'].map(self.sub_grade_map).fillna(17) # Median fallback
                
        # 9. Categorical Encoding - One-Hot
        categorical_cols = ['home_ownership', 'verification_status', 'purpose', 'addr_state']
        if model_type == 'lr':
            categorical_cols.append('sub_grade')
            
        # Convert to explicit CategoricalDtype to ensure exact number of dummies
        for col in categorical_cols:
            if col in df.columns:
                cat_type = pd.CategoricalDtype(categories=self.categories[col], ordered=False)
                df[col] = df[col].astype(cat_type)
            
        df = pd.get_dummies(df, columns=[c for c in categorical_cols if c in df.columns], drop_first=True, dtype=int)
        
        # Save or align dummy columns
        if is_fit:
            if model_type == 'lr':
                self.dummy_columns_lr = [c for c in df.columns if any(c.startswith(prefix + '_') for prefix in categorical_cols)]
            else:
                self.dummy_columns_xgb = [c for c in df.columns if any(c.startswith(prefix + '_') for prefix in categorical_cols)]
        else:
            expected_dummies = self.dummy_columns_lr if model_type == 'lr' else self.dummy_columns_xgb
            current_dummies = [c for c in df.columns if any(c.startswith(prefix + '_') for prefix in categorical_cols)]
            
            for col in expected_dummies:
                if col not in df.columns:
                    df[col] = 0
            for col in current_dummies:
                if col not in expected_dummies:
                    df = df.drop(columns=[col])
                    
        return df

    def transform(self, X, model_type='lr'):
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before calling transform.")
        
        self._validate_input(X)
        df = self._apply_transformations(X, is_fit=False, model_type=model_type)
        
        # 10. Scaling (For LR only)
        if model_type == 'lr':
            for col, scaler in self.scalers.items():
                if col in df.columns:
                    df[col] = scaler.transform(df[col].fillna(0).values.reshape(-1, 1)).flatten()
                    
        # Sort columns to ensure consistent ordering, keep metadata at the end
        cols = sorted(list(df.columns))
        metadata = [c for c in ['id', 'issue_d'] if c in cols]
        features = [c for c in cols if c not in metadata]
        
        return df[features + metadata]

    def fit_transform(self, X, model_type='lr'):
        return self.fit(X).transform(X, model_type=model_type)
