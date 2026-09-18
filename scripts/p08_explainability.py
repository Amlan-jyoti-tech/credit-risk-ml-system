"""
P0.8 — Model Explainability & Interpretation Script

This script produces explanations of model behavior for both Tier-1 baselines.
It does NOT retrain, refit, or alter any frozen artifact.

Frozen Artifacts (READ-ONLY):
  - src/models/logistic_regression_baseline.joblib
  - src/models/xgboost_baseline.joblib
  - data/processed/temporal_splits/val_lr.parquet
  - data/processed/temporal_splits/val_xgb.parquet
  - data/processed/temporal_splits/train_xgb.parquet  (read-only, for emp_length_missing investigation)
  - results/p07_results.json

Outputs:
  - results/feature_importance/*.csv
  - results/feature_importance/*.json
  - results/feature_importance/*.png
"""

import json
import warnings
import sys
import os
import gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================
# 0. PATHS & CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'src' / 'models'
DATA_DIR = PROJECT_ROOT / 'data' / 'processed' / 'temporal_splits'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'feature_importance'
P07_RESULTS = PROJECT_ROOT / 'results' / 'p07_results.json'

RANDOM_STATE = 42
PERM_N_REPEATS = 10
PERM_SCORING = 'roc_auc'

# Visualization settings
plt.rcParams.update({
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.edgecolor': '#CCCCCC',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
    'font.family': 'sans-serif',
})

# Feature metadata from P0.5 contract
LOG_TRANSFORM_COLS = [
    'annual_inc', 'revol_bal', 'tot_cur_bal', 'avg_cur_bal',
    'bc_open_to_buy', 'total_bc_limit', 'total_bal_ex_mort',
    'tot_hi_cred_lim', 'total_il_high_credit_limit',
    'total_rev_hi_lim', 'tot_coll_amt'
]

REFERENCE_CATEGORIES = {
    'home_ownership': 'MORTGAGE',
    'verification_status': 'Source Verified',
    'purpose': 'debt_consolidation',
    'addr_state': 'CA',
    'sub_grade': 'A1',
}

BINARY_FEATURES = ['term', 'application_type', 'initial_list_status']
ORDINAL_FEATURES_LR = ['emp_length']
ORDINAL_FEATURES_XGB = ['emp_length', 'sub_grade']
MISSINGNESS_FLAGS = [
    'emp_length_missing', 'has_prior_delinq', 'has_public_record_history',
    'has_major_derog', 'has_bc_delinq', 'has_revol_delinq'
]

ONE_HOT_PREFIXES = ['home_ownership_', 'verification_status_', 'purpose_', 'addr_state_', 'sub_grade_']
ONE_HOT_PREFIXES_XGB = ['home_ownership_', 'verification_status_', 'purpose_', 'addr_state_']

CAPPED_FEATURES = {
    'annual_inc': '99.5th pctl ($300K)',
    'dti': '99.5th pctl (36.27)',
    'revol_util': 'hard cap (100%)',
    'bc_util': 'hard cap (100%)',
}


def classify_feature_transformation(name, model_type='lr'):
    """Classify each feature by its transformation chain."""
    if name in MISSINGNESS_FLAGS:
        return 'binary_indicator'
    if name in BINARY_FEATURES:
        return 'binary'
    
    if model_type == 'lr':
        if any(name.startswith(p) for p in ONE_HOT_PREFIXES):
            return 'one_hot'
        if name in ORDINAL_FEATURES_LR:
            return 'ordinal'
    else:
        if any(name.startswith(p) for p in ONE_HOT_PREFIXES_XGB):
            return 'one_hot'
        if name in ORDINAL_FEATURES_XGB:
            return 'ordinal'
    
    if name in LOG_TRANSFORM_COLS:
        return 'log1p_scaled' if model_type == 'lr' else 'log1p_raw'
    
    return 'scaled' if model_type == 'lr' else 'raw'


def get_reference_category(name):
    """For one-hot features, return the reference (dropped) category."""
    for prefix, ref in REFERENCE_CATEGORIES.items():
        if name.startswith(prefix + '_'):
            return ref
    return None


def get_feature_group(name):
    """Classify feature into high-level groups."""
    if any(name.startswith(p) for p in ['addr_state_']):
        return 'geographic'
    if any(name.startswith(p) for p in ['sub_grade_']) or name == 'sub_grade':
        return 'credit_grade'
    if any(name.startswith(p) for p in ['purpose_']):
        return 'loan_purpose'
    if any(name.startswith(p) for p in ['home_ownership_']):
        return 'housing'
    if any(name.startswith(p) for p in ['verification_status_']):
        return 'verification'
    if name in MISSINGNESS_FLAGS:
        return 'missingness_indicator'
    if name in ['loan_amnt', 'term', 'int_rate', 'installment', 'initial_list_status']:
        return 'loan_terms'
    if name in ['annual_inc', 'dti', 'emp_length', 'application_type']:
        return 'borrower_profile'
    if name in ['fico_score', 'credit_history_months']:
        return 'derived'
    return 'credit_bureau'


# ============================================================
# 1. ENVIRONMENT VALIDATION
# ============================================================

def validate_environment():
    """Load models and data, verify dimensions and metrics match P0.7."""
    print("=" * 70)
    print("STEP 1: ENVIRONMENT VALIDATION")
    print("=" * 70)
    
    # Load frozen models
    lr_model = joblib.load(MODELS_DIR / 'logistic_regression_baseline.joblib')
    xgb_model = joblib.load(MODELS_DIR / 'xgboost_baseline.joblib')
    
    # Verify dimensions
    assert lr_model.coef_.shape == (1, 168), f"LR coef shape mismatch: {lr_model.coef_.shape}"
    assert lr_model.n_features_in_ == 168, f"LR n_features mismatch: {lr_model.n_features_in_}"
    assert xgb_model.n_features_in_ == 135, f"XGB n_features mismatch: {xgb_model.n_features_in_}"
    print(f"  [PASS] LR dimensions: {lr_model.n_features_in_}")
    print(f"  [PASS] XGB dimensions: {xgb_model.n_features_in_}")
    
    # Load validation data
    val_lr_df = pd.read_parquet(DATA_DIR / 'val_lr.parquet')
    val_xgb_df = pd.read_parquet(DATA_DIR / 'val_xgb.parquet')
    
    # Separate features and target
    lr_feature_names = list(lr_model.feature_names_in_)
    xgb_feature_names = list(xgb_model.feature_names_in_)
    
    # Extract target and feature DataFrames (keep as DataFrame for feature name support)
    if 'default_flag' not in val_lr_df.columns:
        raise ValueError("default_flag not found in validation parquet")
    
    y_val = val_lr_df['default_flag'].values
    X_val_lr = val_lr_df[lr_feature_names]  # DataFrame, not .values
    X_val_xgb = val_xgb_df[xgb_feature_names]  # DataFrame, not .values
    
    print(f"  [PASS] Val LR shape: {X_val_lr.shape}")
    print(f"  [PASS] Val XGB shape: {X_val_xgb.shape}")
    
    # Verify predictions match P0.7
    with open(P07_RESULTS, 'r') as f:
        p07 = json.load(f)
    
    lr_val_proba = lr_model.predict_proba(X_val_lr)[:, 1]
    xgb_val_proba = xgb_model.predict_proba(X_val_xgb)[:, 1]
    
    lr_val_auc = roc_auc_score(y_val, lr_val_proba)
    xgb_val_auc = roc_auc_score(y_val, xgb_val_proba)
    
    lr_expected_auc = p07['LR']['val_metrics']['roc_auc']
    xgb_expected_auc = p07['XGB']['val_metrics']['roc_auc']
    
    assert abs(lr_val_auc - lr_expected_auc) < 1e-6, \
        f"LR val AUC mismatch: {lr_val_auc:.10f} vs {lr_expected_auc:.10f}"
    assert abs(xgb_val_auc - xgb_expected_auc) < 1e-6, \
        f"XGB val AUC mismatch: {xgb_val_auc:.10f} vs {xgb_expected_auc:.10f}"
    
    print(f"  [PASS] LR val ROC-AUC: {lr_val_auc:.10f} matches P0.7: {lr_expected_auc:.10f}")
    print(f"  [PASS] XGB val ROC-AUC: {xgb_val_auc:.10f} matches P0.7: {xgb_expected_auc:.10f}")
    
    # Verify test set is NOT loaded
    print(f"  [PASS] Test set files NOT loaded (by design)")
    
    # Log library versions
    import sklearn
    import xgboost
    print(f"\n  Library versions:")
    print(f"    scikit-learn: {sklearn.__version__}")
    print(f"    xgboost: {xgboost.__version__}")
    print(f"    numpy: {np.__version__}")
    print(f"    pandas: {pd.__version__}")
    print(f"    matplotlib: {matplotlib.__version__}")
    
    print(f"\n  [ALL CHECKS PASSED]")
    
    return lr_model, xgb_model, X_val_lr, X_val_xgb, y_val, lr_feature_names, xgb_feature_names


# ============================================================
# 2. LR COEFFICIENT EXTRACTION
# ============================================================

def extract_lr_coefficients(lr_model, lr_feature_names):
    """Extract and annotate all LR coefficients."""
    print("\n" + "=" * 70)
    print("STEP 2: LR COEFFICIENT EXTRACTION")
    print("=" * 70)
    
    coefs = lr_model.coef_[0]
    intercept = lr_model.intercept_[0]
    
    records = []
    for i, name in enumerate(lr_feature_names):
        records.append({
            'feature_name': name,
            'coefficient': coefs[i],
            'abs_coefficient': abs(coefs[i]),
            'sign': '+' if coefs[i] >= 0 else '-',
            'transformation': classify_feature_transformation(name, 'lr'),
            'reference_category': get_reference_category(name),
            'feature_group': get_feature_group(name),
            'capping': CAPPED_FEATURES.get(name, None),
        })
    
    df = pd.DataFrame(records)
    df = df.sort_values('abs_coefficient', ascending=False).reset_index(drop=True)
    df['rank_by_abs'] = df.index + 1
    
    # Save
    df.to_csv(RESULTS_DIR / 'lr_coefficients.csv', index=False)
    
    # JSON version
    lr_json = {
        'intercept': intercept,
        'n_features': len(lr_feature_names),
        'coefficients': df.to_dict(orient='records'),
    }
    with open(RESULTS_DIR / 'lr_coefficients.json', 'w') as f:
        json.dump(lr_json, f, indent=2)
    
    print(f"  Saved lr_coefficients.csv ({len(df)} rows)")
    print(f"  Saved lr_coefficients.json")
    print(f"  Intercept: {intercept:.6f}")
    print(f"\n  Top 10 by |coefficient|:")
    for _, row in df.head(10).iterrows():
        print(f"    {row['rank_by_abs']:3d}. {row['feature_name']:45s}  {row['coefficient']:+.6f}  ({row['transformation']})")
    
    return df


# ============================================================
# 3. LR VISUALIZATIONS
# ============================================================

def create_lr_visualizations(lr_df):
    """Generate LR coefficient plots."""
    print("\n" + "=" * 70)
    print("STEP 3: LR VISUALIZATIONS")
    print("=" * 70)
    
    # --- Plot A: Top 30 features by |coefficient| ---
    top30 = lr_df.head(30).copy()
    top30 = top30.sort_values('abs_coefficient', ascending=True)  # For horizontal bar
    
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ['#E74C3C' if c > 0 else '#2E86C1' for c in top30['coefficient']]
    bars = ax.barh(range(len(top30)), top30['coefficient'], color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30['feature_name'], fontsize=8)
    ax.set_xlabel('Coefficient (change in log-odds per unit)', fontsize=11)
    ax.set_title('Logistic Regression — Top 30 Features by |Coefficient|\n(Standardized features; red = increases P(default), blue = decreases)', 
                 fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)
    
    # Add value labels
    for bar, val in zip(bars, top30['coefficient']):
        x_pos = val + 0.02 if val >= 0 else val - 0.02
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.3f}', 
                va='center', ha=ha, fontsize=7, color='#333333')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'lr_top30_barplot.png', bbox_inches='tight')
    plt.close()
    print(f"  Saved lr_top30_barplot.png")
    
    # --- Plot B: Sub-grade coefficient gradient ---
    sg_rows = lr_df[lr_df['feature_name'].str.startswith('sub_grade_')].copy()
    # Sort by sub_grade order (A2, A3, ..., G5)
    grade_order = [f"sub_grade_{g}{n}" for g in 'ABCDEFG' for n in '12345']
    sg_rows['sort_key'] = sg_rows['feature_name'].map(
        {name: i for i, name in enumerate(grade_order)}
    )
    sg_rows = sg_rows.dropna(subset=['sort_key']).sort_values('sort_key')
    
    fig, ax = plt.subplots(figsize=(14, 5))
    x = range(len(sg_rows))
    # Color by grade letter
    grade_colors = {'A': '#27AE60', 'B': '#2ECC71', 'C': '#F1C40F', 
                    'D': '#E67E22', 'E': '#E74C3C', 'F': '#C0392B', 'G': '#8E44AD'}
    bar_colors = [grade_colors.get(name.split('_')[1][0], '#999') for name in sg_rows['feature_name']]
    
    ax.bar(x, sg_rows['coefficient'], color=bar_colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('sub_grade_', '') for n in sg_rows['feature_name']], 
                       rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Coefficient (relative to A1 reference)', fontsize=11)
    ax.set_title('LR Sub-Grade Coefficients — Risk Gradient Relative to A1 (Dropped Reference)\n'
                 'Expected: monotonically increasing from A→G', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.8, label='A1 (reference)')
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'lr_subgrade_coefficients.png', bbox_inches='tight')
    plt.close()
    print(f"  Saved lr_subgrade_coefficients.png")
    
    # --- Plot C: Purpose coefficients ---
    purpose_rows = lr_df[lr_df['feature_name'].str.startswith('purpose_')].copy()
    purpose_rows = purpose_rows.sort_values('coefficient', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#E74C3C' if c > 0 else '#2E86C1' for c in purpose_rows['coefficient']]
    bars = ax.barh(range(len(purpose_rows)), purpose_rows['coefficient'], color=colors, 
                   edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(purpose_rows)))
    ax.set_yticklabels([n.replace('purpose_', '') for n in purpose_rows['feature_name']], fontsize=9)
    ax.set_xlabel('Coefficient (relative to debt_consolidation reference)', fontsize=11)
    ax.set_title('LR Purpose Coefficients — Effect Relative to Debt Consolidation\n'
                 'Red = higher default risk than debt_consolidation, Blue = lower', 
                 fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8, label='debt_consolidation (reference)')
    ax.legend(fontsize=9)
    
    for bar, val in zip(bars, purpose_rows['coefficient']):
        x_pos = val + 0.01 if val >= 0 else val - 0.01
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.3f}', 
                va='center', ha=ha, fontsize=8, color='#333333')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'lr_purpose_coefficients.png', bbox_inches='tight')
    plt.close()
    print(f"  Saved lr_purpose_coefficients.png")


# ============================================================
# 4. XGB IMPORTANCE EXTRACTION
# ============================================================

def extract_xgb_importance(xgb_model, xgb_feature_names):
    """Extract gain and weight importance from XGB booster."""
    print("\n" + "=" * 70)
    print("STEP 4: XGB IMPORTANCE EXTRACTION")
    print("=" * 70)
    
    booster = xgb_model.get_booster()
    
    gain_imp = booster.get_score(importance_type='gain')
    weight_imp = booster.get_score(importance_type='weight')
    
    records = []
    for name in xgb_feature_names:
        records.append({
            'feature_name': name,
            'gain_importance': gain_imp.get(name, 0.0),
            'weight_importance': weight_imp.get(name, 0.0),
            'zero_importance': name not in gain_imp,
            'transformation': classify_feature_transformation(name, 'xgb'),
            'feature_group': get_feature_group(name),
        })
    
    df = pd.DataFrame(records)
    df = df.sort_values('gain_importance', ascending=False).reset_index(drop=True)
    df['gain_rank'] = range(1, len(df) + 1)
    
    # Weight rank
    df_weight_sorted = df.sort_values('weight_importance', ascending=False).reset_index(drop=True)
    weight_rank_map = {row['feature_name']: i + 1 for i, row in df_weight_sorted.iterrows()}
    df['weight_rank'] = df['feature_name'].map(weight_rank_map)
    df['rank_discrepancy'] = abs(df['gain_rank'] - df['weight_rank'])
    
    # Normalize gain to sum to 1 for easier comparison
    total_gain = df['gain_importance'].sum()
    df['gain_importance_normalized'] = df['gain_importance'] / total_gain if total_gain > 0 else 0
    
    # Save
    df.to_csv(RESULTS_DIR / 'xgb_gain_importance.csv', index=False)
    
    xgb_json = {
        'n_features': len(xgb_feature_names),
        'n_zero_importance': int(df['zero_importance'].sum()),
        'zero_importance_features': list(df[df['zero_importance']]['feature_name']),
        'importances': df.to_dict(orient='records'),
    }
    with open(RESULTS_DIR / 'xgb_gain_importance.json', 'w') as f:
        json.dump(xgb_json, f, indent=2)
    
    print(f"  Saved xgb_gain_importance.csv ({len(df)} rows)")
    print(f"  Saved xgb_gain_importance.json")
    print(f"  Zero-importance features: {int(df['zero_importance'].sum())}")
    print(f"\n  Top 10 by gain:")
    for _, row in df.head(10).iterrows():
        print(f"    {row['gain_rank']:3d}. {row['feature_name']:45s}  gain={row['gain_importance']:.2f}  weight_rank={row['weight_rank']}")
    
    return df


# ============================================================
# 5. XGB VISUALIZATIONS
# ============================================================

def create_xgb_visualizations(xgb_df):
    """Generate XGB importance plots."""
    print("\n" + "=" * 70)
    print("STEP 5: XGB VISUALIZATIONS")
    print("=" * 70)
    
    # --- Plot A: Top 30 by gain ---
    top30 = xgb_df.head(30).copy()
    top30 = top30.sort_values('gain_importance', ascending=True)
    
    # Color by feature group
    group_colors = {
        'credit_grade': '#8E44AD',
        'loan_terms': '#2E86C1',
        'borrower_profile': '#27AE60',
        'credit_bureau': '#E67E22',
        'derived': '#E74C3C',
        'geographic': '#95A5A6',
        'loan_purpose': '#F1C40F',
        'housing': '#1ABC9C',
        'verification': '#3498DB',
        'missingness_indicator': '#D35400',
    }
    colors = [group_colors.get(g, '#999') for g in top30['feature_group']]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(range(len(top30)), top30['gain_importance'], color=colors, 
                   edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30['feature_name'], fontsize=8)
    ax.set_xlabel('Mean Gain (loss reduction per split)', fontsize=11)
    ax.set_title('XGBoost — Top 30 Features by Gain Importance\n'
                 '(Higher gain = greater contribution to prediction accuracy)', 
                 fontsize=12, fontweight='bold')
    
    # Legend for groups
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=g) for g, c in group_colors.items() 
                       if g in top30['feature_group'].values]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, title='Feature Group')
    
    for bar, val in zip(bars, top30['gain_importance']):
        ax.text(val + 5, bar.get_y() + bar.get_height()/2, f'{val:.1f}', 
                va='center', ha='left', fontsize=7, color='#333333')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'xgb_top30_gain_barplot.png', bbox_inches='tight')
    plt.close()
    print(f"  Saved xgb_top30_gain_barplot.png")
    
    # --- Plot B: Gain rank vs weight rank scatter ---
    non_zero = xgb_df[~xgb_df['zero_importance']].copy()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(non_zero['gain_rank'], non_zero['weight_rank'], 
               alpha=0.5, s=30, c='#2E86C1', edgecolors='white', linewidth=0.3)
    
    # Annotate high-discrepancy features
    high_disc = non_zero[non_zero['rank_discrepancy'] > 30]
    for _, row in high_disc.iterrows():
        ax.annotate(row['feature_name'], (row['gain_rank'], row['weight_rank']),
                    fontsize=6, alpha=0.8, 
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5),
                    xytext=(5, 5), textcoords='offset points')
    
    # Diagonal line (perfect agreement)
    max_rank = max(non_zero['gain_rank'].max(), non_zero['weight_rank'].max())
    ax.plot([1, max_rank], [1, max_rank], 'k--', alpha=0.3, label='Perfect agreement')
    
    ax.set_xlabel('Gain Rank (higher rank = less important)', fontsize=11)
    ax.set_ylabel('Weight Rank (higher rank = less frequently split)', fontsize=11)
    ax.set_title('XGBoost — Gain Rank vs. Weight Rank\n'
                 'Features far from diagonal have discrepant importance measures', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'xgb_gain_vs_weight_scatter.png', bbox_inches='tight')
    plt.close()
    print(f"  Saved xgb_gain_vs_weight_scatter.png")


# ============================================================
# 6. PERMUTATION IMPORTANCE
# ============================================================

def compute_permutation_importance(lr_model, xgb_model, X_val_lr, X_val_xgb, y_val,
                                    lr_feature_names, xgb_feature_names):
    """Compute permutation importance for both models on validation set.
    
    Uses a stratified subsample of 50K rows to fit within available memory,
    with n_jobs=1 to avoid spawning parallel processes.
    """
    print("\n" + "=" * 70)
    print("STEP 6: PERMUTATION IMPORTANCE")
    print("=" * 70)
    
    # Subsample for memory efficiency (50K rows, stratified by target)
    SUBSAMPLE_N = 50000
    rng = np.random.RandomState(RANDOM_STATE)
    
    if len(y_val) > SUBSAMPLE_N:
        # Stratified subsample
        idx_pos = np.where(y_val == 1)[0]
        idx_neg = np.where(y_val == 0)[0]
        pos_ratio = len(idx_pos) / len(y_val)
        n_pos = int(SUBSAMPLE_N * pos_ratio)
        n_neg = SUBSAMPLE_N - n_pos
        
        sampled_pos = rng.choice(idx_pos, size=min(n_pos, len(idx_pos)), replace=False)
        sampled_neg = rng.choice(idx_neg, size=min(n_neg, len(idx_neg)), replace=False)
        subsample_idx = np.sort(np.concatenate([sampled_pos, sampled_neg]))
        
        X_sub_xgb = X_val_xgb.iloc[subsample_idx].copy()
        X_sub_lr = X_val_lr.iloc[subsample_idx].copy()
        y_sub = y_val[subsample_idx]
        print(f"  Subsampled {SUBSAMPLE_N} rows (stratified) from {len(y_val)} for permutation importance")
        print(f"  Subsample default rate: {y_sub.mean():.4f} (full: {y_val.mean():.4f})")
    else:
        X_sub_xgb = X_val_xgb
        X_sub_lr = X_val_lr
        y_sub = y_val
    
    # --- XGB Permutation Importance ---
    print("  Computing XGB permutation importance (n_jobs=1, sequential)...", flush=True)
    xgb_perm = permutation_importance(
        xgb_model, X_sub_xgb, y_sub,
        n_repeats=PERM_N_REPEATS,
        random_state=RANDOM_STATE,
        scoring=PERM_SCORING,
        n_jobs=1  # Sequential to avoid memory issues
    )
    
    xgb_perm_df = pd.DataFrame({
        'feature_name': xgb_feature_names,
        'perm_importance_mean': xgb_perm.importances_mean,
        'perm_importance_std': xgb_perm.importances_std,
    }).sort_values('perm_importance_mean', ascending=False).reset_index(drop=True)
    xgb_perm_df['perm_rank'] = range(1, len(xgb_perm_df) + 1)
    
    xgb_perm_df.to_csv(RESULTS_DIR / 'permutation_importance_xgb.csv', index=False)
    print(f"  Saved permutation_importance_xgb.csv")
    
    print(f"\n  XGB Top 10 by permutation importance:")
    for _, row in xgb_perm_df.head(10).iterrows():
        print(f"    {row['perm_rank']:3d}. {row['feature_name']:45s}  "
              f"mean={row['perm_importance_mean']:.6f} ± {row['perm_importance_std']:.6f}")
    
    # --- LR Permutation Importance ---
    print("\n  Computing LR permutation importance (n_jobs=1, sequential)...", flush=True)
    lr_perm = permutation_importance(
        lr_model, X_sub_lr, y_sub,
        n_repeats=PERM_N_REPEATS,
        random_state=RANDOM_STATE,
        scoring=PERM_SCORING,
        n_jobs=1  # Sequential to avoid memory issues
    )
    
    lr_perm_df = pd.DataFrame({
        'feature_name': lr_feature_names,
        'perm_importance_mean': lr_perm.importances_mean,
        'perm_importance_std': lr_perm.importances_std,
    }).sort_values('perm_importance_mean', ascending=False).reset_index(drop=True)
    lr_perm_df['perm_rank'] = range(1, len(lr_perm_df) + 1)
    
    lr_perm_df.to_csv(RESULTS_DIR / 'permutation_importance_lr.csv', index=False)
    print(f"  Saved permutation_importance_lr.csv")
    
    print(f"\n  LR Top 10 by permutation importance:")
    for _, row in lr_perm_df.head(10).iterrows():
        print(f"    {row['perm_rank']:3d}. {row['feature_name']:45s}  "
              f"mean={row['perm_importance_mean']:.6f} ± {row['perm_importance_std']:.6f}")
    
    # --- Permutation importance visualization (XGB) ---
    top30_perm = xgb_perm_df.head(30).copy()
    top30_perm = top30_perm.sort_values('perm_importance_mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(range(len(top30_perm)), top30_perm['perm_importance_mean'], 
            xerr=top30_perm['perm_importance_std'],
            color='#2E86C1', edgecolor='white', linewidth=0.5, capsize=2, ecolor='#95A5A6')
    ax.set_yticks(range(len(top30_perm)))
    ax.set_yticklabels(top30_perm['feature_name'], fontsize=8)
    ax.set_xlabel('Mean decrease in ROC-AUC when feature is permuted', fontsize=11)
    ax.set_title('XGBoost — Top 30 Features by Permutation Importance (Validation Set)\n'
                 f'(10 repeats, seed={RANDOM_STATE}, scoring=roc_auc)', 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'permutation_top30_barplot.png', bbox_inches='tight')
    plt.close()
    print(f"  Saved permutation_top30_barplot.png")
    
    return lr_perm_df, xgb_perm_df


# ============================================================
# 7. CROSS-MODEL COMPARISON
# ============================================================

def cross_model_comparison(lr_coef_df, xgb_imp_df, lr_perm_df, xgb_perm_df,
                            lr_feature_names, xgb_feature_names):
    """Compare feature importance across LR and XGB."""
    print("\n" + "=" * 70)
    print("STEP 7: CROSS-MODEL COMPARISON")
    print("=" * 70)
    
    # For cross-model comparison, we need to handle the encoding differences:
    # LR has 34 sub_grade dummies → aggregate to single "sub_grade" concept
    # XGB has 1 ordinal sub_grade
    
    # Create LR conceptual importance by aggregating one-hot groups
    lr_conceptual = []
    
    # Track which features have been grouped
    grouped_features = set()
    
    # Aggregate one-hot groups
    for prefix, ref_cat in REFERENCE_CATEGORIES.items():
        if prefix == 'sub_grade':
            # Only applies to LR (XGB uses ordinal)
            dummy_rows = lr_coef_df[lr_coef_df['feature_name'].str.startswith(prefix + '_')]
            if len(dummy_rows) > 0:
                lr_conceptual.append({
                    'conceptual_feature': prefix,
                    'lr_max_abs_coef': dummy_rows['abs_coefficient'].max(),
                    'lr_mean_abs_coef': dummy_rows['abs_coefficient'].mean(),
                    'lr_n_dummies': len(dummy_rows),
                    'lr_best_rank': dummy_rows['rank_by_abs'].min(),
                })
                grouped_features.update(dummy_rows['feature_name'].tolist())
        else:
            # Both models have same one-hot encoding for these
            dummy_rows_lr = lr_coef_df[lr_coef_df['feature_name'].str.startswith(prefix + '_')]
            if len(dummy_rows_lr) > 0:
                lr_conceptual.append({
                    'conceptual_feature': prefix,
                    'lr_max_abs_coef': dummy_rows_lr['abs_coefficient'].max(),
                    'lr_mean_abs_coef': dummy_rows_lr['abs_coefficient'].mean(),
                    'lr_n_dummies': len(dummy_rows_lr),
                    'lr_best_rank': dummy_rows_lr['rank_by_abs'].min(),
                })
                grouped_features.update(dummy_rows_lr['feature_name'].tolist())
    
    # Add non-grouped LR features
    for _, row in lr_coef_df.iterrows():
        if row['feature_name'] not in grouped_features:
            lr_conceptual.append({
                'conceptual_feature': row['feature_name'],
                'lr_max_abs_coef': row['abs_coefficient'],
                'lr_mean_abs_coef': row['abs_coefficient'],
                'lr_n_dummies': 1,
                'lr_best_rank': row['rank_by_abs'],
            })
    
    lr_conceptual_df = pd.DataFrame(lr_conceptual)
    lr_conceptual_df = lr_conceptual_df.sort_values('lr_max_abs_coef', ascending=False).reset_index(drop=True)
    lr_conceptual_df['lr_conceptual_rank'] = range(1, len(lr_conceptual_df) + 1)
    
    # XGB conceptual importance (aggregate one-hot groups)
    xgb_conceptual = []
    xgb_grouped = set()
    
    for prefix in ['home_ownership', 'verification_status', 'purpose', 'addr_state']:
        dummy_rows = xgb_imp_df[xgb_imp_df['feature_name'].str.startswith(prefix + '_')]
        if len(dummy_rows) > 0:
            xgb_conceptual.append({
                'conceptual_feature': prefix,
                'xgb_total_gain': dummy_rows['gain_importance'].sum(),
                'xgb_max_gain': dummy_rows['gain_importance'].max(),
                'xgb_n_dummies': len(dummy_rows),
                'xgb_best_gain_rank': dummy_rows['gain_rank'].min(),
            })
            xgb_grouped.update(dummy_rows['feature_name'].tolist())
    
    for _, row in xgb_imp_df.iterrows():
        if row['feature_name'] not in xgb_grouped:
            xgb_conceptual.append({
                'conceptual_feature': row['feature_name'],
                'xgb_total_gain': row['gain_importance'],
                'xgb_max_gain': row['gain_importance'],
                'xgb_n_dummies': 1,
                'xgb_best_gain_rank': row['gain_rank'],
            })
    
    xgb_conceptual_df = pd.DataFrame(xgb_conceptual)
    xgb_conceptual_df = xgb_conceptual_df.sort_values('xgb_total_gain', ascending=False).reset_index(drop=True)
    xgb_conceptual_df['xgb_conceptual_rank'] = range(1, len(xgb_conceptual_df) + 1)
    
    # Merge
    comparison = pd.merge(
        lr_conceptual_df[['conceptual_feature', 'lr_max_abs_coef', 'lr_conceptual_rank', 'lr_n_dummies']],
        xgb_conceptual_df[['conceptual_feature', 'xgb_total_gain', 'xgb_conceptual_rank', 'xgb_n_dummies']],
        on='conceptual_feature', how='outer',
        suffixes=('_lr', '_xgb')
    )
    
    # Fill missing ranks with max+1
    max_lr = comparison['lr_conceptual_rank'].max()
    max_xgb = comparison['xgb_conceptual_rank'].max()
    comparison['lr_conceptual_rank'] = comparison['lr_conceptual_rank'].fillna(max_lr + 1)
    comparison['xgb_conceptual_rank'] = comparison['xgb_conceptual_rank'].fillna(max_xgb + 1)
    comparison['rank_agreement'] = abs(comparison['lr_conceptual_rank'] - comparison['xgb_conceptual_rank'])
    
    comparison = comparison.sort_values('lr_conceptual_rank').reset_index(drop=True)
    
    comparison.to_csv(RESULTS_DIR / 'cross_model_comparison.csv', index=False)
    print(f"  Saved cross_model_comparison.csv ({len(comparison)} conceptual features)")
    
    # Show top 20
    print(f"\n  Top 20 conceptual features (ranked by LR):")
    print(f"  {'Feature':30s} {'LR Rank':>8s} {'XGB Rank':>9s} {'Agreement':>10s}")
    print(f"  {'-'*30} {'-'*8} {'-'*9} {'-'*10}")
    for _, row in comparison.head(20).iterrows():
        print(f"  {row['conceptual_feature']:30s} {row['lr_conceptual_rank']:8.0f} "
              f"{row['xgb_conceptual_rank']:9.0f} {row['rank_agreement']:10.0f}")
    
    # --- Visualization ---
    top20_both = comparison.head(25).copy()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(top20_both))
    width = 0.35
    
    # Normalize for visual comparison
    lr_vals = top20_both['lr_conceptual_rank'].values
    xgb_vals = top20_both['xgb_conceptual_rank'].values
    
    bars1 = ax.barh(x + width/2, lr_vals, width, label='LR Rank (lower=more important)', 
                    color='#2E86C1', alpha=0.8)
    bars2 = ax.barh(x - width/2, xgb_vals, width, label='XGB Rank (lower=more important)', 
                    color='#E74C3C', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(top20_both['conceptual_feature'], fontsize=8)
    ax.set_xlabel('Rank (lower = more important)', fontsize=11)
    ax.set_title('Cross-Model Feature Importance Comparison\n'
                 'LR (|coefficient| rank, aggregated) vs XGB (gain rank, aggregated)', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.invert_xaxis()  # Lower rank = more important, show on right
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'cross_model_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"  Saved cross_model_comparison.png")
    
    return comparison


# ============================================================
# 8. PLAUSIBILITY AUDIT & EMP_LENGTH_MISSING INVESTIGATION
# ============================================================

def plausibility_audit(lr_df, xgb_df, xgb_perm_df):
    """Assess credit-risk plausibility of top features and investigate suspicious ones."""
    print("\n" + "=" * 70)
    print("STEP 8: PLAUSIBILITY AUDIT")
    print("=" * 70)
    
    # Define expected patterns
    plausibility_checks = {
        # feature_name: (expected_sign_lr_or_None, expected_importance_xgb, note)
        'term': ('+', 'high', '60m term → longer exposure → more default'),
        'int_rate': ('+', 'high', 'Higher rate = higher risk borrower'),
        'annual_inc': ('-', 'high', 'Higher income → better repayment capacity'),
        'fico_score': ('-', 'high', 'Higher FICO → better credit history'),
        'dti': ('+', 'high', 'Higher DTI → more debt burden'),
        'revol_util': ('+', 'medium', 'Higher utilization → more credit stress'),
        'emp_length': ('-', 'low-medium', 'Longer employment → more stability'),
    }
    
    audit_results = []
    
    # Check LR plausibility
    print("\n  LR Plausibility Check (non-one-hot features):")
    for feat, (expected_sign, _, note) in plausibility_checks.items():
        lr_row = lr_df[lr_df['feature_name'] == feat]
        if len(lr_row) > 0:
            actual_sign = '+' if lr_row.iloc[0]['coefficient'] > 0 else '-'
            match = actual_sign == expected_sign
            status = "✓ PLAUSIBLE" if match else "⚠ INVESTIGATE"
            print(f"    {feat:30s}  expected={expected_sign}  actual={actual_sign}  {status}  ({note})")
            audit_results.append({
                'feature': feat, 'model': 'LR', 'expected_sign': expected_sign,
                'actual_sign': actual_sign, 'plausible': match, 'note': note
            })
    
    # Check sub-grade monotonicity in LR
    sg_coefs = []
    for _, row in lr_df.iterrows():
        if row['feature_name'].startswith('sub_grade_'):
            sg_coefs.append((row['feature_name'], row['coefficient']))
    
    # Sort by sub_grade order
    grade_order = {f"sub_grade_{g}{n}": i for i, (g, n) in enumerate(
        [(g, n) for g in 'ABCDEFG' for n in '12345']
    )}
    sg_coefs.sort(key=lambda x: grade_order.get(x[0], 999))
    
    # Check broadly increasing trend (A→G)
    if sg_coefs:
        first_coef = sg_coefs[0][1]
        last_coef = sg_coefs[-1][1]
        monotonic = last_coef > first_coef
        print(f"\n    Sub-grade monotonicity: A2 coef={first_coef:.4f}, G5 coef={last_coef:.4f}  "
              f"{'✓ PLAUSIBLE (increasing)' if monotonic else '⚠ NON-MONOTONIC'}")
    
    # Investigation: emp_length_missing
    print("\n\n  === SUSPICIOUS FEATURE INVESTIGATION: emp_length_missing ===")
    print("  Loading training data (READ-ONLY) to examine default rates by emp_length missingness...")
    
    train_xgb = pd.read_parquet(DATA_DIR / 'train_xgb.parquet')
    
    if 'default_flag' in train_xgb.columns and 'emp_length_missing' in train_xgb.columns:
        grouped = train_xgb.groupby('emp_length_missing')['default_flag'].agg(['mean', 'count'])
        grouped.columns = ['default_rate', 'count']
        print(f"\n  Train set default rates by emp_length_missing:")
        for val, row in grouped.iterrows():
            pct = row['count'] / len(train_xgb) * 100
            print(f"    emp_length_missing={val}: default_rate={row['default_rate']:.4f} "
                  f"({row['count']:,} rows, {pct:.1f}%)")
        
        rate_diff = abs(grouped.loc[1, 'default_rate'] - grouped.loc[0, 'default_rate']) if len(grouped) == 2 else 0
        print(f"\n  Default rate difference: {rate_diff:.4f}")
        if rate_diff > 0.02:
            print(f"  FINDING: emp_length_missing has a meaningful default rate difference ({rate_diff:.1%}).")
            print(f"  This could be a legitimate signal (self-employed/unemployed) or a data collection artifact.")
        else:
            print(f"  FINDING: emp_length_missing has a small default rate difference ({rate_diff:.1%}).")
            print(f"  Its high XGB importance may come from interactions with other features.")
    
    # Check addr_state_CO (unusually high for a single state)
    print("\n\n  === SUSPICIOUS FEATURE INVESTIGATION: addr_state_CO ===")
    xgb_co = xgb_df[xgb_df['feature_name'] == 'addr_state_CO']
    if len(xgb_co) > 0:
        co_rank = xgb_co.iloc[0]['gain_rank']
        co_gain = xgb_co.iloc[0]['gain_importance']
        print(f"  XGB gain rank: {co_rank}, gain: {co_gain:.2f}")
        
        # Check permutation importance
        co_perm = xgb_perm_df[xgb_perm_df['feature_name'] == 'addr_state_CO']
        if len(co_perm) > 0:
            co_perm_rank = co_perm.iloc[0]['perm_rank']
            co_perm_val = co_perm.iloc[0]['perm_importance_mean']
            print(f"  Permutation importance rank: {co_perm_rank}, value: {co_perm_val:.6f}")
            if co_perm_rank > 50:
                print(f"  FINDING: addr_state_CO has high GAIN importance but LOW permutation importance.")
                print(f"  This suggests the model is using CO for splits but the feature is not")
                print(f"  genuinely important for prediction. Likely an artifact of tree construction.")
            else:
                print(f"  FINDING: addr_state_CO has high importance by both metrics.")
                print(f"  May reflect genuine geographic credit risk variation in Colorado.")
    
    return audit_results


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 70)
    print("P0.8 — MODEL EXPLAINABILITY & INTERPRETATION")
    print("=" * 70)
    print(f"Random seed: {RANDOM_STATE}")
    print(f"Permutation repeats: {PERM_N_REPEATS}")
    print(f"Output directory: {RESULTS_DIR}")
    print()
    
    # Step 1: Validate environment
    lr_model, xgb_model, X_val_lr, X_val_xgb, y_val, lr_features, xgb_features = validate_environment()
    
    # Step 2: LR coefficients
    lr_df = extract_lr_coefficients(lr_model, lr_features)
    
    # Step 3: LR visualizations
    create_lr_visualizations(lr_df)
    
    # Step 4: XGB importance
    xgb_df = extract_xgb_importance(xgb_model, xgb_features)
    
    # Step 5: XGB visualizations
    create_xgb_visualizations(xgb_df)
    
    # Free memory before expensive permutation importance
    gc.collect()
    print("\n  [Memory cleanup before permutation importance]", flush=True)
    
    # Step 6: Permutation importance
    lr_perm_df, xgb_perm_df = compute_permutation_importance(
        lr_model, xgb_model, X_val_lr, X_val_xgb, y_val, lr_features, xgb_features
    )
    
    # Step 7: Cross-model comparison
    comparison_df = cross_model_comparison(lr_df, xgb_df, lr_perm_df, xgb_perm_df, 
                                            lr_features, xgb_features)
    
    # Step 8: Plausibility audit
    audit_results = plausibility_audit(lr_df, xgb_df, xgb_perm_df)
    
    # Final summary
    print("\n" + "=" * 70)
    print("P0.8 EXECUTION COMPLETE")
    print("=" * 70)
    print(f"\n  Output files in {RESULTS_DIR}:")
    for f in sorted(RESULTS_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:45s}  {size_kb:.1f} KB")
    
    print(f"\n  VALIDATION SUMMARY:")
    print(f"    [PASS] Frozen LR model used (168 features, coef shape (1, 168))")
    print(f"    [PASS] Frozen XGB model used (135 features)")
    print(f"    [PASS] No preprocessing refit (FeatureEngineer not instantiated)")
    print(f"    [PASS] No model retrain (no .fit() calls)")
    print(f"    [PASS] No prediction alteration")
    print(f"    [PASS] No threshold changes")
    print(f"    [PASS] Test set not loaded for explanations")
    print(f"    [PASS] No new features introduced")
    print(f"    [PASS] Val metrics match P0.7 exactly")
    print(f"    [PASS] Reproducibility: random_state={RANDOM_STATE}")
    print(f"    [PASS] No git operations performed")
    

if __name__ == '__main__':
    import traceback
    try:
        main()
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
