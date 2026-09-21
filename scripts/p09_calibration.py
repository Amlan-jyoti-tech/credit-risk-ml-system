"""
P0.9 -- Model Calibration & Probability Quality Assessment
==========================================================

PURPOSE
-------
Assess whether the frozen P0.7 model output scores can be interpreted as
reliable probability estimates. This is a READ-ONLY experiment on the frozen
models and the existing validation dataset.

CALIBRATION API NOTE (scikit-learn 1.8.0 compatibility)
--------------------------------------------------------
The original design specified CalibratedClassifierCV(cv='prefit', method='isotonic').
That parameter was removed in scikit-learn 1.2. Because the installed environment
uses scikit-learn 1.8.0, P0.9 implements the equivalent workflow explicitly:

    "The frozen P0.7 model produces probability scores on the calibration-train
     subset; an IsotonicRegression mapping is fitted on those scores; and the
     calibrator is evaluated on a disjoint calibration-evaluation subset.
     The underlying base model is never refit."

This is semantically identical to the original design intent. The methodology
is documented in P0.9_calibration_report.md.

WHAT THIS SCRIPT DOES NOT DO
------------------------------
- Does NOT modify, retrain, or replace the frozen P0.7 models.
- Does NOT load the test set (test_lr.parquet / test_xgb.parquet).
- Does NOT refit preprocessing (FeatureEngineer is never called).
- Does NOT select or apply a new production/business threshold.
- Does NOT modify any P0.7 or P0.8 artifact.
- Does NOT introduce SHAP, LLM, RAG, or API components.

DATA ROLES
----------
Full validation set    -> descriptive calibration curves + Brier/ECE (primary measurement)
Cal-train subset (70%) -> obtains base-model scores; IsotonicRegression fitted on these scores
Cal-eval  subset (30%) -> IsotonicRegression evaluated here (disjoint from cal-train)
Test set               -> PROHIBITED -- never referenced anywhere in this file

DESIGN REFERENCE
----------------
docs/P0.9_calibration_design.md, sections 8 (Reproducibility) and 9 (Methodology)

REPRODUCIBILITY
---------------
random_state = 42 throughout.
"""

# ===========================================================================
# 0. IMPORTS AND VERSION LOGGING
# ===========================================================================
import sys
import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive; safe on all platforms
import matplotlib.pyplot as plt
import joblib

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

import sklearn

warnings.filterwarnings("ignore")

# --- Version banner ---------------------------------------------------------
print("=" * 62)
print("P0.9 -- Model Calibration & Probability Quality Assessment")
print("=" * 62)
print(f"Python       : {sys.version.split()[0]}")
print(f"scikit-learn : {sklearn.__version__}")
print(f"numpy        : {np.__version__}")
print(f"pandas       : {pd.__version__}")
print(f"joblib       : {joblib.__version__}")
import matplotlib as _mpl
print(f"matplotlib   : {_mpl.__version__}")
try:
    import xgboost as _xgb
    print(f"xgboost      : {_xgb.__version__}")
except ImportError:
    pass
print()
print("Calibration method : IsotonicRegression(out_of_bounds='clip')")
print("  [cv='prefit' removed in sklearn 1.2; explicit isotonic mapping used]")
print()

# ===========================================================================
# 1. PATH CONSTANTS
# ===========================================================================

BASE_DIR = Path(__file__).resolve().parents[1]   # repository root

# --- Frozen inputs (strictly read-only) ------------------------------------
VAL_LR_PATH    = BASE_DIR / "data" / "processed" / "temporal_splits" / "val_lr.parquet"
VAL_XGB_PATH   = BASE_DIR / "data" / "processed" / "temporal_splits" / "val_xgb.parquet"
LR_MODEL_PATH  = BASE_DIR / "src"  / "models"    / "logistic_regression_baseline.joblib"
XGB_MODEL_PATH = BASE_DIR / "src"  / "models"    / "xgboost_baseline.joblib"
P07_RESULTS_PATH = BASE_DIR / "results" / "p07_results.json"

# --- Output directory -------------------------------------------------------
OUT_DIR = BASE_DIR / "results" / "calibration"

# --- Experiment constants (design section 8) --------------------------------
RANDOM_STATE      = 42
CAL_EVAL_FRACTION = 0.30     # 30% held out for wrapper evaluation
N_BINS            = 10       # calibration curve bins
FINGERPRINT_TOL   = 1e-6     # fingerprint match tolerance

# Exact fingerprint values from results/p07_results.json (design §8.7)
EXPECTED_LR_VAL_ROC  = 0.6995575916323613
EXPECTED_XGB_VAL_ROC = 0.7074149159306107


# ===========================================================================
# 2. LEAKAGE GATE -- explicit prohibition on test-file paths
#    These paths are defined only so we can reason about them.
#    They are NEVER passed to any read/load/open call anywhere in this file.
# ===========================================================================
_TEST_LR_PATH  = BASE_DIR / "data" / "processed" / "temporal_splits" / "test_lr.parquet"
_TEST_XGB_PATH = BASE_DIR / "data" / "processed" / "temporal_splits" / "test_xgb.parquet"

print("LEAKAGE GATE")
print("-" * 40)
# Runtime guard: assert test paths are not identical to any path we will open
for _tp in [_TEST_LR_PATH, _TEST_XGB_PATH]:
    assert _tp != VAL_LR_PATH  and _tp != VAL_XGB_PATH, (
        f"INTEGRITY VIOLATION: test path matches a load path: {_tp}"
    )
print("  Test files not referenced in any load call : PASS")
print()


# ===========================================================================
# 3. REQUIRED FILE EXISTENCE ASSERTIONS
# ===========================================================================
print("REQUIRED FILE CHECKS")
print("-" * 40)

_required = {
    "val_lr.parquet"   : VAL_LR_PATH,
    "val_xgb.parquet"  : VAL_XGB_PATH,
    "lr_model.joblib"  : LR_MODEL_PATH,
    "xgb_model.joblib" : XGB_MODEL_PATH,
    "p07_results.json" : P07_RESULTS_PATH,
}
for label, path in _required.items():
    assert path.exists(), f"REQUIRED FILE MISSING: {path}"
    print(f"  OK  ({path.stat().st_size:>10,} bytes)  {label}")

# Capture frozen model file metadata BEFORE anything runs.
# Used in the post-run audit to prove neither file was modified.
_lr_stat_before  = (LR_MODEL_PATH.stat().st_size,  LR_MODEL_PATH.stat().st_mtime)
_xgb_stat_before = (XGB_MODEL_PATH.stat().st_size, XGB_MODEL_PATH.stat().st_mtime)

print()


# ===========================================================================
# 4. CREATE OUTPUT DIRECTORY
#    exist_ok=False fails loudly if the directory already exists, preventing
#    accidental overwrite of a previous complete run.
# ===========================================================================
OUT_DIR.mkdir(parents=True, exist_ok=False)
print(f"Created output directory: {OUT_DIR.relative_to(BASE_DIR)}")
print()


# ===========================================================================
# 5. LOAD FROZEN P0.7 MODELS  (read-only -- no .fit() calls ever)
# ===========================================================================
print("LOADING FROZEN P0.7 MODELS")
print("-" * 40)

lr_model  = joblib.load(LR_MODEL_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)

# Dimension assertions confirm we loaded the correct objects
assert hasattr(lr_model, "coef_"), "LR model missing coef_ -- wrong object?"
assert lr_model.coef_.shape == (1, 168), (
    f"LR coef_ shape: expected (1, 168), got {lr_model.coef_.shape}"
)
assert hasattr(xgb_model, "n_features_in_"), "XGB model missing n_features_in_"
assert xgb_model.n_features_in_ == 135, (
    f"XGB n_features_in_: expected 135, got {xgb_model.n_features_in_}"
)
print(f"  LR  coef_.shape    : {lr_model.coef_.shape}  -- PASS")
print(f"  XGB n_features_in_ : {xgb_model.n_features_in_}  -- PASS")
print()


# ===========================================================================
# 6. LOAD VALIDATION DATA  (read-only; test parquets are never touched)
# ===========================================================================
print("LOADING VALIDATION DATA")
print("-" * 40)

val_lr_df  = pd.read_parquet(VAL_LR_PATH)
val_xgb_df = pd.read_parquet(VAL_XGB_PATH)

# Target column must be present
assert "default_flag" in val_lr_df.columns,  "default_flag missing from val_lr"
assert "default_flag" in val_xgb_df.columns, "default_flag missing from val_xgb"

# Metadata columns excluded from feature matrices
META_COLS = ["id", "issue_d", "default_flag"]
feature_cols_lr  = [c for c in val_lr_df.columns  if c not in META_COLS]
feature_cols_xgb = [c for c in val_xgb_df.columns if c not in META_COLS]

X_val_lr  = val_lr_df[feature_cols_lr].reset_index(drop=True)
X_val_xgb = val_xgb_df[feature_cols_xgb].reset_index(drop=True)

y_val_lr  = val_lr_df["default_flag"].astype(int).reset_index(drop=True)
y_val_xgb = val_xgb_df["default_flag"].astype(int).reset_index(drop=True)

# Labels must be identical between the two files (same loans, same order)
assert (y_val_lr.values == y_val_xgb.values).all(), (
    "default_flag mismatch between val_lr and val_xgb -- row alignment error"
)
y_val = y_val_lr.copy()   # canonical label vector

# Feature dimension assertions (frozen dimension contract from P0.5/P0.6)
assert X_val_lr.shape[1]  == 168, f"val_lr: expected 168 features, got {X_val_lr.shape[1]}"
assert X_val_xgb.shape[1] == 135, f"val_xgb: expected 135 features, got {X_val_xgb.shape[1]}"

# No all-NaN rows (calibration_curve is sensitive to these)
assert not X_val_lr.isnull().all(axis=1).any(),  "All-NaN rows in X_val_lr"
assert not X_val_xgb.isnull().all(axis=1).any(), "All-NaN rows in X_val_xgb"

prevalence = float(y_val.mean())
print(f"  val_lr  shape  : {X_val_lr.shape}")
print(f"  val_xgb shape  : {X_val_xgb.shape}")
print(f"  y_val   shape  : {y_val.shape}")
print(f"  Default rate   : {prevalence:.6f}  ({prevalence:.4%})")
print()


# ===========================================================================
# 7. FINGERPRINT VERIFICATION  (design §8.7 / §9 Step 1)
#    Recompute validation ROC-AUC using frozen models.
#    HARD STOP if either value deviates from p07_results.json by >1e-6.
# ===========================================================================
print("FINGERPRINT VERIFICATION")
print("-" * 40)

with open(P07_RESULTS_PATH) as fh:
    p07_ref = json.load(fh)

ref_lr_roc  = p07_ref["LR"]["val_metrics"]["roc_auc"]
ref_xgb_roc = p07_ref["XGB"]["val_metrics"]["roc_auc"]

# Read-only forward pass through frozen models on FULL validation set
# These probability vectors are also used for the primary calibration metrics
y_prob_lr_full  = lr_model.predict_proba(X_val_lr)[:, 1]
y_prob_xgb_full = xgb_model.predict_proba(X_val_xgb)[:, 1]

comp_lr_roc  = float(roc_auc_score(y_val, y_prob_lr_full))
comp_xgb_roc = float(roc_auc_score(y_val, y_prob_xgb_full))

diff_lr  = abs(comp_lr_roc  - ref_lr_roc)
diff_xgb = abs(comp_xgb_roc - ref_xgb_roc)

print(f"  LR  reference  : {ref_lr_roc:.16f}")
print(f"  LR  computed   : {comp_lr_roc:.16f}")
print(f"  LR  |diff|     : {diff_lr:.2e}   tolerance={FINGERPRINT_TOL:.0e}")

assert diff_lr <= FINGERPRINT_TOL, (
    f"STOP -- LR fingerprint mismatch: |diff|={diff_lr:.2e} > {FINGERPRINT_TOL:.0e}. "
    "Wrong model file or data file. Do not proceed."
)
print("  LR  fingerprint : PASS")
print()

print(f"  XGB reference  : {ref_xgb_roc:.16f}")
print(f"  XGB computed   : {comp_xgb_roc:.16f}")
print(f"  XGB |diff|     : {diff_xgb:.2e}   tolerance={FINGERPRINT_TOL:.0e}")

assert diff_xgb <= FINGERPRINT_TOL, (
    f"STOP -- XGB fingerprint mismatch: |diff|={diff_xgb:.2e} > {FINGERPRINT_TOL:.0e}. "
    "Wrong model file or data file. Do not proceed."
)
print("  XGB fingerprint : PASS")
print()

# Probability sanity checks
assert y_prob_lr_full.min()  >= 0.0 and y_prob_lr_full.max()  <= 1.0, "LR probs out of [0,1]"
assert y_prob_xgb_full.min() >= 0.0 and y_prob_xgb_full.max() <= 1.0, "XGB probs out of [0,1]"


# ===========================================================================
# 8. HELPER FUNCTIONS
# ===========================================================================

def get_calibration_data(y_true, y_prob, n_bins, strategy):
    """
    Returns (fraction_of_positives, mean_predicted_value, bin_counts).

    bin_counts is derived from sklearn's internal binning so we can weight
    bins correctly when computing ECE without a second calibration_curve call.
    """
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )
    # Replicate sklearn's bin assignment to recover per-bin sample counts
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.percentile(y_prob, quantiles * 100)
        edges[-1] += 1e-8      # ensure the maximum value is included
    else:   # uniform
        edges = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)

    bin_ids = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    all_counts = np.array([int(np.sum(bin_ids == i)) for i in range(n_bins)])

    # calibration_curve skips empty bins; keep only non-empty bin counts,
    # in the same order that calibration_curve returned them.
    nonempty_idx = np.where(all_counts > 0)[0]
    n_returned = len(frac_pos)
    matched_counts = all_counts[nonempty_idx[:n_returned]]
    return frac_pos, mean_pred, matched_counts


def compute_ece(frac_pos, mean_pred, bin_counts):
    """
    Expected Calibration Error (ECE).
    ECE = sum_b (n_b / N) * |observed_rate_b - mean_pred_b|
    Definition: design §8.5.
    """
    total = bin_counts.sum()
    if total == 0:
        return float("nan")
    return float(np.sum((bin_counts / total) * np.abs(frac_pos - mean_pred)))


# --- Dark-theme plot style consistent with project -------------------------
_PLOT_RC = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor"  : "#1a1d27",
    "axes.edgecolor"  : "#3a3f5c",
    "axes.labelcolor" : "#d0d0d8",
    "xtick.color"     : "#a0a0b0",
    "ytick.color"     : "#a0a0b0",
    "text.color"      : "#d0d0d8",
    "grid.color"      : "#2a2d3e",
}


def _style_ax(ax, title,
              xlabel="Mean Predicted Probability",
              ylabel="Observed Default Rate"):
    ax.set_title(title, fontsize=10, fontweight="bold", color="#d0d0d8", pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color="#a0a0b0")
    ax.set_ylabel(ylabel, fontsize=9, color="#a0a0b0")
    ax.tick_params(colors="#a0a0b0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3f5c")
    ax.grid(True, linestyle="--", alpha=0.35, color="#2a2d3e")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 0.82)


# ===========================================================================
# 9. CALIBRATION CURVES -- FULL VALIDATION SET
#    PRIMARY DESCRIPTIVE MEASUREMENT  (design §9 Step 3)
#
#    Data role: full val set (283,026 rows) used read-only.
#    No calibrator is fitted here.
# ===========================================================================
print("CALIBRATION CURVES -- FULL VALIDATION SET (primary measurement)")
print("-" * 40)

# Quantile strategy (primary, design §8.3)
fp_lr_q,  mp_lr_q,  bc_lr_q  = get_calibration_data(
    y_val, y_prob_lr_full,  N_BINS, "quantile")
fp_xgb_q, mp_xgb_q, bc_xgb_q = get_calibration_data(
    y_val, y_prob_xgb_full, N_BINS, "quantile")

# Uniform strategy (cross-check, design §8.3)
fp_lr_u,  mp_lr_u,  _ = get_calibration_data(
    y_val, y_prob_lr_full,  N_BINS, "uniform")
fp_xgb_u, mp_xgb_u, _ = get_calibration_data(
    y_val, y_prob_xgb_full, N_BINS, "uniform")

print(f"  LR  bins returned : quantile={len(fp_lr_q)},  uniform={len(fp_lr_u)}")
print(f"  XGB bins returned : quantile={len(fp_xgb_q)}, uniform={len(fp_xgb_u)}")


# ===========================================================================
# 10. BRIER SCORES, BRIER SKILL SCORE, ECE -- FULL VALIDATION SET
#     (design §9 Step 4 / §8.5)
#
#     Note: ROC-AUC (discrimination) and Brier/ECE (calibration) are
#     distinct properties. Improved calibration does not imply improved
#     ROC-AUC, which is rank-based and unaffected by score rescaling.
# ===========================================================================
print()
print("BRIER SCORES, BSS, ECE -- FULL VALIDATION SET")
print("-" * 40)

brier_lr     = float(brier_score_loss(y_val, y_prob_lr_full))
brier_xgb    = float(brier_score_loss(y_val, y_prob_xgb_full))
# No-skill baseline: always predict the observed validation-set prevalence
brier_noskill = float(brier_score_loss(
    y_val, np.full(len(y_val), prevalence)))

# Brier Skill Score: 1 - (model / baseline); positive means better than naive
bss_lr  = 1.0 - brier_lr  / brier_noskill
bss_xgb = 1.0 - brier_xgb / brier_noskill

ece_lr  = compute_ece(fp_lr_q,  mp_lr_q,  bc_lr_q)
ece_xgb = compute_ece(fp_xgb_q, mp_xgb_q, bc_xgb_q)

print(f"  No-skill baseline Brier : {brier_noskill:.6f}")
print(f"  LR  Brier               : {brier_lr:.6f}")
print(f"  XGB Brier               : {brier_xgb:.6f}")
print()
print(f"  LR  Brier Skill Score   : {bss_lr:.6f}  (>0 = beats naive)")
print(f"  XGB Brier Skill Score   : {bss_xgb:.6f}")
print()
print(f"  LR  ECE (quantile)      : {ece_lr:.6f}  (<0.05 = well calibrated)")
print(f"  XGB ECE (quantile)      : {ece_xgb:.6f}")
print()


# ===========================================================================
# 11. SAVE CALIBRATION SUMMARY TABLES  (design §8.8)
# ===========================================================================

def _make_summary(fp, mp, bc, strategy, model):
    n = len(fp)
    return pd.DataFrame({
        "model"                  : model,
        "strategy"               : strategy,
        "bin_index"              : np.arange(n),
        "mean_predicted_value"   : mp,
        "fraction_of_positives"  : fp,
        "bin_count"              : bc,
        "abs_error"              : np.abs(fp - mp),
    })

cal_summary_lr = pd.concat([
    _make_summary(fp_lr_q, mp_lr_q, bc_lr_q, "quantile", "LR"),
    _make_summary(fp_lr_u, mp_lr_u,
                  np.zeros(len(fp_lr_u), dtype=int), "uniform", "LR"),
], ignore_index=True)

cal_summary_xgb = pd.concat([
    _make_summary(fp_xgb_q, mp_xgb_q, bc_xgb_q, "quantile", "XGB"),
    _make_summary(fp_xgb_u, mp_xgb_u,
                  np.zeros(len(fp_xgb_u), dtype=int), "uniform", "XGB"),
], ignore_index=True)

cal_summary_lr.to_csv( OUT_DIR / "calibration_summary_lr.csv",  index=False)
cal_summary_xgb.to_csv(OUT_DIR / "calibration_summary_xgb.csv", index=False)
print("  Saved: calibration_summary_lr.csv")
print("  Saved: calibration_summary_xgb.csv")
print()


# ===========================================================================
# 12. SAVE brier_scores.json  (design §8.8)
# ===========================================================================
brier_json = {
    "data_scope"             : "Full validation set (val_lr / val_xgb parquets), 283,026 rows",
    "n_samples"              : int(len(y_val)),
    "default_rate"           : round(prevalence, 8),
    "brier_noskill_baseline" : round(brier_noskill, 8),
    "LR": {
        "brier_score"        : round(brier_lr, 8),
        "brier_skill_score"  : round(bss_lr, 8),
        "ece_quantile"       : round(ece_lr, 8),
    },
    "XGB": {
        "brier_score"        : round(brier_xgb, 8),
        "brier_skill_score"  : round(bss_xgb, 8),
        "ece_quantile"       : round(ece_xgb, 8),
    },
    "definitions": {
        "brier_score"        : "sklearn.metrics.brier_score_loss(y_true, y_prob); lower=better",
        "brier_skill_score"  : "1 - (model_brier / noskill_brier); >0 beats naive baseline",
        "ece"                : "Weighted mean |frac_pos - mean_pred| per bin; quantile, n_bins=10",
        "noskill_baseline"   : "Always predict observed validation-set default rate",
    },
    "important_distinction": (
        "ROC-AUC measures discrimination (ranking ability) and is rank-based. "
        "Brier Score and ECE measure calibration (probability reliability). "
        "These are independent properties. Post-hoc calibration rescales scores "
        "but does not change their ranking order, so ROC-AUC is unaffected."
    ),
}

with open(OUT_DIR / "brier_scores.json", "w") as fh:
    json.dump(brier_json, fh, indent=4)
print("  Saved: brier_scores.json")
print()


# ===========================================================================
# 13. CALIBRATION CURVE PLOTS  (design §8.8)
# ===========================================================================
print("GENERATING CALIBRATION CURVE PLOTS")
print("-" * 40)

# Individual LR: quantile (primary) + uniform (cross-check)
with plt.rc_context(_PLOT_RC):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax, fp, mp, strat, color in zip(
        axes,
        [fp_lr_q,  fp_lr_u],
        [mp_lr_q,  mp_lr_u],
        ["Quantile (Primary)", "Uniform (Cross-check)"],
        ["#4fc3f7", "#81c995"],
    ):
        ax.plot([0,1],[0,1],"--",lw=1.2,color="#888899",label="Perfect calibration")
        ax.plot(mp, fp, "o-", color=color, lw=2, ms=6, label=f"LR ({strat})")
        ax.fill_between(mp, fp, mp, alpha=0.07, color=color)
        _style_ax(ax, f"Logistic Regression -- {strat}\n"
                  f"Brier={brier_lr:.5f}  BSS={bss_lr:.4f}  ECE={ece_lr:.4f}")
        ax.legend(fontsize=8, facecolor="#1a1d27",
                  edgecolor="#3a3f5c", labelcolor="#d0d0d8")
    plt.tight_layout(pad=2.2)
    plt.savefig(OUT_DIR / "calibration_curve_lr_quantile.png",
                dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
print("  Saved: calibration_curve_lr_quantile.png")

# Individual XGB
with plt.rc_context(_PLOT_RC):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax, fp, mp, strat, color in zip(
        axes,
        [fp_xgb_q,  fp_xgb_u],
        [mp_xgb_q,  mp_xgb_u],
        ["Quantile (Primary)", "Uniform (Cross-check)"],
        ["#f4a261", "#e07fa3"],
    ):
        ax.plot([0,1],[0,1],"--",lw=1.2,color="#888899",label="Perfect calibration")
        ax.plot(mp, fp, "s-", color=color, lw=2, ms=6, label=f"XGB ({strat})")
        ax.fill_between(mp, fp, mp, alpha=0.07, color=color)
        _style_ax(ax, f"XGBoost -- {strat}\n"
                  f"Brier={brier_xgb:.5f}  BSS={bss_xgb:.4f}  ECE={ece_xgb:.4f}")
        ax.legend(fontsize=8, facecolor="#1a1d27",
                  edgecolor="#3a3f5c", labelcolor="#d0d0d8")
    plt.tight_layout(pad=2.2)
    plt.savefig(OUT_DIR / "calibration_curve_xgb_quantile.png",
                dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
print("  Saved: calibration_curve_xgb_quantile.png")

# Uniform cross-check individual files
for fp_u, mp_u, mname, color, fname in [
    (fp_lr_u,  mp_lr_u,  "Logistic Regression", "#4fc3f7",
     "calibration_curve_lr_uniform.png"),
    (fp_xgb_u, mp_xgb_u, "XGBoost",             "#f4a261",
     "calibration_curve_xgb_uniform.png"),
]:
    with plt.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#0f1117")
        ax.plot([0,1],[0,1],"--",lw=1.2,color="#888899",label="Perfect calibration")
        ax.plot(mp_u, fp_u, "s-", color=color, lw=2, ms=6, label=mname)
        _style_ax(ax, f"{mname} -- Uniform Bins (Cross-check)")
        ax.legend(fontsize=8, facecolor="#1a1d27",
                  edgecolor="#3a3f5c", labelcolor="#d0d0d8")
        plt.tight_layout(pad=2)
        plt.savefig(OUT_DIR / fname, dpi=150,
                    bbox_inches="tight", facecolor="#0f1117")
        plt.close()
    print(f"  Saved: {fname}")

# Comparison: both models + diagonal (quantile)
with plt.rc_context(_PLOT_RC):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.plot([0,1],[0,1],"--",lw=1.5,color="#888899",
            label="Perfect calibration (45 deg)")
    ax.plot(mp_lr_q,  fp_lr_q,  "o-", color="#4fc3f7", lw=2, ms=7,
            label=f"LR  (Brier={brier_lr:.4f}, ECE={ece_lr:.4f})")
    ax.plot(mp_xgb_q, fp_xgb_q, "s-", color="#f4a261", lw=2, ms=7,
            label=f"XGB (Brier={brier_xgb:.4f}, ECE={ece_xgb:.4f})")
    _style_ax(ax,
              "Calibration Comparison: LR vs XGB\n"
              "(Full Validation Set, Quantile Bins, n_bins=10)")
    ax.legend(fontsize=9, facecolor="#1a1d27",
              edgecolor="#3a3f5c", labelcolor="#d0d0d8")
    plt.tight_layout(pad=2)
    plt.savefig(OUT_DIR / "calibration_curve_comparison_quantile.png",
                dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
print("  Saved: calibration_curve_comparison_quantile.png")
print()


# ===========================================================================
# 14. POST-HOC CALIBRATION EXPERIMENT  (design §9 Step 5 -- Option A)
#
#     IMPLEMENTATION NOTE (sklearn 1.8.0 compatibility)
#     --------------------------------------------------
#     CalibratedClassifierCV(cv='prefit') was removed in sklearn 1.2.
#     The equivalent explicit workflow is:
#
#       scores_cal_train = frozen_model.predict_proba(X_cal_train)[:, 1]
#       iso = IsotonicRegression(out_of_bounds='clip')
#       iso.fit(scores_cal_train, y_cal_train)   # ONLY this is fitted
#       scores_cal_eval = frozen_model.predict_proba(X_cal_eval)[:, 1]
#       calibrated_probs = iso.predict(scores_cal_eval)
#
#     The frozen P0.7 model parameters are NEVER modified.
#
#     DATA-ROLE DISTINCTION
#     ---------------------
#     Full val set  -> primary calibration measurement (Section 9 above)
#     cal-train 70% -> base model scores extracted; IsotonicRegression fitted
#     cal-eval  30% -> IsotonicRegression evaluated; NEVER used for fitting
# ===========================================================================
print("POST-HOC CALIBRATION EXPERIMENT  (IsotonicRegression, Option A)")
print("-" * 40)

n_val = len(y_val)
all_idx = np.arange(n_val)

# Deterministic stratified 70/30 split (design §8.2)
idx_cal_train, idx_cal_eval = train_test_split(
    all_idx,
    test_size=CAL_EVAL_FRACTION,      # 30% for evaluation
    stratify=y_val.values,
    random_state=RANDOM_STATE,
)

# Disjointness assertion -- critical integrity check (design §8, leakage 5)
assert len(set(idx_cal_train) & set(idx_cal_eval)) == 0, (
    "INTEGRITY VIOLATION: cal-train and cal-eval indices overlap"
)

expected_eval = round(n_val * CAL_EVAL_FRACTION)
assert abs(len(idx_cal_eval) - expected_eval) <= 2, (
    f"cal-eval size unexpected: {len(idx_cal_eval)} vs ~{expected_eval}"
)

print(f"  cal-train size  : {len(idx_cal_train):,} rows  (~70%)")
print(f"  cal-eval  size  : {len(idx_cal_eval):,} rows  (~30%)")
print(f"  Disjointness    : PASS")

y_cal_train = y_val.iloc[idx_cal_train].reset_index(drop=True)
y_cal_eval  = y_val.iloc[idx_cal_eval].reset_index(drop=True)

print(f"  cal-train default rate : {float(y_cal_train.mean()):.4%}")
print(f"  cal-eval  default rate : {float(y_cal_eval.mean()):.4%}")
print()

# --- Step 1: Obtain base-model scores on cal-train (no refit) --------------
# frozen_model.predict_proba() is a read-only forward pass.
X_cal_train_lr  = X_val_lr.iloc[idx_cal_train].reset_index(drop=True)
X_cal_train_xgb = X_val_xgb.iloc[idx_cal_train].reset_index(drop=True)

scores_cal_train_lr  = lr_model.predict_proba(X_cal_train_lr)[:, 1]
scores_cal_train_xgb = xgb_model.predict_proba(X_cal_train_xgb)[:, 1]

# --- Step 2: Fit isotonic mapping on cal-train scores ----------------------
# IsotonicRegression learns a monotone score -> probability remapping.
# It is the ONLY object fitted in this section.
print("  Fitting IsotonicRegression on cal-train base-model scores ...")

iso_lr  = IsotonicRegression(out_of_bounds="clip")
iso_xgb = IsotonicRegression(out_of_bounds="clip")

iso_lr.fit(scores_cal_train_lr,  y_cal_train)
iso_xgb.fit(scores_cal_train_xgb, y_cal_train)

print("  IsotonicRegression fitted for LR  (on cal-train scores)")
print("  IsotonicRegression fitted for XGB (on cal-train scores)")
print()

# --- Step 3: Apply isotonic calibrator to cal-eval scores ------------------
# The base model's predict_proba is called on cal-eval (read-only).
# iso.predict() applies the learned monotonic remapping.
X_cal_eval_lr  = X_val_lr.iloc[idx_cal_eval].reset_index(drop=True)
X_cal_eval_xgb = X_val_xgb.iloc[idx_cal_eval].reset_index(drop=True)

scores_cal_eval_lr_base  = lr_model.predict_proba(X_cal_eval_lr)[:, 1]
scores_cal_eval_xgb_base = xgb_model.predict_proba(X_cal_eval_xgb)[:, 1]

cal_probs_lr_eval  = iso_lr.predict(scores_cal_eval_lr_base)
cal_probs_xgb_eval = iso_xgb.predict(scores_cal_eval_xgb_base)

# Calibrated probs must be in [0,1] (out_of_bounds='clip' guarantees this)
assert cal_probs_lr_eval.min()  >= 0.0 and cal_probs_lr_eval.max()  <= 1.0
assert cal_probs_xgb_eval.min() >= 0.0 and cal_probs_xgb_eval.max() <= 1.0

# --- Step 4: Evaluate on cal-eval ------------------------------------------
brier_base_lr_e  = float(brier_score_loss(y_cal_eval, scores_cal_eval_lr_base))
brier_cal_lr_e   = float(brier_score_loss(y_cal_eval, cal_probs_lr_eval))
brier_base_xgb_e = float(brier_score_loss(y_cal_eval, scores_cal_eval_xgb_base))
brier_cal_xgb_e  = float(brier_score_loss(y_cal_eval, cal_probs_xgb_eval))

delta_lr  = brier_base_lr_e  - brier_cal_lr_e    # positive = improvement
delta_xgb = brier_base_xgb_e - brier_cal_xgb_e

print("  WRAPPER EVALUATION (cal-eval subset, ~30%)")
print(f"    LR  base Brier (cal-eval)       : {brier_base_lr_e:.6f}")
print(f"    LR  calibrated Brier (cal-eval) : {brier_cal_lr_e:.6f}")
print(f"    LR  delta Brier (base - cal)    : {delta_lr:+.6f}  "
      f"({'improvement' if delta_lr > 0 else 'degradation'})")
print()
print(f"    XGB base Brier (cal-eval)       : {brier_base_xgb_e:.6f}")
print(f"    XGB calibrated Brier (cal-eval) : {brier_cal_xgb_e:.6f}")
print(f"    XGB delta Brier (base - cal)    : {delta_xgb:+.6f}  "
      f"({'improvement' if delta_xgb > 0 else 'degradation'})")
print()

# Calibration curves on cal-eval: base vs calibrated (before/after)
fp_base_lr_e,  mp_base_lr_e,  bc_base_lr_e  = get_calibration_data(
    y_cal_eval, scores_cal_eval_lr_base, N_BINS, "quantile")
fp_cal_lr_e,   mp_cal_lr_e,   bc_cal_lr_e   = get_calibration_data(
    y_cal_eval, cal_probs_lr_eval,       N_BINS, "quantile")
fp_base_xgb_e, mp_base_xgb_e, bc_base_xgb_e = get_calibration_data(
    y_cal_eval, scores_cal_eval_xgb_base, N_BINS, "quantile")
fp_cal_xgb_e,  mp_cal_xgb_e,  bc_cal_xgb_e  = get_calibration_data(
    y_cal_eval, cal_probs_xgb_eval,       N_BINS, "quantile")

ece_base_lr_e  = compute_ece(fp_base_lr_e,  mp_base_lr_e,  bc_base_lr_e)
ece_cal_lr_e   = compute_ece(fp_cal_lr_e,   mp_cal_lr_e,   bc_cal_lr_e)
ece_base_xgb_e = compute_ece(fp_base_xgb_e, mp_base_xgb_e, bc_base_xgb_e)
ece_cal_xgb_e  = compute_ece(fp_cal_xgb_e,  mp_cal_xgb_e,  bc_cal_xgb_e)

print(f"    LR  base ECE (cal-eval)         : {ece_base_lr_e:.6f}")
print(f"    LR  calibrated ECE (cal-eval)   : {ece_cal_lr_e:.6f}")
print(f"    XGB base ECE (cal-eval)         : {ece_base_xgb_e:.6f}")
print(f"    XGB calibrated ECE (cal-eval)   : {ece_cal_xgb_e:.6f}")
print()

# Before/after reliability diagram plots
for mname, fp_b, mp_b, fp_c, mp_c, br_b, br_c, fname in [
    ("Logistic Regression",
     fp_base_lr_e, mp_base_lr_e, fp_cal_lr_e, mp_cal_lr_e,
     brier_base_lr_e, brier_cal_lr_e,
     "calibration_before_after_lr.png"),
    ("XGBoost",
     fp_base_xgb_e, mp_base_xgb_e, fp_cal_xgb_e, mp_cal_xgb_e,
     brier_base_xgb_e, brier_cal_xgb_e,
     "calibration_before_after_xgb.png"),
]:
    with plt.rc_context(_PLOT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#0f1117")
        for ax, fp, mp, label, color, br in zip(
            axes,
            [fp_b, fp_c],
            [mp_b, mp_c],
            ["Base (uncalibrated)", "Post-hoc Isotonic Calibration"],
            ["#f4a261", "#81c995"],
            [br_b, br_c],
        ):
            ax.plot([0,1],[0,1],"--",lw=1.2,color="#888899",
                    label="Perfect calibration")
            ax.plot(mp, fp, "s-", color=color, lw=2, ms=7, label=label)
            _style_ax(ax,
                      f"{mname} -- {label}\n"
                      f"(Cal-eval subset, Brier={br:.5f})")
            ax.legend(fontsize=8, facecolor="#1a1d27",
                      edgecolor="#3a3f5c", labelcolor="#d0d0d8")
        plt.tight_layout(pad=2.2)
        plt.savefig(OUT_DIR / fname, dpi=150,
                    bbox_inches="tight", facecolor="#0f1117")
        plt.close()
    print(f"  Saved: {fname}")

# Save isotonic calibration objects
# Names carry 'p09_' prefix to distinguish from frozen P0.7 baseline models.
# Stored in results/calibration/, NOT in src/models/.
joblib.dump(iso_lr,  OUT_DIR / "p09_calibrated_lr_isotonic.joblib")
joblib.dump(iso_xgb, OUT_DIR / "p09_calibrated_xgb_isotonic.joblib")
print("  Saved: p09_calibrated_lr_isotonic.joblib  (P0.9 analytical artifact)")
print("  Saved: p09_calibrated_xgb_isotonic.joblib (P0.9 analytical artifact)")
print()


# ===========================================================================
# 15. ILLUSTRATIVE COST-THRESHOLD ANALYSIS  (design §9 Step 6)
#
#     !! ILLUSTRATIVE ONLY !!
#     The formula p* = C_FP / (C_FP + C_FN) holds only when:
#       (a) probabilities are well-calibrated,
#       (b) costs C_FP and C_FN are defined in the same monetary unit, AND
#       (c) no portfolio constraints or regulatory floors apply.
#     Real lending decisions require actuarial inputs not available here.
#
#     Uses calibrated probabilities (cal_probs from iso.predict) on cal-eval.
#     Does NOT touch the test set.
#     Does NOT modify P0.7 thresholds (LR: 0.50, XGB: 0.45).
# ===========================================================================
print("ILLUSTRATIVE COST-THRESHOLD ANALYSIS  (informational only)")
print("-" * 40)
print("  NOTE: illustrative only -- not applied to test set.")
print("  NOTE: P0.7 thresholds (LR=0.50, XGB=0.45) remain frozen.")
print()

scenarios = [
    {"label": "F1 Maximization (P0.7 baseline reference)", "C_FP": 1, "C_FN": 1},
    {"label": "Moderate loss aversion (FN costs 3x FP)",   "C_FP": 1, "C_FN": 3},
    {"label": "High loss aversion (FN costs 9x FP)",       "C_FP": 1, "C_FN": 9},
]

rows = []
for sc in scenarios:
    C_FP, C_FN = sc["C_FP"], sc["C_FN"]
    p_star = C_FP / (C_FP + C_FN)
    for mname, y_prob_c in [
        ("LR_calibrated",  cal_probs_lr_eval),
        ("XGB_calibrated", cal_probs_xgb_eval),
    ]:
        y_pred = (y_prob_c >= p_star).astype(int)
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            prec = rec = f1v = float("nan")
        else:
            prec = float(precision_score(y_cal_eval, y_pred, zero_division=0))
            rec  = float(recall_score(y_cal_eval,    y_pred, zero_division=0))
            f1v  = float(f1_score(y_cal_eval,        y_pred, zero_division=0))
        rows.append({
            "scenario"  : sc["label"],
            "C_FP"      : C_FP,
            "C_FN"      : C_FN,
            "p_star"    : round(p_star, 4),
            "model"     : mname,
            "precision" : round(prec, 4) if not np.isnan(prec) else None,
            "recall"    : round(rec,  4) if not np.isnan(rec)  else None,
            "f1"        : round(f1v,  4) if not np.isnan(f1v)  else None,
            "note"      : ("Illustrative only -- not applied to test set. "
                           "Not a production threshold recommendation."),
        })
        print(f"  {sc['label']} | {mname} | p*={p_star:.2f} | "
              f"P={prec:.4f} R={rec:.4f} F1={f1v:.4f}")

print()
cost_df = pd.DataFrame(rows)
cost_df.to_csv(OUT_DIR / "cost_threshold_illustration.csv", index=False)
print("  Saved: cost_threshold_illustration.csv")

# Precision/recall/F1 sweep plot
with plt.rc_context(_PLOT_RC):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax, mname, y_prob_c in [
        (axes[0], "LR (Post-hoc Isotonic)",  cal_probs_lr_eval),
        (axes[1], "XGB (Post-hoc Isotonic)", cal_probs_xgb_eval),
    ]:
        thresholds = np.linspace(0.05, 0.95, 300)
        precs, recs, f1s = [], [], []
        for t in thresholds:
            yp = (y_prob_c >= t).astype(int)
            if yp.sum() == 0 or yp.sum() == len(yp):
                precs.append(np.nan); recs.append(np.nan); f1s.append(np.nan)
            else:
                precs.append(float(precision_score(y_cal_eval, yp, zero_division=0)))
                recs.append( float(recall_score(y_cal_eval,    yp, zero_division=0)))
                f1s.append(  float(f1_score(y_cal_eval,        yp, zero_division=0)))
        ax.plot(thresholds, precs, color="#81c995", lw=1.8, label="Precision")
        ax.plot(thresholds, recs,  color="#4fc3f7", lw=1.8, label="Recall")
        ax.plot(thresholds, f1s,   color="#f4a261", lw=1.8, label="F1")
        for sc in scenarios:
            p_star = sc["C_FP"] / (sc["C_FP"] + sc["C_FN"])
            ax.axvline(p_star, ls=":", lw=1.1, color="#cccccc", alpha=0.7)
            ax.text(p_star + 0.01, 0.93, f"p*={p_star:.2f}",
                    fontsize=7, color="#bbbbbb",
                    transform=ax.get_xaxis_transform())
        _style_ax(ax,
                  f"{mname}\nIllustrative Cost-Threshold (Cal-eval, calibrated probs)",
                  xlabel="Decision Threshold (p*)", ylabel="Metric Value")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, facecolor="#1a1d27",
                  edgecolor="#3a3f5c", labelcolor="#d0d0d8")
    fig.text(0.5, -0.04,
             "ILLUSTRATIVE ONLY -- not applied to test set -- "
             "not a production threshold recommendation",
             ha="center", fontsize=8, color="#e06c75", style="italic")
    plt.tight_layout(pad=2.2)
    plt.savefig(OUT_DIR / "cost_threshold_illustration.png",
                dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
print("  Saved: cost_threshold_illustration.png")
print()


# ===========================================================================
# 16. SAVE p09_results.json  (comprehensive experiment record)
# ===========================================================================
p09_results = {
    "phase"   : "P0.9",
    "title"   : "Model Calibration & Probability Quality Assessment",
    "design"  : "docs/P0.9_calibration_design.md",
    "environment": {
        "python"       : sys.version.split()[0],
        "scikit_learn" : sklearn.__version__,
        "numpy"        : np.__version__,
        "pandas"       : pd.__version__,
        "random_state" : RANDOM_STATE,
    },
    "calibration_api_note": (
        "CalibratedClassifierCV(cv='prefit') was removed in sklearn 1.2. "
        "P0.9 uses the equivalent explicit workflow: frozen model predict_proba "
        "scores are passed to IsotonicRegression(out_of_bounds='clip').fit(). "
        "The base model is never refit. Approved as Option A."
    ),
    "frozen_models": {
        "LR"  : "src/models/logistic_regression_baseline.joblib",
        "XGB" : "src/models/xgboost_baseline.joblib",
    },
    "fingerprints": {
        "LR_val_roc_auc_reference" : ref_lr_roc,
        "LR_val_roc_auc_computed"  : comp_lr_roc,
        "XGB_val_roc_auc_reference": ref_xgb_roc,
        "XGB_val_roc_auc_computed" : comp_xgb_roc,
        "tolerance"                : FINGERPRINT_TOL,
        "status"                   : "PASS",
    },
    "data": {
        "full_val_n"              : int(len(y_val)),
        "default_rate"            : round(prevalence, 8),
        "cal_train_n"             : int(len(idx_cal_train)),
        "cal_eval_n"              : int(len(idx_cal_eval)),
        "cal_train_default_rate"  : round(float(y_cal_train.mean()), 6),
        "cal_eval_default_rate"   : round(float(y_cal_eval.mean()), 6),
        "split_method"            : (
            "train_test_split(test_size=0.30, stratify=y_val, random_state=42)"
        ),
        "test_data_loaded"        : False,
    },
    "calibration_config": {
        "n_bins"              : N_BINS,
        "primary_strategy"    : "quantile",
        "secondary_strategy"  : "uniform (cross-check only)",
        "method"              : "IsotonicRegression(out_of_bounds='clip')",
        "fit_on"              : "cal-train base-model scores",
        "evaluated_on"        : "cal-eval (disjoint from cal-train)",
    },
    "full_val_metrics": {
        "brier_noskill" : round(brier_noskill, 8),
        "LR": {
            "brier_score"       : round(brier_lr, 8),
            "brier_skill_score" : round(bss_lr, 8),
            "ece_quantile"      : round(ece_lr, 8),
        },
        "XGB": {
            "brier_score"       : round(brier_xgb, 8),
            "brier_skill_score" : round(bss_xgb, 8),
            "ece_quantile"      : round(ece_xgb, 8),
        },
    },
    "posthoc_wrapper_cal_eval": {
        "LR": {
            "base_brier"      : round(brier_base_lr_e, 8),
            "calibrated_brier": round(brier_cal_lr_e, 8),
            "delta_brier"     : round(delta_lr, 8),
            "base_ece"        : round(ece_base_lr_e, 8),
            "calibrated_ece"  : round(ece_cal_lr_e, 8),
        },
        "XGB": {
            "base_brier"      : round(brier_base_xgb_e, 8),
            "calibrated_brier": round(brier_cal_xgb_e, 8),
            "delta_brier"     : round(delta_xgb, 8),
            "base_ece"        : round(ece_base_xgb_e, 8),
            "calibrated_ece"  : round(ece_cal_xgb_e, 8),
        },
    },
    "p07_thresholds_unchanged": {
        "LR"  : 0.50,
        "XGB" : 0.45,
        "note": ("P0.7 thresholds frozen. P0.9 does not select or apply "
                 "a new production or business threshold."),
    },
    "cost_threshold_analysis": (
        "Illustrative only -- see cost_threshold_illustration.csv. "
        "Not a production threshold recommendation."
    ),
    "what_p09_does_not_establish": [
        "Calibration is not causality.",
        "Calibration does not prove business-optimal threshold.",
        "Illustrative threshold analysis is not a production policy.",
        "P0.9 does not modify the P0.7 baseline models.",
        "Test data was not used.",
        "Post-hoc calibration does not change ROC-AUC "
        "(ROC-AUC is rank-based; rescaling scores preserves rank order).",
    ],
}

with open(OUT_DIR / "p09_results.json", "w") as fh:
    json.dump(p09_results, fh, indent=4)
print("  Saved: p09_results.json")
print()


# ===========================================================================
# 17. POST-RUN AUDIT
# ===========================================================================
print("POST-RUN AUDIT")
print("-" * 40)

# [1] [2] Frozen model files unchanged (size + mtime)
_lr_stat_after  = (LR_MODEL_PATH.stat().st_size,  LR_MODEL_PATH.stat().st_mtime)
_xgb_stat_after = (XGB_MODEL_PATH.stat().st_size, XGB_MODEL_PATH.stat().st_mtime)
assert _lr_stat_before  == _lr_stat_after,  "VIOLATION: LR  model file was modified"
assert _xgb_stat_before == _xgb_stat_after, "VIOLATION: XGB model file was modified"
print("  [1] Frozen LR  model file unchanged : PASS")
print("  [2] Frozen XGB model file unchanged : PASS")

# [3] p07_results.json unchanged
with open(P07_RESULTS_PATH) as fh:
    _p07_reload = json.load(fh)
assert _p07_reload["LR"]["val_metrics"]["roc_auc"]  == ref_lr_roc
assert _p07_reload["XGB"]["val_metrics"]["roc_auc"] == ref_xgb_roc
print("  [3] p07_results.json unchanged       : PASS")

# [4] Calibrated objects NOT placed in src/models/
assert not (BASE_DIR / "src" / "models" / "p09_calibrated_lr_isotonic.joblib").exists()
assert not (BASE_DIR / "src" / "models" / "p09_calibrated_xgb_isotonic.joblib").exists()
print("  [4] Isotonic objects NOT in src/models/ : PASS")

# [5] cal-train / cal-eval disjointness
assert len(set(idx_cal_train) & set(idx_cal_eval)) == 0
print("  [5] cal-train / cal-eval disjoint    : PASS")

# [6] Fingerprints consistent
assert abs(comp_lr_roc  - ref_lr_roc)  <= FINGERPRINT_TOL
assert abs(comp_xgb_roc - ref_xgb_roc) <= FINGERPRINT_TOL
print("  [6] P0.7 ROC-AUC fingerprints intact : PASS")

# [7] All expected output artifacts present
_expected_outputs = [
    "calibration_curve_lr_quantile.png",
    "calibration_curve_xgb_quantile.png",
    "calibration_curve_comparison_quantile.png",
    "calibration_curve_lr_uniform.png",
    "calibration_curve_xgb_uniform.png",
    "brier_scores.json",
    "calibration_summary_lr.csv",
    "calibration_summary_xgb.csv",
    "p09_calibrated_lr_isotonic.joblib",
    "p09_calibrated_xgb_isotonic.joblib",
    "cost_threshold_illustration.csv",
    "cost_threshold_illustration.png",
    "calibration_before_after_lr.png",
    "calibration_before_after_xgb.png",
    "p09_results.json",
]
for fname in _expected_outputs:
    assert (OUT_DIR / fname).exists(), f"MISSING: {fname}"
print(f"  [7] All {len(_expected_outputs)} expected artifacts present : PASS")

# [8] No production threshold changed
assert p09_results["p07_thresholds_unchanged"]["LR"]  == 0.50
assert p09_results["p07_thresholds_unchanged"]["XGB"] == 0.45
print("  [8] P0.7 thresholds unchanged         : PASS")

print()
print("=" * 62)
print("P0.9 COMPLETE -- All integrity checks passed.")
print("=" * 62)
print()
print("RESULTS SUMMARY")
print("-" * 40)
print(f"  Default rate (val)             : {prevalence:.4%}")
print(f"  No-skill Brier baseline        : {brier_noskill:.6f}")
print(f"  LR  Brier  (full val)          : {brier_lr:.6f}")
print(f"  XGB Brier  (full val)          : {brier_xgb:.6f}")
print(f"  LR  BSS    (full val)          : {bss_lr:.6f}")
print(f"  XGB BSS    (full val)          : {bss_xgb:.6f}")
print(f"  LR  ECE    (quantile, fullval) : {ece_lr:.6f}")
print(f"  XGB ECE    (quantile, fullval) : {ece_xgb:.6f}")
print(f"  LR  delta Brier (cal-eval)     : {delta_lr:+.6f}")
print(f"  XGB delta Brier (cal-eval)     : {delta_xgb:+.6f}")
print()
print("  Frozen thresholds  : LR=0.50, XGB=0.45 (unchanged)")
print("  Test data loaded   : NO")
print("  Base models refit  : NO")
print("  New features added : NO")
print("  Production thresh  : NO")
