import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FE_ARTIFACT = os.path.join(PROJECT_ROOT, "artifacts", "feature_engineer_xgb.joblib")
XGB_MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "models", "xgboost_baseline.joblib")
CALIBRATOR_PATH = os.path.join(PROJECT_ROOT, "results", "calibration", "p09_calibrated_xgb_isotonic.joblib")
METADATA_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model_metadata.json")
