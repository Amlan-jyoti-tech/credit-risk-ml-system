import os
import json
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from app.schemas import LoanApplicationRequest, PredictionResponse, HealthResponse, ModelInfoResponse
from app.inference import predict_single_application, load_artifacts
import app.inference as inf
from app.policy import get_risk_band
from app.config import METADATA_PATH

# Global cache for metadata
_model_metadata = {}
_artifacts_loaded = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_metadata, _artifacts_loaded
    try:
        load_artifacts()
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                _model_metadata = json.load(f)
        _artifacts_loaded = True
    except Exception as e:
        print(f"Error during startup: {e}")
        _artifacts_loaded = False
    yield
    # Cleanup on shutdown if needed

app = FastAPI(
    title="Credit Risk Prediction API",
    description="ML engineering portfolio implementation. Not a production lending system.",
    version="P0.10",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse)
def health_check():
    if not _artifacts_loaded or inf._fe is None or inf._xgb_model is None or inf._calibrator is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    return HealthResponse(status="ok")

@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if not _model_metadata:
        raise HTTPException(status_code=500, detail="Model metadata not found")
    return ModelInfoResponse(**_model_metadata)

@app.post("/predict", response_model=PredictionResponse)
def predict(request: LoanApplicationRequest):
    if not _artifacts_loaded:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
        
    try:
        # Pydantic has already validated the payload and forbidden leakage fields
        application_data = request.model_dump(exclude_none=False)
        
        # Core inference
        raw_prob, cal_prob = predict_single_application(application_data)
        
        # Policy
        risk_band = get_risk_band(cal_prob)
        
        model_name = _model_metadata.get("model_name", "unknown")
        model_version = _model_metadata.get("training_phase", "unknown")
        calibration_method = _model_metadata.get("calibration_artifact", "unknown")
        
        return PredictionResponse(
            raw_default_probability=raw_prob,
            calibrated_default_probability=cal_prob,
            model_name=model_name,
            model_version=model_version,
            calibration_method=calibration_method,
            risk_band=risk_band
        )
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(ve)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Do not expose stack traces to client
        raise HTTPException(status_code=500, detail="Internal Inference Error")
