import os
import json
from fastapi import FastAPI, HTTPException
from app.schemas import LoanApplicationRequest, PredictionResponse, HealthResponse, ModelInfoResponse
from app.inference import predict_single_application, load_artifacts
from app.policy import get_risk_band

app = FastAPI(
    title="Credit Risk Prediction API",
    description="ML engineering portfolio implementation. Not a production lending system.",
    version="P0.10"
)

# Load artifacts at startup
@app.on_event("startup")
def startup_event():
    try:
        load_artifacts()
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        # Not exiting here so /health can still report, but /predict will fail.
        
@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok")

@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    METADATA_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model_metadata.json")
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        return ModelInfoResponse(**metadata)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model metadata not found")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: LoanApplicationRequest):
    try:
        # Pydantic has already validated the payload and forbidden leakage fields
        application_data = request.model_dump(exclude_none=False)
        
        # Core inference
        raw_prob, cal_prob = predict_single_application(application_data)
        
        # Policy
        risk_band = get_risk_band(cal_prob)
        
        # Metadata
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        METADATA_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model_metadata.json")
        try:
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            model_name = metadata.get("model_name", "unknown")
            model_version = metadata.get("training_phase", "unknown")
            calibration_method = metadata.get("calibration_artifact", "unknown")
        except FileNotFoundError:
            model_name = "unknown"
            model_version = "unknown"
            calibration_method = "unknown"
        
        return PredictionResponse(
            raw_default_probability=raw_prob,
            calibrated_default_probability=cal_prob,
            model_name=model_name,
            model_version=model_version,
            calibration_method=calibration_method,
            risk_band=risk_band
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Do not expose stack traces to client
        raise HTTPException(status_code=500, detail="Internal Inference Error")
