"""
FastAPI application for phishing detection API.

Provides REST endpoints for:
1. Email phishing prediction (versioned)
2. Model information
3. Health check
"""

from fastapi import FastAPI, HTTPException, APIRouter, Request
from pydantic import BaseModel
from typing import List
import os
from pathlib import Path
import time
import logging
import json

from src.inference.predict import PhishingPredictor

# ============================================================================
# LOGGING SETUP
# ============================================================================

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

# Create a dedicated logger for API requests
api_logger = logging.getLogger("api_logger")
api_logger.setLevel(logging.INFO)

# Use a file handler to log to a file
handler = logging.FileHandler("logs/api_requests.log")

# Create a JSON formatter
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "details": record.args,
        }
        return json.dumps(log_record)

formatter = JsonFormatter()
handler.setFormatter(formatter)
api_logger.addHandler(handler)


# ============================================================================
# DATA MODELS
# ============================================================================


class EmailInput(BaseModel):
    """Input model for single email prediction."""
    text: str


class BatchEmailInput(BaseModel):
    """Input model for batch email prediction."""
    emails: List[str]


class PredictionOutput(BaseModel):
    """Output model for prediction result."""
    raw_score: float
    label: str
    confidence: float
    is_phishing: bool


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_name: str
    version: str
    languages: List[str]
    description: str


# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================


# Create FastAPI app
app = FastAPI(
    title="Phisher2025 API",
    description="Multilingual phishing email detection API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global predictor (loaded once at startup)
predictor = None


# ============================================================================
# MIDDLEWARE
# ============================================================================


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log requests and processing time."""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000  # in milliseconds

    log_details = {
        "method": request.method,
        "path": request.url.path,
        "client_ip": request.client.host,
        "user_agent": request.headers.get("user-agent"),
        "status_code": response.status_code,
        "process_time_ms": f"{process_time:.2f}",
    }
    
    api_logger.info("Request processed", extra=log_details)
    
    return response


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize predictor on app startup."""
    
    global predictor
    
    print("\n" + "=" * 70)
    print("PHISHER2025 API - STARTING UP")
    print("=" * 70)
    
    try:
        # Try to load model - prefer rebuilt .keras if it exists, else fall back to .h5
        rebuilt_model = Path("models/final_model/model_rebuilt.keras")
        default_model = "models/final_model/model.h5"
        model_path = os.getenv("MODEL_PATH", str(rebuilt_model if rebuilt_model.exists() else default_model))
        
        if not Path(model_path).exists():
            # Try alternative path
            model_path = "models/final_model/model_saved"
            
            if not Path(model_path).exists():
                print(f"✗ Model not found at {model_path}")
                print("  API will run but predictions will fail until model is loaded")
                predictor = None
            else:
                predictor = PhishingPredictor(model_path)
                print(f"✓ Loaded model from {model_path}")
        else:
            predictor = PhishingPredictor(model_path)
            print(f"✓ Loaded model from {model_path}")
    
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        predictor = None
    
    print("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on app shutdown."""
    print("\n✓ Shutting down Phisher2025 API")


# ============================================================================
# API ROUTER (V1)
# ============================================================================


# Create a router for version 1 of the API
api_v1_router = APIRouter(prefix="/api/v1", tags=["Predictions"])


@api_v1_router.post("/predict", response_model=PredictionOutput)
async def predict_email(email: EmailInput):
    """
    Predict if an email is phishing.
    
    Args:
        email: EmailInput with email text
        
    Returns:
        PredictionOutput with prediction result
        
    Raises:
        HTTPException: If model is not loaded
    """
    
    # Check if model is loaded
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists."
        )
    
    try:
        # Validate input
        if not email.text or len(email.text.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Email text cannot be empty"
            )
        
        # Make prediction
        score, label = predictor.predict_single(email.text)
        
        # Calculate confidence
        confidence = score if label == "PHISHING" else 1.0 - score
        
        return PredictionOutput(
            raw_score=score,
            label=label,
            confidence=confidence,
            is_phishing=label == "PHISHING"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@api_v1_router.post("/predict-batch", response_model=List[PredictionOutput])
async def predict_batch(batch: BatchEmailInput):
    """
    Predict if multiple emails are phishing (batch prediction).
    
    Args:
        batch: BatchEmailInput with list of email texts
        
    Returns:
        List of PredictionOutput with prediction results
        
    Raises:
        HTTPException: If model is not loaded or batch size is invalid
    """
    
    # Check if model is loaded
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists."
        )
    
    try:
        # Validate input
        if not batch.emails or len(batch.emails) == 0:
            raise HTTPException(
                status_code=400,
                detail="Email list cannot be empty"
            )
        
        if len(batch.emails) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 emails per batch"
            )
        
        # Make predictions
        results = predictor.predict_batch(batch.emails)
        
        # Convert to output format
        output = []
        for score, label in results:
            confidence = score if label == "PHISHING" else 1.0 - score
            output.append(
                PredictionOutput(
                    raw_score=score,
                    label=label,
                    confidence=confidence,
                    is_phishing=label == "PHISHING"
                )
            )
        
        return output
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during batch prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


# Include the versioned router in the main app
app.include_router(api_v1_router)


# ============================================================================
# ENDPOINTS - SYSTEM & INFO
# ============================================================================


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with status and model availability
    """
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """
    Get model information.
    
    Returns:
        ModelInfoResponse with model details
    """
    return ModelInfoResponse(
        model_name="Hybrid CNN-LSTM",
        version="1.0.0",
        languages=["English", "Swahili", "Chinese", "Russian"],
        description="Multilingual phishing email detection model using CNN-LSTM architecture"
    )


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Phisher2025 API",
        "version": "1.0.0",
        "description": "Multilingual phishing email detection",
        "endpoints": {
            "health": "/health",
            "model-info": "/model-info",
            "predict": "/api/v1/predict (POST)",
            "predict-batch": "/api/v1/predict-batch (POST)",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return {
        "error": "Internal server error",
        "message": str(exc)
    }


# ============================================================================
# MAIN - FOR RUNNING WITH UVICORN
# ============================================================================


if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    # Command: python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
