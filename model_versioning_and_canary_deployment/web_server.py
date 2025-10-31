import os
import threading
from typing import Any, Dict, List

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc


def _sanitize_uri(uri: str) -> str:
    # Remove accidental surrounding quotes that may come from YAML/env formatting
    if uri and ((uri.startswith('"') and uri.endswith('"')) or (uri.startswith("'") and uri.endswith("'"))):
        return uri[1:-1]
    return uri


def load_model(model_uri: str):
    try:
        clean_uri = _sanitize_uri(model_uri)
        return mlflow.pyfunc.load_model(clean_uri)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model from URI '{model_uri}': {e}")


app = FastAPI(title="MLflow Model Service", version="1.0.0")

# Initialize global state (model is loaded on startup via env vars)
app.state.model = None
app.state.model_uri = None
app.state.lock = threading.Lock()


@app.on_event("startup")
def startup_event():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    initial_model_uri = os.getenv("MODEL_URI")
    if initial_model_uri:
        # Load initial model if provided
        model = load_model(initial_model_uri)
        with app.state.lock:
            app.state.model = model
            app.state.model_uri = _sanitize_uri(initial_model_uri)


class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]


@app.post("/predict")
def predict(req: PredictRequest):
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Set MODEL_URI or call /update-model.")
    try:
        df = pd.DataFrame(req.instances)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload format: {e}")
    try:
        preds = app.state.model.predict(df)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


class UpdateRequest(BaseModel):
    model_uri: str


@app.post("/update-model")
def update_model(req: UpdateRequest):
    new_model = load_model(req.model_uri)
    with app.state.lock:
        app.state.model = new_model
        app.state.model_uri = _sanitize_uri(req.model_uri)
    return {"status": "updated", "model_uri": app.state.model_uri}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.model is not None}


@app.get("/info")
def info():
    return {"model_uri": app.state.model_uri}


@app.get("/")
def root():
    return {"message": "MLflow Model Service is running.", "model_uri": app.state.model_uri}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
