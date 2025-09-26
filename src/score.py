# score.py
import json
import os
import joblib
import pandas as pd
import subprocess


MODEL_ENV_NAME = "AZUREML_MODEL_PATH"  # optional env var
DEFAULT_MODEL_FILE = "churn_pipeline.joblib"

def init():
    global model
    # Azure ML mounts the model here
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    model_path = os.path.join(model_dir, "churn_pipeline.joblib")
    
    # Load model
    model = joblib.load(model_path)

def run(raw_data):
    """
    Azure will send JSON body. Accepts:
     - list of dicts: [{"Age":..., ...}, ...]
     - single dict: {"Age":...}
    Returns JSON serializable dict.
    """
    try:
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data

        if isinstance(data, dict):
            # single example
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            # maybe Azure sends {"data": [...]}
            if isinstance(data, dict) and "data" in data:
                df = pd.DataFrame(data["data"])
            else:
                return {"error": "Unsupported input format."}

        preds = model.predict(df).tolist()
        proba = model.predict_proba(df)[:, 1].tolist() if hasattr(model, "predict_proba") else None

        result = {"predictions": preds}
        if proba is not None:
            result["probabilities"] = proba
        return result
    except Exception as e:
        return {"error": str(e)}
