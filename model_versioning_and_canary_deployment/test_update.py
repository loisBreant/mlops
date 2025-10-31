import json
import os
import requests

BASE = os.getenv("BASE_URL", "http://localhost:8000")

# Example model URI values you can try:
# - "models:/your-registered-model/1"
# - "runs:/<run_id>/random-forest-model"
# - "s3://..." or any supported MLflow artifact store
MODEL_URI = os.getenv("MODEL_URI_TO_LOAD")
if not MODEL_URI:
    raise SystemExit("Please set MODEL_URI_TO_LOAD env var to a valid MLflow model URI")

print("Updating model to:", MODEL_URI)
resp = requests.post(f"{BASE}/update-model", json={"model_uri": MODEL_URI})
print("Status:", resp.status_code)
print("Body:", json.dumps(resp.json(), indent=2))

# Verify predict still works after update (optional basic smoke test)
try:
    payload = {
        "instances": [
            {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}
        ]
    }
    r2 = requests.post(f"{BASE}/predict", json=payload)
    print("Predict status:", r2.status_code)
    print("Predict body:", json.dumps(r2.json(), indent=2))
except Exception as e:
    print("Predict failed after update:", e)
