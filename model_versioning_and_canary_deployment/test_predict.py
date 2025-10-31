import json
import os

import requests

BASE = os.getenv("BASE_URL", "http://localhost:8000")

payload = {
    "instances": [
        {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2},
        {"sepal length (cm)": 6.7, "sepal width (cm)": 3.0, "petal length (cm)": 5.2, "petal width (cm)": 2.3}
    ]
}

resp = requests.post(f"{BASE}/predict", json=payload)
print("Status:", resp.status_code)
print("Body:", json.dumps(resp.json(), indent=2))
