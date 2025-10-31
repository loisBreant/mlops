from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import List
import eurybia 

model = joblib.load('regression.joblib')

production_data = []

training_data = pd.read_csv('houses.csv')
X_train = training_data[['size', 'nb_rooms', 'garden']].copy()

app = FastAPI()


from functools import partial
from alibi_detect.cd.tensorflow import preprocess_drift

import numpy as np

idx_ref = np.random.choice(len(X_train), size=len(X_train)//10, replace=False)
X_ref = X_train.iloc[idx_ref].to_numpy()

from alibi_detect.cd import KSDrift
detector: KSDrift = KSDrift(
    X_ref,
    p_val=0.05,
)

class PredictionInput(BaseModel):
    size: float
    nb_rooms: int
    garden: bool

class PredictionResponse(BaseModel):
    y_pred: float

class DriftResponse(BaseModel):
    has_drift: bool
    message: str


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionInput):
    global production_data
    
    features = [[input_data.size, input_data.nb_rooms, int(input_data.garden)]]
    
    prediction = model.predict(features)
    
    production_data.append({
        'size': input_data.size,
        'nb_rooms': input_data.nb_rooms,
        'garden': int(input_data.garden)
    })

    return PredictionResponse(y_pred=float(prediction[0]))


@app.get('/detect-drift', response_model=DriftResponse)
def detect_drift():
    global production_data, X_train
    
    min_samples = 10
    if len(production_data) < min_samples:
        return DriftResponse(
            has_drift=False,
            message=f"Insufficient data. Need {min_samples} samples, have {len(production_data)}",
        )
    
    X_prod = pd.DataFrame(production_data).to_numpy()
    has_drift = detector.predict(X_prod)['data']['is_drift']

    if has_drift:
        message = f"Drift detected."
    else:
        message = "No significant drift detected."
    
    return DriftResponse(
        has_drift=has_drift, 
        message=message,
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8051)