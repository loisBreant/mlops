from fastapi import FastAPI
import joblib

model = joblib.load('regression.joblib')


app = FastAPI()

@app.get("/")
def root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(size: int, nb_rooms: int, garden: bool):
    prediction = model.predict([[size, nb_rooms, garden]])
    return {"y_pred": prediction[0]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8051)

