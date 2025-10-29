import pandas as pd 
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv('houses.csv')
X = df[['size', 'nb_rooms', 'garden']]
y = df['price']
model = LinearRegression()
model.fit(X, y)
joblib.dump(model, "regression.joblib")

X = pd.DataFrame({'size': [100], 'nb_rooms': [3], 'garden': [0]})
y_pred = model.predict(X)
print(f"Predicted price: {y_pred[0]}")
