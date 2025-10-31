import streamlit as st
import joblib

model = joblib.load('regression.joblib')
size = st.number_input('Size', 1, None, step=1)
num_bedrooms = st.number_input('Number of rooms', 0, None, step=1)
is_garden = st.toggle('Is Garden')

prediction = model.predict([[size, num_bedrooms, is_garden]])
st.write(f'Prediction: {prediction[0]}')
