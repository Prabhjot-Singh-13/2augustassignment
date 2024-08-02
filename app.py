import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Load the pre-trained model
model = joblib.load('linear_regression_model.pkl')

# Streamlit application
st.title('Linear Regression Model Deployment')

st.write("""
    Enter a value for X below to get the predicted value of y based on the linear regression model.
""")

# Input from the user
input_X = st.number_input('Enter value for X:', min_value=0.0, max_value=100.0, value=50.0)

# Make prediction
input_df = pd.DataFrame({'X': [input_X]})
prediction = model.predict(input_df)[0]

# Display the prediction
st.write(f'The predicted value of y is: {prediction:.2f}')

# Optionally show model details
if st.checkbox('Show Model Details'):
    st.write(f'Coefficient: {model.coef_[0]:.2f}')
    st.write(f'Intercept: {model.intercept_:.2f}')
