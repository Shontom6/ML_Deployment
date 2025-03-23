import streamlit as st
import pickle
import numpy as np

# Load the trained Random Forest model
with open("winequality-white.pkl", "rb") as f:
    model = pickle.load(f)

# Define input features
features = [
    "Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar",
    "Chlorides", "Free Sulfur Dioxide", "Total Sulfur Dioxide", "Density",
    "pH", "Sulphates", "Alcohol"
]

# Streamlit UI
st.title("🍷 Wine Quality Prediction App")
st.write("Enter the wine's chemical properties to predict its quality.")

# Collect user input for each feature
input_data = []
for feature in features:
    value = st.number_input(f"Enter {feature}:", min_value=0.0, step=0.01)
    input_data.append(value)

# Convert input list to numpy array
input_array = np.array([input_data]).reshape(1, -1)

# Predict wine quality when the user clicks the button
if st.button("Predict Quality"):
    prediction = model.predict(input_array)
    predicted_quality = int(round(prediction[0]))  # Round to nearest integer
    st.success(f"🍾 Predicted Wine Quality: {predicted_quality}")
