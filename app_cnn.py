import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("car_bike_classifier.h5")

# Define class labels
class_labels = ["Bike", "Car"]  # Adjust according to your dataset

# Function to preprocess and predict image
def predict_image(img):
    img = img.resize((150, 150))  # Resize to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    prediction = model.predict(img_array)
    class_index = int(prediction[0][0] > 0.5)  # 0: Bike, 1: Car
    return class_labels[class_index]

# Streamlit UI
st.title("Car vs. Bike Image Classifier")
st.write("Upload an image, and the model will classify it as a Car or Bike.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)  # Load image
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    # Predict on uploaded image
    result = predict_image(image_pil)

    # Display result
    st.subheader("Prediction:")
    st.write(f"### 🚀 The model predicts: **{result}**")

