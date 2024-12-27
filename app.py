import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Load the pre-trained model
model = tf.keras.models.load_model("model.h5")

# Function to preprocess the uploaded image for prediction
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 (same size as MNIST images)
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = image.reshape(1, 28, 28, 1)  # Add batch dimension
    return image

# Streamlit UI
st.title("Hand Written Digit Recognition")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    
    # Display the result
    st.write(f"Predicted digit: {predicted_class}")
