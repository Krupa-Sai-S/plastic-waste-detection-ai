import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your trained model
model = tf.keras.models.load_model("plastic_classifier.h5")

# Update this list to match your model's class names
class_names = ["Plastic Bottle", "Plastic Bag", "Other Plastic"]  # Replace with your actual class labels

# Set image size as per your training model
IMAGE_SIZE = (224, 224)  # Update if your model expects a different size

st.title("Plastic Waste Classification")
st.write("Upload an image of plastic waste to classify it")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_name = class_names[class_index]
    confidence = prediction[0][class_index]

    st.success(f"Prediction: **{class_name}**")
    st.info(f"Confidence: **{confidence:.2f}**")
