import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

st.title('🍎🥦 Image Classification - Fruits & Vegetables')
st.subheader('Upload an image to classify its content')

MODEL_PATH = 'models/image_classify.keras'
model = load_model(MODEL_PATH)

data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
    'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange',
    'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish',
    'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

IMG_HEIGHT, IMG_WIDTH = 180, 180

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read image file bytes to numpy array (in memory)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image. Please upload a valid image file.")

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (IMG_WIDTH, IMG_HEIGHT))

        # Show uploaded image
        st.image(image_resized, caption='Uploaded Image', width=250)

        # Prepare batch dimension and normalize if needed
        img_batch = np.expand_dims(image_resized, axis=0)

        # Predict
        prediction = model.predict(img_batch)
        score = tf.nn.softmax(prediction[0])
        predicted_class = data_cat[np.argmax(score)]
        confidence = np.max(score) * 100

        # Display prediction
        st.success(f"🔍 Prediction: **{predicted_class.capitalize()}**")
        st.write(f"🧠 Confidence: **{confidence:.2f}%**")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
