import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# App Title
st.title('🍎🥦 Image Classification - Fruits & Vegetables')
st.subheader('Upload an image to classify its content')

# Load trained model
model_path = 'models/image_classify.keras'
model = load_model(model_path)

# Class labels
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
    'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange',
    'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish',
    'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

img_height, img_width = 180, 180

# File uploader
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', width=250)

        # Preprocess image
        image = image.resize((img_width, img_height))
        img_array = tf.keras.utils.img_to_array(image)
        img_batch = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_batch)
        score = tf.nn.softmax(prediction[0])
        predicted_class = data_cat[np.argmax(score)]
        confidence = np.max(score) * 100

        # Display result
        st.success(f"🔍 Prediction: **{predicted_class.capitalize()}**")
        st.write(f"🧠 Confidence: **{confidence:.2f}%**")

    except Exception as e:
        st.error(f"Error processing image: {e}")
