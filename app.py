import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('model/plant_disease_model.h5')

# Customize based on your dataset classes
class_names = ['Apple___healthy', 'Apple___rust', 'Tomato___Late_blight', 'Tomato___healthy']

st.title("🌿 AgroVision AI")
st.markdown("Upload a leaf image to detect plant disease.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((128, 128))
    st.image(image, caption='Uploaded Leaf', use_column_width=True)

    img = img_to_array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    disease = class_names[class_index]
    confidence = round(np.max(prediction) * 100, 2)

    st.success(f"**Prediction:** {disease}")
    st.info(f"**Confidence:** {confidence}%")

    if "healthy" in disease:
        st.success("✅ No disease detected. Your plant is healthy.")
    else:
        st.warning("⚠️ Disease detected. Suggest spraying neem oil and isolating the plant.")
