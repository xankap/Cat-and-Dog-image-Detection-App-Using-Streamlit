import os

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("best.onnx")

# Load model
model2 = tf.keras.models.load_model("best.onnx")

st.title("🐱🐶 Cats vs Dogs Classifier")
st.write("Upload an image and the model will predict whether it is a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((150, 150))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success("🐶 This is a Dog!")
    else:
        st.success("🐱 This is a Cat!")

st.title("Ripe and unripe tomato classifier")
st.write("Upload an image and the model will predict whether tomato is ripe or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((150, 150))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model2.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success("Tomato is ripe!")
    else:
        st.success("Tomatoe is unripe")
