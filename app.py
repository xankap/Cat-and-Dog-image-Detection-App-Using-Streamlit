import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐱🐶",
    layout="centered"
)

st.title("🐱🐶 Cats vs Dogs Image Classifier")
st.write("Upload an image and the model will predict whether it is a cat or a dog.")


# Load model from Hugging Face

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="xannychuks/cats_dog_model",
        filename="cats_dogs_model.h5"
    )

    model = tf.keras.models.load_model(model_path)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = load_model()


# Image upload

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((150, 150))

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success("🐶 Prediction: Dog")
    else:
        st.success("🐱 Prediction: Cat")
