"""
Plant Disease Detection - Streamlit web app.

Upload a leaf image and the CNN in `plant_disease_model.h5` classifies it as:
  - Corn (maize)  - Common rust
  - Potato        - Early blight
  - Tomato        - Bacterial spot

Preprocessing here MUST match training (see Plant_Disease_Detection.ipynb):
  OpenCV BGR image -> resize to 256x256 -> divide by 225.0
"""

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_PATH = "plant_disease_model.h5"
IMG_SIZE = (256, 256)
NORMALIZATION = 225.0  # same constant used during training

# Class order MUST match the label indices used in training.
# Training read the dataset folders with os.listdir(), which returned:
#   0 -> Corn_(maize)___Common_rust_
#   1 -> Potato___Early_blight
#   2 -> Tomato___Bacterial_spot
CLASS_NAMES = (
    ("Corn", "Common rust"),
    ("Potato", "Early blight"),
    ("Tomato", "Bacterial spot"),
)


@st.cache_resource
def load_model():
    """Load the trained Keras model once and reuse it across reruns."""
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess(bgr_image: np.ndarray) -> np.ndarray:
    """Resize + normalise exactly like the training pipeline, add batch dim."""
    resized = cv2.resize(bgr_image, IMG_SIZE)
    scaled = resized.astype("float32") / NORMALIZATION
    return np.expand_dims(scaled, axis=0)  # (1, 256, 256, 3)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌱", layout="centered")

st.title("🌱 Plant Disease Detection")
st.markdown(
    "Upload an image of a plant leaf. The model detects "
    "**Corn Common rust**, **Potato Early blight** and **Tomato Bacterial spot**."
)

model = load_model()

plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
submit = st.button("Predict Disease", type="primary")

if submit:
    if plant_image is None:
        st.warning("Please upload a leaf image first.")
    else:
        # Decode upload into an OpenCV (BGR) image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if opencv_image is None:
            st.error("Could not read this file as an image. Please try another one.")
        else:
            st.image(opencv_image, channels="BGR", caption=f"Uploaded image {opencv_image.shape}")

            probs = model.predict(preprocess(opencv_image), verbose=0)[0]
            idx = int(np.argmax(probs))
            plant, disease = CLASS_NAMES[idx]
            confidence = float(probs[idx]) * 100

            st.success(f"This is a **{plant}** leaf with **{disease}**  ({confidence:.1f}% confidence)")

            st.markdown("##### Class probabilities")
            for (p, d), prob in zip(CLASS_NAMES, probs):
                st.progress(float(prob), text=f"{p} - {d}: {prob * 100:.1f}%")

            st.caption(
                "This model only knows the 3 classes above. Images of other plants "
                "or diseases will still be forced into one of these classes."
            )
