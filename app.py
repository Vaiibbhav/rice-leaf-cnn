import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

# -------------------------------
# CONFIG
# -------------------------------
IMG_SIZE = (150, 150)  # must match training size
MODEL_PATH = Path(r"C:\Users\Vaibhav\Documents\Data science Class\projects internship\riceleaf-disease-detection\models\rice_leaf_savedmodel.keras")

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Define class labels (must match training order)
class_labels = [
    "Bacterial leaf blight",
    "Brown spot",
    "Leaf smut"
]

# -------------------------------
# Prediction function
# -------------------------------
def predict_image(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    x = np.array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)
    p = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(p))
    return class_labels[idx], float(p[idx]), p

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Rice Leaf Disease Detection", layout="centered")

st.title("🌾 Rice Leaf Disease Detection")
st.write("Upload a rice leaf image to predict the disease type.")

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("🔍 Predict"):
        label, confidence, probs = predict_image(image)
        st.success(f"Prediction: **{label}** (Confidence: {confidence:.2%})")

        # Show all class probabilities as bar chart
        st.subheader("Confidence by class:")
        st.bar_chart({class_labels[i]: float(probs[i]) for i in range(len(class_labels))})
