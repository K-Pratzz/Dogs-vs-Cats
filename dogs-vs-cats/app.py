import streamlit as st
import numpy as np
import joblib
from PIL import Image
import cv2
from skimage.feature import hog

# Page config
st.set_page_config(page_title="Cats vs Dogs", layout="centered")

st.title("Cats vs Dogs Classifier 🐦‍🔥")

# Load everything
# Make sure these files are in the same directory as app.py
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# Constant must match training
IMG_SIZE = 128

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # 2. Convert PIL image to numpy array
    img = np.array(image)

    # 3. Resize BEFORE color conversion (matches training logic flow)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # 4. Normalize (matches training: img / 255.0)
    img = img / 255.0

    # 5. Convert to Grayscale
    # Since PIL opens as RGB, we convert RGB to GRAY
    if len(img.shape) == 3:  # Check if image is colored
        img_gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img

    # 6. 🔥 HOG Feature Extraction (Must match training parameters exactly)
    features = hog(
        img_gray,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    # 7. Reshape for transformation (1, -1) because it's a single sample
    features_reshaped = features.reshape(1, -1)

    # 8. Apply Scaler and PCA
    img_scaled = scaler.transform(features_reshaped)
    img_pca = pca.transform(img_scaled)

    # 9. Predict
    pred = model.predict(img_pca)[0]

    # Confidence (if available)
    try:
        proba = model.predict_proba(img_pca)
        confidence = np.max(proba)
    except:
        confidence = None

    # 10. Output
    st.divider()
    if pred == 1:
        st.subheader("Result: 🐶 Dog")
    else:
        st.subheader("Result: 🐱 Cat")

    if confidence is not None:
        st.write(f"**Confidence:** {confidence:.2%}")