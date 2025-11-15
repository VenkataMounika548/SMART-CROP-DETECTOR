"""
SMART SOIL → CROP RECOMMENDATION SYSTEM
MAIN CODE (TRAIN + PREDICT + UI)

Run:
    pip install streamlit tensorflow pillow numpy scikit-learn seaborn
    streamlit run app.py
"""

import os
import zipfile
import shutil
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ---------------- CONFIG ----------------
ZIP_PATH = r"C:\Users\mouni\OneDrive\Documents\smartcropdetector\DATASET.zip"  # YOUR ZIP PATH
RAW_DIR = "soil_raw"
DATASET_DIR = "soil_dataset"
MODEL_FILE = "smart_soil_model.h5"
CLASS_JSON = "soil_classes.json"

IMG_SIZE = (180, 180)
BATCH = 32
EPOCHS = 10

# Soil → Crop Mapping
SOIL_TO_CROP = {
    "Red Soil": ["Cotton", "Millets", "Groundnut", "Potato"],
    "Black Soil": ["Soybean", "Sunflower", "Cotton"],
    "Loamy Soil": ["Tomato", "Banana", "Wheat", "Sugarcane"],
    "Clay Soil": ["Rice", "Wheat", "Sugarcane"],
    "Sandy Soil": ["Carrot", "Watermelon", "Groundnut"]
}

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Soil → Crop Recommendation", layout="centered")
st.title("🌱 Smart Soil → Crop Recommendation System")
st.write("Upload a soil image to detect soil type and get suitable crops.")


# ---------------- Extract ZIP ----------------
def extract_dataset():
    if not os.path.exists(ZIP_PATH):
        st.error(f"Dataset ZIP not found at: {ZIP_PATH}")
        st.stop()

    if os.path.exists(RAW_DIR):
        shutil.rmtree(RAW_DIR)
    os.makedirs(RAW_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(RAW_DIR)


# ---------------- Prepare dataset ----------------
def prepare_dataset():
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    os.makedirs(DATASET_DIR, exist_ok=True)

    for root, dirs, files in os.walk(RAW_DIR):
        imgs = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(imgs) >= 5:
            class_name = os.path.basename(root)
            dest = os.path.join(DATASET_DIR, class_name)
            os.makedirs(dest, exist_ok=True)
            for img in imgs:
                shutil.copy(os.path.join(root, img), dest)


# ---------------- Build CNN Model ----------------
def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=IMG_SIZE+(3,)),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model


# ---------------- Train Model ----------------
def train_model():
    datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH,
        class_mode="categorical", subset="training"
    )

    val_gen = datagen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH,
        class_mode="categorical", subset="validation"
    )

    with open(CLASS_JSON, "w") as f:
        json.dump(train_gen.class_indices, f)

    model = build_model(train_gen.num_classes)
    model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
    model.save(MODEL_FILE)

    return model


# ---------------- Load or Train ----------------
def load_or_train():
    if os.path.exists(MODEL_FILE) and os.path.exists(CLASS_JSON):
        model = load_model(MODEL_FILE)
        with open(CLASS_JSON) as f:
            class_map = json.load(f)
        return model, class_map

    extract_dataset()
    prepare_dataset()
    model = train_model()

    with open(CLASS_JSON) as f:
        class_map = json.load(f)

    return model, class_map


# Load model
model, class_map = load_or_train()
idx_to_class = {v:k for k,v in class_map.items()}


# ---------------- Predict Soil ----------------
def predict_soil(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0)

    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    soil = idx_to_class[idx]
    conf = round(float(preds[idx]) * 100, 2)

    return soil, conf


# ---------------- UI TABS ----------------
tab1, tab2 = st.tabs(["📁 Upload Soil Image", "📷 Camera Capture"])

with tab1:
    file = st.file_uploader("Upload soil image", type=["jpg","jpeg","png"])
    if file:
        img = Image.open(file)
        st.image(img)

        soil, conf = predict_soil(img)
        st.success(f"Detected Soil: {soil} ({conf}%)")

        st.subheader("🌾 Suitable Crops:")
        for crop in SOIL_TO_CROP.get(soil, []):
            st.write(f"- {crop}")

with tab2:
    cam = st.camera_input("Capture soil image")
    if cam:
        img = Image.open(cam)
        st.image(img)

        soil, conf = predict_soil(img)
        st.success(f"Detected Soil: {soil} ({conf}%)")

        st.subheader("🌾 Suitable Crops:")
        for crop in SOIL_TO_CROP.get(soil, []):
            st.write(f"- {crop}")
