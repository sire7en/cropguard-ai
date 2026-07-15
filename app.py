# Import libraries

import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from datetime import datetime
import os, tempfile

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from fpdf import FPDF

# Page setup
st.set_page_config(page_title="CropGuard-AI", page_icon="🍅", layout="wide")

# Model loading 
@st.cache_resource
def load_trained_model():
    return load_model("model.keras")

model = load_trained_model()
    

# Crop diseases information
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Healthy"
]

DISEASE_INFO = {
    "Tomato___Bacterial_spot": {"severity":"High", "advice":"Remove infected leaves immediately. Apply copper-based bactericide. Avoid overhead irrigation."},
    "Tomato___Early_blight": {"severity":"Medium", "advice":"Apply chlorothalonil or mancozeb fungicide. Remove infected leaves and improve spacing."},
    "Tomato___Late_blight": {"severity":"High", "advice":"Apply copper-based fungicide immediately. Remove and destroy infected plant material."},
    "Tomato___Leaf_Mold": {"severity":"Medium", "advice":"Improve ventilation and reduce humidity. Apply fungicide containing chlorothalonil."},
    "Tomato___Septoria_leaf_spot": {"severity":"Medium", "advice":"Remove infected leaves. Apply fungicide and mulch around plant base."},
    "Tomato___Spider_mites Two-spotted_spider_mite": {"severity":"Medium", "advice":"Apply miticide or neem oil spray. Increase humidity around plants."},
    "Tomato___Target_Spot": {"severity":"Medium", "advice":"Apply fungicide containing azoxystrobin. Avoid overhead watering."},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"severity":"High", "advice":"No cure available. Remove infected plants and control whitefly vectors."},
    "Tomato___Tomato_mosaic_virus": {"severity":"High", "advice":"Remove infected plants. Disinfect tools and control aphid vectors."},
    "Tomato___Healthy": {"severity":"None", "advice":"No action needed. Your tomato plant appears healthy!"}
}

if "history_logs" not in st.session_state:
    st.session_state.history_logs = []

def clean_classname(name):
    return name.replace("Tomato___", "").replace("_", " ")

def predict_leaf(image, model):
    img = preprocess_input(np.expand_dims(np.array(image.resize((224,224)), dtype="float32"), axis=0))
    prediction = model.predict(img, verbose=0)[0]
    index = np.argmax(prediction)
    return CLASS_NAMES[index], float(prediction[index])

# PDF report generator
def generate_safe_pdf(disease, confidence, severity, advice, timestamp):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "CropGuard-AI - Field Diagnostic Report", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, f"Generated: {timestamp}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Diagnostic Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 7, f"Condition: {disease}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Confidence: {confidence:.2f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Severity: {severity}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Recommended Treatment", new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, advice)

    file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(file.name)
    return file.name


# Sidebar navigation
st.sidebar.title("🍅 CropGuard-AI")
st.sidebar.caption("Version 2.1")
st.sidebar.divider()

page = st.sidebar.radio("App Menu", ["🔍 Scan Leaf", "📋 History", "ℹ️ Info"])
st.sidebar.divider()
st.sidebar.info("**Note:** This tool processes tomato leaf samples only. Use clear images with good lighting for better accuracy.")

# Scan Leaf page
if page == "🔍 Scan Leaf":
    st.title("Tomato Leaf Disease Detector")
    st.warning(
        "**⚠️ Warning:** Upload or capture clear images of **tomato leaves only.**\n\n"
        "Uploading non-tomato images may produce incorrect predictions. A validation layer is currently in development."
    )

    upload_tab, camera_tab = st.tabs(["📁 Upload", "📷 Camera"])
    image = None

    with upload_tab:
        uploaded = st.file_uploader("Select Target Leaf Image File", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if uploaded:
            image = Image.open(uploaded).convert("RGB")

    with camera_tab:
        captured = st.camera_input("Capture Live Field Specimen")
        if captured:
            image = Image.open(captured).convert("RGB")
    if image:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Image Uploaded", use_container_width=True)
            analyze = st.button("🔬 Analyze", use_container_width=True, type="primary")

        with col2:
            if analyze:
                if model is None:
                    st.error("Model unavailable.")
                    st.stop()

                with st.spinner("Processing image..."):
                    prediction, confidence = predict_leaf(image, model)

                confidence_pct = confidence * 100
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Confidence gate prevents unreliable predictions
                if confidence >= 0.75:

                    info = DISEASE_INFO[prediction]
                    disease = clean_classname(prediction)

                    st.success(f"### Diagnosis: {disease}")
                    st.metric("Inference Confidence", f"{confidence_pct:.2f}%")
                    st.write(f"**Severity:** {info['severity']}")
                    st.info(f"**Recommended Treatment Plan:**\n{info['advice']}")

                    st.session_state.history_logs.append({
                        "Timestamp": timestamp,
                        "Prediction": disease,
                        "Confidence": f"{confidence_pct:.2f}%",
                        "Severity": info["severity"],
                        "Status": "Healthy" if "Healthy" in prediction.lower() else "Diseased"
                    })

                    try:
                        pdf_path = generate_safe_pdf(disease, confidence_pct, info["severity"], info["advice"], timestamp)

                        with open(pdf_path, "rb") as pdf:
                            st.download_button(
                                "📥 Download PDF Report",
                                data=pdf,
                                file_name=f"cropguard_report_{datetime.now():%Y%m%d_%H%M%S}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )

                        os.remove(pdf_path)

                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")

                else:
                    st.error("### ❌ Specimen Validation Failed")
                    st.write(f"**Prediction Confidence:** {confidence_pct:.2f}%")
                    st.markdown(
                        """
                        **Action Recommended:**
                        1. Verify the image contains a tomato leaf only.
                        2. Ensure good lighting and reduce background interference.
                        """
                    )

# History page
elif page == "📋 History":
    st.title("📋 Session Logs")

    if not st.session_state.history_logs:
        st.info("No logs generated.")

    else:
        df = pd.DataFrame(st.session_state.history_logs)
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export CSV",
            data=csv,
            file_name=f"cropguard_logs_{datetime.now():%Y%m%d}.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary"
        )

        if st.button("🗑️ Clear Logs"):
            st.session_state.history_logs = []
            st.rerun()


# Information page
elif page == "ℹ️ Info":

    st.markdown(
        """
        ## What is CropGuard?

        CropGuard-AI is a deep learning image classification system
        designed to detect tomato leaf diseases from images.

        The model classifies tomato leaves into 10 categories:
        - 9 disease classes
        - 1 healthy class

        The application provides disease identification,
        confidence scoring, severity level, and recommended
        treatment guidance.

        Designed to support tomato farmers and agricultural
        extension services.
        """
    )

    st.caption("CropGuard-AI · Capstone Project · Built with Python, TensorFlow & Streamlit")