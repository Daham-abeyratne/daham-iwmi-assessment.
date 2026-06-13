import sys
import tempfile
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.FaceMaskInference import BasicInference

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pth"


st.set_page_config(page_title="Face Mask Detector", page_icon=":)", layout="centered")


@st.cache_resource
def load_inferer():
    return BasicInference(str(MODEL_PATH))


st.title("""
        Face Mask Detection

        Deep Learning based face mask classification using a custom CNN trained from scratch.

        **Features**
        - Face detection using Haar Cascade
        - Mask / No-Mask classification
        - Confidence visualization
        - Real-time image analysis
        """)

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Validation Accuracy", "97.0%")

with m2:
    st.metric("Test Accuracy", "95.9%")

with m3:
    st.metric("Classes", "2")

if not MODEL_PATH.exists():
    st.error(
        "Model checkpoint not found. Run `train.py` first to generate "
        "`models/best_model.pth`, then restart this app."
    )
    st.stop()

inferer = load_inferer()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    tab1, tab2 = st.tabs(["Classify Whole Image", "Detect Faces"])

    # ---- Tab 1: whole-image classification ----
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", width="stretch")

        label, confidence, all_probs = inferer.predict(image)

        with col2:
            st.subheader("Prediction")
            st.markdown(f"### {label}")
            st.write(f"Confidence: **{confidence:.2f}%**")

        # all_probs is a dict {class_name: probability}
        classes = list(all_probs.keys())
        values = list(all_probs.values())

        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(classes, values, color="#4C72B0")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.set_title("Class Probabilities")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v * 100:.1f}%", ha="center")
        st.pyplot(fig)

    # ---- Tab 2: Haar Cascade face detection + per-face classification ----
    with tab2:
        st.write("Faces detected via Haar Cascade, each classified individually.")

        # detect_images() expects a file path, so write the upload to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            tmp_path = Path(tmp.name)

        try:
            results, annotated_bgr = inferer.detect_images(str(tmp_path))

            # annotated image comes back as BGR (OpenCV) -> convert to RGB for display
            annotated_rgb = annotated_bgr[:, :, ::-1]
            st.image(annotated_rgb, caption="Detected Faces", width="stretch")

            if len(results) == 0:
                st.info("No faces detected and no fallback prediction returned.")
            else:
                for i, r in enumerate(results, start=1):
                    box_info = f" at box {r['box']}" if r.get("box") else ""
                    st.write(f"**Face {i}:** {r['label']} — {r['confidence']:.2f}%{box_info}")
        finally:
            tmp_path.unlink()
else:
    st.info("Upload an image above to get a prediction.")
    st.write("Upload a `.jpg`, `.jpeg`, or `.png` image.")  
