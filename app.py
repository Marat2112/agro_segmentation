import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import tempfile
import os

from morphometry import analyze_segmentation
from calibration import auto_compute_scale_from_folder
from morphometry import draw_metrics_overlay


# конфиг

MODEL_PATH = "runs/segment/yolo26_final/weights/best.pt"
CALIB_DIR = "calib"
SQUARE_MM = 10

st.set_page_config(page_title="Agro Morphometry", layout="wide")
st.title("🌱 Plant Morphometry Demo")


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()


#  масштаб

@st.cache_data
def get_scale():
    return auto_compute_scale_from_folder(CALIB_DIR, square_size_mm=SQUARE_MM)

scale = get_scale()

if scale is None:
    st.error("❌ Не удалось вычислить масштаб по шахматке")
else:
    st.success(f"✅ Масштаб: {scale:.5f} mm/pixel")


# интерфейс загрузки обработки 

uploaded = st.file_uploader(
    "Загрузите изображение растения",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    suffix = Path(uploaded.name).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        temp_path = tmp.name

    # инференс
    result = model(temp_path, imgsz=640)[0]

    # морфометрия
    metrics_px = analyze_segmentation(result)

    if scale is not None:
        metrics_mm = {
            "root_length_mm": metrics_px["root_length_px"] * scale,
            "root_area_mm2": metrics_px["root_area_px"] * (scale ** 2),
            "stem_length_mm": metrics_px["stem_length_px"] * scale,
            "leaf_area_mm2": metrics_px["leaf_area_px"] * (scale ** 2),
            "leaf_count": metrics_px["leaf_count"],
        }
    else:
        metrics_mm = None

    # визуализация
    plotted = result.plot()
    plotted = draw_metrics_overlay(plotted, metrics_mm)
    plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(plotted, caption="Segmentation result", use_container_width=True)

    with col2:
        st.subheader("📊 Morphometry")

        st.write("**Pixels:**")
        st.json(metrics_px)

        if metrics_mm:
            st.write("**Millimeters:**")

            st.json(metrics_mm)
