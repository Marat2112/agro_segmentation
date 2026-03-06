import streamlit as st
import pandas as pd
import cv2
import tempfile
from pathlib import Path

from ultralytics import YOLO
from morphometry import analyze_segmentation, draw_metrics_overlay
from calibration import auto_compute_scale_from_folder

MODEL_PATH = "runs/segment/yolo26_final/weights/best.pt"
CALIB_DIR = "calib"

st.title("🌱 AI анализ растений")

model = YOLO(MODEL_PATH)

scale = auto_compute_scale_from_folder(CALIB_DIR, square_size_mm=10)

st.write("Масштаб:", scale, "мм/пиксель")

uploaded_files = st.file_uploader(
    "Загрузите изображения растений",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:

    results = []

    for uploaded in uploaded_files:

        suffix = Path(uploaded.name).suffix

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            path = tmp.name

        result = model(path, imgsz=640)[0]

        metrics_px = analyze_segmentation(result)

        metrics_mm = {
            "root_length_mm": metrics_px["root_length_px"] * scale,
            "root_area_mm2": metrics_px["root_area_px"] * scale**2,
            "stem_length_mm": metrics_px["stem_length_px"] * scale,
            "leaf_area_mm2": metrics_px["leaf_area_px"] * scale**2,
            "leaf_count": metrics_px["leaf_count"]
        }

        results.append({
            "image": uploaded.name,
            **metrics_mm
        })

        img = result.plot()
        img = draw_metrics_overlay(img, metrics_mm)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.image(img, caption=uploaded.name)

    df = pd.DataFrame(results)

    st.subheader("📊 Таблица результатов")

    st.dataframe(df)

    st.subheader("📈 Графики")

    st.bar_chart(df["leaf_area_mm2"])
    st.bar_chart(df["root_length_mm"])

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Скачать CSV",
        csv,
        "plant_analysis.csv",
        "text/csv"
    )