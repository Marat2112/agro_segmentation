import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

model = YOLO("runs/segment/yolo26_final/weights/best.pt")

PIXEL_TO_MM = 0.5  # это коэффициент пересчета так как в реальном виде


class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(img)

        for r in results:
            for box in r.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                width_px = x2 - x1
                width_mm = width_px * PIXEL_TO_MM

                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.putText(
                    img,
                    f"{width_mm:.1f} mm",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

        return img


st.title("Live Agro Morphometry")

webrtc_streamer(
    key="agro",
    video_processor_factory=VideoProcessor

)
