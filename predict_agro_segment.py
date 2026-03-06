import os
import cv2
from pathlib import Path
from ultralytics import YOLO

from morphometry import analyze_segmentation
from calibration import auto_compute_scale_from_folder



MODEL_PATH = "runs/segment/yolo26_final/weights/best.pt"
INPUT_DIR = "test_images"      # эта папка с входными фото
OUTPUT_DIR = "results"         # папка не забудьте что фото лежит там
CALIB_DIR = "calib"
SQUARE_MM = 10
SHOW_WINDOWS = False           # если изменить на True то показывает окна если потребуется



os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[INFO] Loading model...")
model = YOLO(MODEL_PATH)

print("[INFO] Computing scale...")
scale = auto_compute_scale_from_folder(CALIB_DIR, square_size_mm=SQUARE_MM)
print("mm_per_pixel:", scale)


# ищет изображения

image_paths = []
for ext in ("*.jpg", "*.png", "*.jpeg", "*.bmp"):
    image_paths.extend(Path(INPUT_DIR).glob(ext))

print(f"[INFO] Found images: {len(image_paths)}")

if len(image_paths) == 0:
    raise ValueError("No images found in input folder")


# как обычно обрабатывает

summary = []

for img_path in image_paths:
    print(f"\n[PROCESS] {img_path.name}")

    # инференс
    result = model(str(img_path), imgsz=640)[0]

    # морфометрия px
    metrics_px = analyze_segmentation(result)

    # перевод в мм
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

    # визуализируем это все
    
    plotted = result.plot()

    from morphometry import draw_metrics_overlay

    plotted = draw_metrics_overlay(plotted, metrics_mm)

    out_path = Path(OUTPUT_DIR) / f"{img_path.stem}_result.jpg"
    cv2.imwrite(str(out_path), plotted)

    # показать если например захотите то уберите коммит и добавьте импорт CV и темпфайл
    
    # if SHOW_WINDOWS:
        # cv2.imshow("Result", plotted)
        # cv2.waitKey(500)

    # сохранить 
    
    summary.append({
        "image": img_path.name,
        **metrics_px,
        **(metrics_mm if metrics_mm else {})
    })

# закрывает окна к примеру если коммит выше убрали
#cv2.destroyAllWindows()


# принтуем сводку

print("\n=== SUMMARY ===")
for row in summary:
    print(row)


print("\n[INFO] Done.")
