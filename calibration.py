import cv2
import numpy as np
import glob
import os


def compute_scale_from_checkerboard(image_path, square_size_mm=10):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 усиливаем контраст
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    # ⚠️ расширенный список
    possible_patterns = [
        (4, 7),  
        (7, 4),  
        (7, 7),
        (8, 8),
        (6, 6),
    ]

    for pattern_size in possible_patterns:
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        if ret and corners is not None:
            corners = corners.squeeze()

            if len(corners) < 2:
                continue

            p1 = corners[0]
            p2 = corners[1]

            pixel_dist = np.linalg.norm(p1 - p2)
            if pixel_dist == 0:
                continue

            mm_per_pixel = square_size_mm / pixel_dist
            return mm_per_pixel

    return None


def auto_compute_scale_from_folder(folder_path, square_size_mm=10):
    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))

    if not image_paths:
        print("[CALIB] Нет изображений в папке calib")
        return None

    scales = []

    print(f"[CALIB] Найдено изображений: {len(image_paths)}")

    for img_path in image_paths:
        scale = compute_scale_from_checkerboard(img_path, square_size_mm)

        if scale is not None:
            scales.append(scale)
            print(f"[CALIB] OK: {os.path.basename(img_path)} -> {scale:.5f} mm/px")
        else:
            print(f"[CALIB] FAIL: {os.path.basename(img_path)}")

    if not scales:
        print("[CALIB] Не удалось найти шахматку ни на одном изображении")
        return None

    mean_scale = float(np.mean(scales))
    print(f"[CALIB] Средний масштаб: {mean_scale:.5f} mm/px")

    return mean_scale