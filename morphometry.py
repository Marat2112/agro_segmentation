import numpy as np
import cv2
from skimage.morphology import skeletonize




def mask_area(mask: np.ndarray) -> int:
    """Площадь маски в пикселях"""
    return int(np.sum(mask > 0))


def skeleton_length(mask: np.ndarray) -> float:
    """Длина объекта через скелетизацию"""
    if mask.sum() == 0:
        return 0.0

    binary = (mask > 0).astype(np.uint8)
    skel = skeletonize(binary).astype(np.uint8)

    # длина = количество пикселей скелета
    length = np.sum(skel > 0)
    return float(length)


def count_connected_components(mask: np.ndarray) -> int:
    """Количество отдельных объектов (например листьев)"""
    num_labels, _ = cv2.connectedComponents((mask > 0).astype(np.uint8))
    return max(0, num_labels - 1)  # вычитаем фон



# анализ


def analyze_segmentation(results):
    """
    results — вывод YOLO (results[0])
    """

    if results.masks is None:
        return {
            "root_length_px": 0,
            "root_area_px": 0,
            "stem_length_px": 0,
            "leaf_area_px": 0,
            "leaf_count": 0,
        }

    masks = results.masks.data.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    # части растений индексируем
    LEAF = 0
    ROOT = 1
    STEM = 2

    root_mask = None
    stem_mask = None
    leaf_mask = None

    # классы группируем
    for mask, cls in zip(masks, classes):
        mask = (mask > 0.5).astype(np.uint8)

        if cls == ROOT:
            root_mask = mask if root_mask is None else np.maximum(root_mask, mask)

        elif cls == STEM:
            stem_mask = mask if stem_mask is None else np.maximum(stem_mask, mask)

        elif cls == LEAF:
            leaf_mask = mask if leaf_mask is None else np.maximum(leaf_mask, mask)

    # математика 

    root_length = skeleton_length(root_mask) if root_mask is not None else 0
    root_area = mask_area(root_mask) if root_mask is not None else 0

    stem_length = skeleton_length(stem_mask) if stem_mask is not None else 0

    leaf_area = mask_area(leaf_mask) if leaf_mask is not None else 0
    leaf_count = count_connected_components(leaf_mask) if leaf_mask is not None else 0

    return {
        "root_length_px": root_length,
        "root_area_px": root_area,
        "stem_length_px": stem_length,
        "leaf_area_px": leaf_area,
        "leaf_count": leaf_count,
    }
def draw_metrics_overlay(image, metrics_mm):
    """Рисует морфометрию поверх изображения."""
    if metrics_mm is None:
        return image

    img = image.copy()

    lines = [
        f"Root length: {metrics_mm.get('root_length_mm', 0):.1f} mm",
        f"Root area: {metrics_mm.get('root_area_mm2', 0):.1f} mm2",
        f"Stem length: {metrics_mm.get('stem_length_mm', 0):.1f} mm",
        f"Leaf area: {metrics_mm.get('leaf_area_mm2', 0):.1f} mm2",
        f"Leaf count: {metrics_mm.get('leaf_count', 0)}",
    ]

    y = 30
    for line in lines:
        cv2.putText(
            img,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 30


    return img
