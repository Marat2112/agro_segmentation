import cv2
from ultralytics import YOLO

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

from morphometry import analyze_segmentation, draw_metrics_overlay
from calibration import auto_compute_scale_from_folder

TOKEN = "8620310753:AAGMXLTAvYnXBiOtfOmyxCqC-pIX7CsjW7Y"

model = YOLO("runs/segment/yolo26_final/weights/best.pt")

scale = auto_compute_scale_from_folder("calib", square_size_mm=10)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):

    photo = update.message.photo[-1]

    file = await photo.get_file()

    path = "temp.jpg"

    await file.download_to_drive(path)

    result = model(path)[0]

    metrics_px = analyze_segmentation(result)

    metrics_mm = {
        "root_length_mm": metrics_px["root_length_px"] * scale,
        "root_area_mm2": metrics_px["root_area_px"] * scale**2,
        "stem_length_mm": metrics_px["stem_length_px"] * scale,
        "leaf_area_mm2": metrics_px["leaf_area_px"] * scale**2,
        "leaf_count": metrics_px["leaf_count"]
    }

    img = result.plot()

    img = draw_metrics_overlay(img, metrics_mm)

    cv2.imwrite("result.jpg", img)

    await update.message.reply_photo(photo=open("result.jpg", "rb"))

    text = f"""
🌱 Результаты анализа

Корень длина: {metrics_mm['root_length_mm']:.2f} мм
Площадь корня: {metrics_mm['root_area_mm2']:.2f} мм²

Длина стебля: {metrics_mm['stem_length_mm']:.2f} мм

Площадь листьев: {metrics_mm['leaf_area_mm2']:.2f} мм²
Количество листьев: {metrics_mm['leaf_count']}
"""

    await update.message.reply_text(text)


app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

app.run_polling()