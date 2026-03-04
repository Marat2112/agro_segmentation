import argparse
import torch
from ultralytics import YOLO


def get_device():
    """Автовыбор устройства."""
    if torch.cuda.is_available():
        print("[INFO] Using GPU:", torch.cuda.get_device_name(0))
        return 0
    else:
        print("[INFO] Using CPU")
        return "cpu"


def main(args):
    device = get_device()

    print("[INFO] Loading model:", args.model)
    model = YOLO(args.model)

    print("[INFO] Starting training...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        optimizer="AdamW",
        lr0=0.003,
        cos_lr=True,
        degrees=5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        project="runs/segment",
        name="yolo26_final",
        exist_ok=True,
    )

    print("[INFO] Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO26 segmentation")

    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n-seg.pt",
        help="Path to model weights",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=160,
        help="Number of epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Dataloader workers",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=40,
        help="Early stopping patience",
    )

    args = parser.parse_args()
    main(args)