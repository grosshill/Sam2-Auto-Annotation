from argparse import ArgumentParser
from ultralytics import YOLO


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Validate a fine-tuned YOLO drone model.")
    parser.add_argument("--weights", default="./runs/drone/finetune/weights/best.pt", help="Trained weights")
    parser.add_argument("--data", default="./datasets/drone.yaml", help="Dataset YAML path")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="0", help="CUDA device id like 0, or cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device)

    print("Validation done.")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()

