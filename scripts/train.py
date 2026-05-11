from argparse import ArgumentParser
from ultralytics import YOLO


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Fine-tune YOLO for drone detection.")
    parser.add_argument("--model", default="yolo26n.pt", help="Base YOLO model weights")
    parser.add_argument("--data", default="datasets/drone.yaml", help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="0", help="CUDA device id like 0, or cpu")
    parser.add_argument("--project", default="drone", help="Output project directory")
    parser.add_argument("--name", default="finetune", help="Run name")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        pretrained=True,
    )


if __name__ == "__main__":
    main()

