from argparse import ArgumentParser
from ultralytics import YOLO


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run prediction with a YOLO drone model.")
    parser.add_argument("--weights", default="./runs/drone/finetune/weights/best.pt", help="Trained weights")
    parser.add_argument("--source", required=True, help="Image/video/folder path")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="0", help="CUDA device id like 0, or cpu")
    parser.add_argument("--save", action="store_true", help="Save visualized predictions")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(args.weights)
    args.source = "/home/hyx/drone_pose_detect/datasets/drone/images/train/drone_indoor_back0023.png"
    args.source = "/home/hyx/drone_pose_detect/captures/d435i_aligned_20260418_170419/rgb/000039_1776503062221.png"
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=args.save,
    )

    print(f"Inference done. Frames/images processed: {len(results)}")
    if args.save and results:
        print(f"Saved to: {results[0].save_dir}")
        print(f"bbox: {results[0].boxes}")


if __name__ == "__main__":
    main()

