import os
from argparse import ArgumentParser
from ultralytics import YOLO
from gen_label import draw_yolo_bbox_on_image
from tqdm import tqdm


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run prediction with a YOLO drone model.")
    parser.add_argument("--weights", '-w', default="/home/hyx/drone_pose_detect/runs/detect/drone/mix_012/weights/best.pt", help="Trained weights")
    parser.add_argument("--source", '-s', help="Image/video/folder path", default="/home/hyx/drone_pose_detect/video_data/camera_static/rgb_frames")
    parser.add_argument("--check", '-c', action="store_true", help="Save visualized predictions")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(args.weights, verbose=False)

    n_valid_detections = 0
    total = len(os.listdir(args.source))
    print(f'Start evaluating {total} images...')
    os.makedirs(os.path.join(args.source, "../labels"), exist_ok=True)
    for img in tqdm(os.listdir(args.source)):
        results = model.predict(
            source=os.path.join(args.source, img),
            save=False,
            verbose=False
        )
        box = results[0].boxes.xywhn
        if box.shape[0] == 1:
            n_valid_detections += 1
            box = box.cpu().numpy()
            with open(os.path.join(args.source, "../labels", img.split('.')[0] + '.txt'), 'w') as f:
                f.write(f"0 {box[0][0]} {box[0][1]} {(box[0][2])} {(box[0][3])}")

    print(f"Detected {n_valid_detections} detections, with {total} images, {total - n_valid_detections} failed.\n"
          f"Successful rate: {n_valid_detections / total:.2%}\n")

    if args.check:
        print("Saving visualized predictions...")
        draw_yolo_bbox_on_image(os.path.join(args.source, "../.."), "camera_static")
if __name__ == "__main__":
    main()

