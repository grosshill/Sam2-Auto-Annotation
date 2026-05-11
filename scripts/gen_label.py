import os
import cv2
from tqdm import tqdm
import numpy as np
from natsort import natsorted
from argparse import ArgumentParser

def draw_yolo_bbox_on_image(video_dir, video_name, color=(0, 0, 255), thickness=5, class_names="drone"):
    """Draw YOLO bbox labels on an image.

    YOLO label format per line: class_id x_center y_center width height (normalized).
    """
    rgb_img_dir = os.path.join(video_dir, video_name, "rgb_frames")
    mask_dir = os.path.join(video_dir, video_name, "mask_frames")
    label_dir = os.path.join(video_dir, video_name, "labels")
    rgb_imgs = os.listdir(rgb_img_dir)
    save_path = os.path.join(video_dir, video_name, "check_labels")
    os.makedirs(save_path, exist_ok=True)
    for image_name in tqdm(rgb_imgs):
        image_path = os.path.join(rgb_img_dir, image_name)
        label_path = os.path.join(label_dir, image_name.split(".")[0] + ".txt")
        mask_path = os.path.join(mask_dir, image_name.split(".")[0] + ".png")
        image = cv2.imread(image_path)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            alpha = 0.35  # 蓝色强度
            blue = np.array([255, 0, 0], dtype=np.float32)  # BGR 的蓝色
            image = image.astype(np.float32)
            image[mask > 0] = image[mask > 0] * (1 - alpha) + blue * alpha
        if image is None or not os.path.exists(label_path):
            continue
        image = image.astype(np.uint8)

        img_h, img_w = image.shape[:2]
        box_count = 0

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])

                x1 = int((cx - bw / 2.0) * img_w)
                y1 = int((cy - bh / 2.0) * img_h)
                x2 = int((cx + bw / 2.0) * img_w)
                y2 = int((cy + bh / 2.0) * img_h)

                x1 = max(0, min(img_w - 1, x1))
                y1 = max(0, min(img_h - 1, y1))
                x2 = max(0, min(img_w - 1, x2))
                y2 = max(0, min(img_h - 1, y2))

                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                cls_name = class_names.get(class_id, str(class_id)) if isinstance(class_names, dict) else str(class_id)
                cv2.putText(
                    image,
                    cls_name,
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    max(1, thickness - 1),
                    cv2.LINE_AA,
                )
                box_count += 1

        cv2.imwrite(os.path.join(save_path, image_name), image)

def match_up(video_dir, video_name):
    mask_frames = natsorted(os.listdir(os.path.join(video_dir, video_name, "mask_frames")))
    rgb_frames = natsorted(os.listdir(os.path.join(video_dir, video_name, "rgb_frames")))
    remained = set(mask_frames) & set(rgb_frames)
    mask_rm = list(set(mask_frames) - remained)
    rgb_rm = list(set(rgb_frames) - remained)
    for mask in mask_rm:
        os.remove(os.path.join(video_dir, video_name, "mask_frames", mask))
    for rgb in rgb_rm:
        os.remove(os.path.join(video_dir, video_name, "rgb_frames", rgb))

    for idx, name in enumerate(natsorted(remained)):
        os.rename(os.path.join(video_dir, video_name, "mask_frames", name),
                  os.path.join(video_dir, video_name, "mask_frames", f"{idx:04d}.png"))
        os.rename(os.path.join(video_dir, video_name, "rgb_frames", name),
                  os.path.join(video_dir, video_name, "rgb_frames", f"{idx:04d}.png"))

    print(f"Match {len(remained)} frames, remove others.")

def post_process(video_dir, video_name):
    frames_dir = os.path.join(video_dir, video_name, "mask_frames")
    frames = os.listdir(frames_dir)
    labels_dir = os.path.join(video_dir, video_name, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    for frame in frames:
        if not frame.endswith('.png'):
            continue
        img = cv2.imread(os.path.join(frames_dir, frame), cv2.IMREAD_GRAYSCALE)
        # img[680:, ...] = 0
        # cv2.imwrite(os.path.join(frames_dir, frame), img)
        h, w = img.shape
        target_mat_row, target_mat_col = np.where(img == 255)
        max_y = np.max(target_mat_row)
        max_x = np.max(target_mat_col)
        min_y = np.min(target_mat_row)
        min_x = np.min(target_mat_col)

        cet_x, cet_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        box_w, box_h = max_x - min_x, max_y - min_y
        cet_x /= w
        box_w /= w
        cet_y /= h
        box_h /= h
        with open(os.path.join(labels_dir, f"{frame.split('.')[0]}.txt"), 'w') as f:
            f.write(f"0 {cet_x} {cet_y} {box_w} {box_h}")

    print(f"{len(frames)} frames post-processed and labels saved.")

def draw_bbox_xyxy(image, boxes_xyxy, color=(0, 0, 255), thickness=2, category="drone"):
    """Simplified drawer: input np image + xyxy boxes, return rendered image."""
    if image is None:
        raise ValueError("image is None")

    out = image.copy()
    h, w = out.shape[:2]
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)

    # Support one label for all boxes, or one label per box.
    if isinstance(category, (list, tuple, np.ndarray)):
        labels = [str(x) for x in category]
    else:
        labels = [str(category)] * len(boxes)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = np.asarray(box, dtype=np.int32)
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        if i < len(labels) and labels[i] != "":
            cv2.putText(
                out,
                labels[i],
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                max(1, thickness - 1),
                cv2.LINE_AA,
            )

    return out

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", "-v", type=str, default="drone_indoor_left")
    parser.add_argument("--base_dir", "-b", type=str, default="/home/hyx/drone_pose_detect/video_data")
    args = parser.parse_args()
    video = args.video
    video_dir = args.base_dir
    post_process(video_dir, video)
    draw_yolo_bbox_on_image(video_dir, video)
