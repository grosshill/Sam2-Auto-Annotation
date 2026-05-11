from os import write
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from natsort import natsorted
from argparse import ArgumentParser
import cv2
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

def show_image_with_dense_grid(img, major_step=20, minor_div=4):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.imshow(img)

    # 显示坐标轴并让原点在左上（符合图像坐标习惯）
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    # 主刻度与次刻度
    ax.xaxis.set_major_locator(MultipleLocator(major_step))
    ax.yaxis.set_major_locator(MultipleLocator(major_step))
    ax.xaxis.set_minor_locator(AutoMinorLocator(minor_div))
    ax.yaxis.set_minor_locator(AutoMinorLocator(minor_div))

    # 网格：主网格粗一点，次网格细一点
    ax.grid(which="major", color="yellow", linestyle="-", linewidth=0.8, alpha=0.8)
    ax.grid(which="minor", color="white", linestyle="--", linewidth=0.4, alpha=0.5)

    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    ax.set_title("Image with Dense Grid")
    plt.tight_layout()
    plt.show()

def extract(video_dir, video_name):
    rgb_cap = cv2.VideoCapture(os.path.join(video_dir, video_name, f"{video_name}.mp4"))
    mask_cap = cv2.VideoCapture(os.path.join(video_dir, video_name, f"{video_name.split('.')[0]}_mask.mp4"))
    mask_out_dir = os.path.join(video_dir, video_name, "mask_frames")
    rgb_out_dir = os.path.join(video_dir, video_name, "rgb_frames")
    os.makedirs(mask_out_dir, exist_ok=True)
    os.makedirs(rgb_out_dir, exist_ok=True)
    frame_idx = 0
    while True:
        ok, frame = mask_cap.read()
        if not ok:
            break

        out_path = os.path.join(mask_out_dir, f"{frame_idx:04d}.png")
        frame[frame > 100] = 255
        frame[frame <= 100] = 0
        success = cv2.imwrite(str(out_path), frame)
        if not success:
            mask_cap.release()
            raise RuntimeError(f"Failed to save frame: {out_path}")
        frame_idx += 1

    mask_num = frame_idx
    print(f"{frame_idx} mask frames extracted.")

    frame_idx = 0
    frame0 = None
    while True:
        ok, frame = rgb_cap.read()
        if frame_idx == 0:
            frame0 = frame
        if not ok:
            break
        out_path = os.path.join(rgb_out_dir, f"{frame_idx:04d}.png")
        success = cv2.imwrite(str(out_path), frame)
        if not success:
            rgb_cap.release()
            raise RuntimeError(f"Failed to save frame: {out_path}")
        frame_idx += 1

    print(f"{frame_idx} rgb frames extracted.")
    show_image_with_dense_grid(
        img=frame0,
        major_step=100,
        minor_div=10,
    )

def post_process(video_dir, video_name):
    frames_dir = os.path.join(video_dir, video_name, "mask_frames")
    frames = os.listdir(frames_dir)
    labels_dir = os.path.join(video_dir, video_name, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    for frame in frames:
        if not frame.endswith('.png'):
            continue
        img = cv2.imread(os.path.join(frames_dir, frame), cv2.IMREAD_GRAYSCALE)
        img[680:, ...] = 0
        cv2.imwrite(os.path.join(frames_dir, frame), img)
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


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--video", "-v", type=str, default="drone_indoor_back")
    args = parser.parse_args()
    video_dir = "./video_data/"
    extract(video_dir, args.video)
    # post_process(video_dir, video)
    # draw_yolo_bbox_on_image(
    #     "/home/hyx/drone_pose_detect/temp/drone_indoor_test01_mask/0000.png",
    #     "/home/hyx/drone_pose_detect/temp/drone_indoor_test01_mask/labels/0000.txt"
    # )

if __name__ == "__main__":
    main()
