from ultralytics.models.sam import SAM2DynamicInteractivePredictor
import os
import cv2
import numpy as np
from natsort import natsorted
from argparse import ArgumentParser
from tqdm import tqdm
import yaml
from pprint import pprint
# Create SAM2DynamicInteractivePredictor

# Define a category by box prompt

# img_list = os.listdir("/data1/hyx/drone_pose_detect/video_data/drone_indoor_front/rgb_frames")[1:]
# base = "/data1/hyx/drone_pose_detect/video_data/drone_indoor_front/rgb_frames/"
# mask_dir = "/data1/hyx/drone_pose_detect/video_data/drone_indoor_front/mask_frames"
# os.makedirs(mask_dir, exist_ok=True)
#
#
# results = predictor(source="/data1/hyx/drone_pose_detect/video_data/drone_indoor_front/rgb_frames/0000.png",
#           bboxes=[[750, 600, 1150, 900]], obj_ids=[0], update_memory=True)
#
# if results and results[0].masks is not None:
#     m = (results[0].masks.data[0].cpu().numpy() > 0.5).astype(np.uint8) * 255
#     cv2.imwrite(os.path.join(mask_dir, "0000.png"), m)
#
# for img in img_list:
#     dir = os.path.join(base, img)
#     results = predictor(source=dir, obj_ids=[0], update_memory=False)
#     if results and results[0].masks is not None:
#         m = (results[0].masks.data[0].cpu().numpy() > 0.5).astype(np.uint8) * 255
#         cv2.imwrite(os.path.join(mask_dir, img), m)

def extract_frames(base_dir, name):
    rgb_cap = cv2.VideoCapture(os.path.join(base_dir, name, f"{name}.mp4"))
    rgb_out_dir = os.path.join(base_dir, name, "rgb_frames")
    os.makedirs(rgb_out_dir, exist_ok=True)
    frame_idx = 0
    while True:
        ok, frame = rgb_cap.read()
        if not ok:
            break
        out_path = os.path.join(rgb_out_dir, f"{frame_idx:04d}.png")
        success = cv2.imwrite(str(out_path), frame)
        if not success:
            rgb_cap.release()
            print(f"Failed to write frame {frame_idx:04d}.png")
        frame_idx += 1

    print(f"{frame_idx} rgb frames extracted.")
    print(f"Please refer to {rgb_out_dir} for results.")


def read_prompt_yaml(base_dir, name, prompt_file="prompt.yaml"):
    """Load prompt YAML for a video folder and return parsed data."""
    prompt_path = os.path.join(base_dir, name, prompt_file)
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt YAML not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Prompt YAML is empty: {prompt_path}")

    return data


def infer_masks(base_dir, name, prompt):
    prompt = read_prompt_yaml(base_dir, name, prompt)
    pprint(prompt)
    initial_box = prompt["initial_box"]
    kernel = np.ones((5, 5), dtype=np.uint8)
    overrides = dict(conf=0.01, task="segment", mode="predict", imgsz=1024, model="sam2.1_b.pt", save=False)
    predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=10)
    frame_list = natsorted(os.listdir(os.path.join(base_dir, name, "rgb_frames")))
    out_dir = os.path.join(base_dir, name, "mask_frames")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Inferring the first frame")
    print(os.path.join(base_dir, name, "rgb_frames", frame_list[int(prompt["start_frame"])]))
    # results = predictor(source=os.path.join(base_dir, name, "rgb_frames", frame_list[int(prompt["start_frame"])]),
    results = predictor(source=os.path.join(base_dir, name, "rgb_frames", frame_list[0]),
                        bboxes=[initial_box], obj_ids=[0], update_memory=True)

    if results and results[0].masks is not None:
        m = (results[0].masks.data[0].cpu().numpy() > 0.5).astype(np.uint8) * 255
        cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(os.path.join(out_dir, frame_list[0]), m)

    print("First frame inferred, memory updated.")

    print(f"Inferring {len(frame_list) - 1 - int(prompt['start_frame'])} frames")
    for idx, frame in enumerate(tqdm(frame_list[int(prompt["start_frame"]) + 1:])):
        if prompt.get("auxiliary") is not None and idx + 1 in prompt["auxiliary"]:
            print(f"Adding auxiliary prompts at frame {idx+1}.")
            if prompt["auxiliary"][idx + 1].get("box") is not None:
                results = predictor(
                    source=os.path.join(base_dir, name, "rgb_frames", frame),
                    bboxes=[prompt["auxiliary"][idx + 1]["box"]],
                    obj_ids=[int(prompt["auxiliary"][idx + 1]["object_id"])],
                    update_memory=True,
                )
            if prompt["auxiliary"][idx + 1].get("point") is not None:
                results = predictor(
                    source=os.path.join(base_dir, name, "rgb_frames", frame),
                    points=[point[:2] for point in prompt["auxiliary"][idx + 1]["point"]],
                    labels=[point[2] for point in prompt["auxiliary"][idx + 1]["point"]],
                    obj_ids=[int(prompt["auxiliary"][idx + 1]["object_id"])],
                    update_memory=True,
                )
        else:
            results = predictor(source=os.path.join(base_dir, name, "rgb_frames", frame),
                            obj_ids=[0], update_memory=False)
        if results and results[0].masks is not None and results[0].masks.data.shape[0] > 0:
            m = (results[0].masks.data[0].cpu().numpy() > 0.5).astype(np.uint8) * 255
            cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
            cv2.imwrite(os.path.join(out_dir, frame), m)
        else:
            print(f"No detections at frame {idx + 1}.")

    print(f"Done.")
    print(f"Please refer to {out_dir} for results.")

if __name__ == "__main__":
    # front [750, 600, 1150, 900]
    # left [1100, 490, 1400, 630]
    # back [780, 370, 1100, 550]
    parser = ArgumentParser()
    parser.add_argument("--video", "-v",type=str, default="drone_indoor_back")
    parser.add_argument("--base_dir", "-b", type=str, default="/data1/hyx/drone_pose_detect/video_data")
    parser.add_argument("--prompt", "-p", type=str, default="prompt.yaml")
    parser.add_argument("--extract", "-e", action="store_true")
    parser.add_argument("--inference", "-i", action="store_true")
    args = parser.parse_args()
    data_base_dir = args.base_dir
    video_name = args.video

    if args.extract:
        extract_frames(data_base_dir, video_name)
    if args.inference:
        infer_masks(data_base_dir, video_name, args.prompt)

    if not args.extract and not args.inference:
        print(f'Please specify either --extract or --inference.')
    # if args.annotate:
    #     annotate(data_base_dir, video_name)
    # if args.percolate:
    #     percolate(data_base_dir, video_name)