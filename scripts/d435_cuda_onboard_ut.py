import pyrealsense2 as rs2
import cv2
import numpy as np
from PyCUDADetector import CUDADetector


def draw_bbox_xyxy(image, boxes_xyxy, color=(0, 0, 255), thickness=2, category="drone"):
    """Simplified drawer: input np image + xyxy boxes, return rendered image."""
    """
        o -- x
        |
        y
    """
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
        x1, y1, x2, y2 = (box[:] * max(h, w)).astype(np.int32)
        # x1, x2 = (box[0] * w).astype(np.int32), (box[2] * w).astype(np.int32)
        # y1, y2 = (box[0] * h).astype(np.int32), (box[2] * h).astype(np.int32)
        # print(w, h)
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
def activate_camera(detector, output_video=None):
    rs2_pipe = rs2.pipeline()
    rs2_config = rs2.config()
    rs2_pipe_wrapper = rs2.pipeline_wrapper(rs2_pipe)
    rs2_pipe_profile = rs2_config.resolve(rs2_pipe_wrapper)
    # rs2_dev = rs2_pipe_profile.get_device()
    # rs2_config.enable_stream(rs2.stream.depth, 640, 480, rs2.format.z16, 30)
    rs2_config.enable_stream(rs2.stream.color, 640, 480, rs2.format.bgr8, 60)

    cam_profile = rs2_pipe.start(rs2_config)
    # rs2_depth_scale = cam_profile.get_device().first_depth_sensor().get_depth_scale()

    frames = rs2_pipe.wait_for_frames()
    intrinsics = frames.get_profile().as_video_stream_profile().get_intrinsics()

    print(intrinsics)
    print(type(intrinsics))
    print(intrinsics.fx)
    print(intrinsics.fy)
    print(intrinsics.ppx)
    print(intrinsics.ppx)

    # Initialize video writer if output path provided
    video_writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video, fourcc, 30.0, (640, 480))
        print(f"Video writer initialized: {output_video}")
    from time import perf_counter as pc
    idx = True
    cnt = 0
    start = pc()
    try:
        while True:
            frames = rs2_pipe.wait_for_frames()
            idx = not idx
            # print(idx)
            # if idx: continue
            if cnt % 100 == 0:
                cnt = 0
                print(f"Current fps: {100 / (pc() - start):2f} Hz")
                start = pc()

            cnt += 1
            img = np.asanyarray(frames.get_color_frame().get_data())
            # print(img.shape)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_img = (rgb_img / 255.0).astype(np.float32)
            input_img = input_img.transpose(2, 0, 1)[None]
            canvas = np.zeros([1, 3, 640, 640], dtype=np.float32)
            canvas[:, :, :480, :640] = input_img
            result = detector.detect(canvas)
            # print(result)
            # print(result.shape)
            # if result.shape[0] == 0:
            #     print("No objects detected")
            if result.shape[0] > 0:
                result = result[result[:, 4] == np.max(result[:, 4])].squeeze()
            bbox = result[:4] / 640

            # print(f"bbox: {bbox}")
            warped_img = draw_bbox_xyxy(img, bbox)

            # warped_img = detector.warp(img, result, intrinsics)

            # Write frame to video if writer is enabled
            if video_writer:
                video_writer.write(warped_img)

            # print(np.asanyarray(frames.get_color_frame().get_data()))
            # cv2.imshow("frame", warped_img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    finally:
        # Release video writer resources
        if video_writer:
            video_writer.release()
            print(f"Video saved: {output_video}")
        rs2_pipe.stop()

if __name__ == "__main__":
    det = CUDADetector(
        "./stage1.engine",
        debug=False,
        enable_nms=True,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=300,
        pre_nms_topk=1000,
    )
    # Optionally specify output video path, e.g.: "output.mp4"
    # activate_camera(det, output_video="detected_output.mp4")
    activate_camera(det, output_video=None)
