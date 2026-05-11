import pyrealsense2 as rs2
import numpy as np
import cv2
from ultralytics import YOLO

class Detector:
    def __init__(self, weight):
        self.model = YOLO(weight, verbose=False)

    @staticmethod
    def warp(img, result, intrinsics):
        canvas = img.copy()

        fx = intrinsics.fx
        fy = intrinsics.fy
        ppx = intrinsics.ppx
        ppy = intrinsics.ppy

        # `predict` returns list[Results]; guard empty outputs.
        if not result:
            return canvas

        r0 = result[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or len(boxes) == 0:
            print(f"I didn't find any drones.")
            return canvas

        names = getattr(r0, "names", {}) or {}
        h, w = canvas.shape[:2]

        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else [None] * len(xyxy)
        cls = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else [-1] * len(xyxy)

        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            x1_i = int(max(0, min(w - 1, x1)))
            y1_i = int(max(0, min(h - 1, y1)))
            x2_i = int(max(0, min(w - 1, x2)))
            y2_i = int(max(0, min(h - 1, y2)))

            extent = max(x2_i - x1_i, y2_i - y1_i)
            z = (0.18 * (fx + fy)) / (2 * extent)
            print(z)
            cx = int((x1_i + x2_i) * 0.5)
            cy = int((y1_i + y2_i) * 0.5)

            x = (cx - ppx) / fx * z
            y = (cy - ppy) / fy * z

            x_body = z
            y_body = -x
            z_body = -y

            # Draw bbox and center pixel.
            cv2.rectangle(canvas, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
            cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1)

            cls_id = int(cls[i]) if i < len(cls) else -1
            cls_name = names.get(cls_id, str(cls_id)) if cls_id >= 0 else "obj"
            conf_val = conf[i] if i < len(conf) else None
            if conf_val is None:
                label = f"{cls_name} ({x_body:.2f}, {y_body:.2f}, {z_body:.2f})"
            else:
                label = f"{cls_name} {conf_val:.2f} ({x_body:.2f}, {y_body:.2f}, {z_body:.2f})"

            text_y = y1_i - 8 if y1_i - 8 > 10 else y1_i + 20
            cv2.putText(canvas, label, (x1_i, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        return canvas


    def detect(self, img):
        # Disable per-inference logging output from Ultralytics.
        return self.model.predict(source=img, device="0", verbose=False)

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
    
    idx = True
    try:
        while True:
            frames = rs2_pipe.wait_for_frames()
            idx = not idx
            # print(idx)
            if idx: continue
            img = np.asanyarray(frames.get_color_frame().get_data())
            result = detector.detect(img)
            warped_img = detector.warp(img, result, intrinsics)
            
            # Write frame to video if writer is enabled
            if video_writer:
                video_writer.write(warped_img)
            
            # print(np.asanyarray(frames.get_color_frame().get_data()))
            cv2.imshow("frame", warped_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release video writer resources
        if video_writer:
            video_writer.release()
            print(f"Video saved: {output_video}")
        rs2_pipe.stop()

if __name__ == "__main__":
    det = Detector("/home/hyx/drone_pose_detect/runs/detect/runs/drone/finetune_stage1/weights/best.pt")
    # Optionally specify output video path, e.g.: "output.mp4"
    activate_camera(det, output_video="detected_output.mp4")
