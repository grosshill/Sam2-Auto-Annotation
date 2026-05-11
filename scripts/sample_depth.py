#!/usr/bin/env python3
import argparse
import csv
import queue
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


class AsyncFrameWriter:
    """Asynchronous writer to reduce capture stalls caused by disk I/O."""

    def __init__(self, out_dir: Path, queue_size: int = 256, save_depth_vis: bool = False):
        self.out_dir = out_dir
        self.rgb_dir = out_dir / "rgb"
        self.depth_dir = out_dir / "depth"
        self.depth_vis_dir = out_dir / "depth_vis"
        self.save_depth_vis = save_depth_vis

        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        if save_depth_vis:
            self.depth_vis_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = out_dir / "meta.csv"
        self.queue = queue.Queue(maxsize=max(8, int(queue_size)))
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, name="frame_writer", daemon=True)

        self.written = 0
        self.dropped = 0

    def start(self):
        with self.meta_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "timestamp_ms", "rgb_path", "depth_path", "depth_scale"])
        self.worker.start()

    def put(self, item):
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            # Keep latest frames: drop oldest on overflow.
            try:
                _ = self.queue.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            self.queue.put_nowait(item)

    def stop(self):
        self.stop_event.set()
        self.worker.join(timeout=5.0)

    def _worker_loop(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                frame_id, ts_ms, color_bgr, depth_u16, depth_scale = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            stem = f"{frame_id:06d}_{ts_ms}"
            rgb_name = f"{stem}.png"
            depth_name = f"{stem}.png"

            rgb_path = self.rgb_dir / rgb_name
            depth_path = self.depth_dir / depth_name

            cv2.imwrite(str(rgb_path), color_bgr)
            cv2.imwrite(str(depth_path), depth_u16)

            if self.save_depth_vis:
                depth_8u = cv2.convertScaleAbs(depth_u16, alpha=0.03)
                depth_vis = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
                cv2.imwrite(str(self.depth_vis_dir / rgb_name), depth_vis)

            with self.meta_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([frame_id, ts_ms, f"rgb/{rgb_name}", f"depth/{depth_name}", depth_scale])

            self.written += 1


def run_capture(
    output_dir: Path,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    duration: float = 0.0,
    max_frames: int = 0,
    queue_size: int = 256,
    warmup_frames: int = 30,
    preview: bool = False,
    save_depth_vis: bool = False,
):
    stop_flag = {"stop": False}

    def _handle_signal(_signum, _frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = AsyncFrameWriter(output_dir, queue_size=queue_size, save_depth_vis=save_depth_vis)
    writer.start()

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    print(f"[INFO] Output: {output_dir}")
    print(f"[INFO] Stream: {width}x{height} @ {fps} FPS")

    started = False
    try:
        profile = pipe.start(cfg)
        started = True
        dev = profile.get_device()
        print(f"[INFO] Device: {dev.get_info(rs.camera_info.name)}")
        print(f"[INFO] Serial: {dev.get_info(rs.camera_info.serial_number)}")

        depth_sensor = dev.first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"[INFO] Depth scale: {depth_scale}")

        align = rs.align(rs.stream.color)

        for _ in range(max(0, warmup_frames)):
            pipe.wait_for_frames()

        frame_id = 0
        start_t = time.time()
        t_prev = start_t
        loop_cnt = 0

        while not stop_flag["stop"]:
            if duration > 0 and (time.time() - start_t) >= duration:
                break
            if max_frames > 0 and frame_id >= max_frames:
                break

            frames = pipe.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data()).copy()
            depth_u16 = np.asanyarray(depth_frame.get_data()).copy()

            ts_ms = int(time.time() * 1000)
            writer.put((frame_id, ts_ms, color_bgr, depth_u16, depth_scale))
            frame_id += 1
            loop_cnt += 1

            now = time.time()
            if now - t_prev >= 1.0:
                real_fps = loop_cnt / (now - t_prev)
                print(
                    f"[INFO] Capture FPS: {real_fps:.2f} | captured={frame_id} | written={writer.written} | dropped={writer.dropped}",
                    flush=True,
                )
                loop_cnt = 0
                t_prev = now

            if preview:
                depth_8u = cv2.convertScaleAbs(depth_u16, alpha=0.03)
                depth_vis = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
                preview_img = np.hstack([color_bgr, depth_vis])
                cv2.imshow("D435i aligned capture (color | depth)", preview_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    finally:
        if started:
            pipe.stop()
        writer.stop()
        cv2.destroyAllWindows()

        print("[INFO] Capture stopped.")
        print(f"[INFO] Final written: {writer.written}, dropped in queue: {writer.dropped}")


def parse_args():
    parser = argparse.ArgumentParser(description="Save aligned D435i depth+RGB frames at 30 FPS.")
    parser.add_argument("--output", type=str, default="captures", help="Output root directory")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS, default 30")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration in seconds, 0 means unlimited")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to capture, 0 means unlimited")
    parser.add_argument("--queue-size", type=int, default=256, help="Writer queue size")
    parser.add_argument("--warmup-frames", type=int, default=30, help="Warmup frames before saving")
    parser.add_argument("--preview", action="store_true", help="Show real-time preview window")
    parser.add_argument("--save-depth-vis", action="store_true", help="Also save false-color depth visualization")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) / f"d435i_aligned_{timestamp}"

    run_capture(
        output_dir=out_dir,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
        max_frames=args.max_frames,
        queue_size=args.queue_size,
        warmup_frames=args.warmup_frames,
        preview=args.preview,
        save_depth_vis=args.save_depth_vis,
    )


if __name__ == "__main__":
    main()

