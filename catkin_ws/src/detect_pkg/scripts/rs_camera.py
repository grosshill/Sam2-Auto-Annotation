#!/usr/bin/env python3
import queue
import threading
import numpy as np
import rospy
from rospy import Publisher
import sys
import pyrealsense2 as rs2
import cv2, cv_bridge
from sensor_msgs.msg import Image, CameraInfo
from time import perf_counter as pc

class D435iCamera:
    def __init__(
            self,
            img_sz: tuple = (480, 640),
            depth: bool = False,
            color: bool = False,
            raw_fps: int = 60,
            ds_factor: int = 2,
            single_thread: bool = False,
            depth_topic: str = '/camera/depth/img_rect_raw',
            color_topic: str = '/camera/color/img_rect_raw',
            cam_info_topic: str = '/camera/camera_info',
    ):
        self.img_sz = img_sz
        self.depth = depth
        self.color = color
        self.raw_fps = raw_fps
        self.ds_factor = max(1, int(ds_factor))
        self.single_thread = bool(single_thread)

        self.depth_pub = Publisher(depth_topic, Image, queue_size=10) if depth else None
        self.color_pub = Publisher(color_topic, Image, queue_size=10) if color else None
        self.cam_info_pub = Publisher(cam_info_topic, CameraInfo, queue_size=10)
        self.bridge = cv_bridge.CvBridge()

        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=2)
        self.capture_thread = None
        self.publish_thread = None
        self.intrinsics = None

        self.rs2_pipe = rs2.pipeline()
        self.rs2_config = rs2.config()
        self.rs2_pipe_wrapper = rs2.pipeline_wrapper(self.rs2_pipe)
        self.rs2_config.resolve(self.rs2_pipe_wrapper)

        if depth:
            self.rs2_config.enable_stream(rs2.stream.depth, img_sz[1], img_sz[0], rs2.format.z16, raw_fps)
        if color:
            self.rs2_config.enable_stream(rs2.stream.color, img_sz[1], img_sz[0], rs2.format.bgr8, raw_fps)

        try:
            cam_profile = self.rs2_pipe.start(self.rs2_config)
            rospy.loginfo("RealSense pipeline started successfully")
        except Exception as e:
            rospy.logerr(f"Failed to start RealSense pipeline: {e}")
            raise

        if self.depth and self.color:
            self.align = rs2.align(align_to=rs2.stream.color)

        # Extract intrinsics once at init time
        self._init_intrinsics(cam_profile)
        self._log_active_profile(cam_profile)

        rospy.on_shutdown(self.stop)

    def _init_intrinsics(self, cam_profile):
        """Extract camera intrinsics at initialization (called once)."""
        try:
            if self.color:
                color_stream = cam_profile.get_stream(rs2.stream.color)
                self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            elif self.depth:
                depth_stream = cam_profile.get_stream(rs2.stream.depth)
                self.intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            rospy.loginfo("Camera intrinsics extracted successfully")
        except Exception as e:
            rospy.logwarn(f"Could not extract intrinsics: {e}")
            self.intrinsics = None

    def _log_active_profile(self, cam_profile):
        """Log the actual negotiated stream profile after pipeline.start()."""
        def _safe_get(profile, attr_names):
            for name in attr_names:
                if hasattr(profile, name):
                    value = getattr(profile, name)
                    try:
                        return value() if callable(value) else value
                    except Exception:
                        continue
            return None

        rospy.loginfo(f"Raw fps: {self.raw_fps}")
        rospy.loginfo(f"Downsample factor: {self.ds_factor}")
        rospy.loginfo(f"Desired fps: {self.raw_fps / self.ds_factor:.0f}")

        try:
            if self.color:
                color_profile = cam_profile.get_stream(rs2.stream.color).as_video_stream_profile()
                rospy.loginfo(
                    "Active color stream: %sx%s %s @ %s FPS",
                    _safe_get(color_profile, ["width"]),
                    _safe_get(color_profile, ["height"]),
                    _safe_get(color_profile, ["format"]),
                    _safe_get(color_profile, ["fps", "get_fps", "get_framerate"]),
                )
            if self.depth:
                depth_profile = cam_profile.get_stream(rs2.stream.depth).as_video_stream_profile()
                rospy.loginfo(
                    "Active depth stream: %sx%s %s @ %s FPS",
                    _safe_get(depth_profile, ["width"]),
                    _safe_get(depth_profile, ["height"]),
                    _safe_get(depth_profile, ["format"]),
                    _safe_get(depth_profile, ["fps", "get_fps", "get_framerate"]),
                )
        except Exception as e:
            rospy.logwarn(f"Could not log active stream profile: {e}")

    def start(self):
        if self.capture_thread and self.capture_thread.is_alive():
            return

        if self.single_thread:
            self.capture_thread = threading.Thread(
                target=self._capture_and_publish_loop,
                name="rs_capture_publish",
                daemon=True,
            )
            self.publish_thread = None
            rospy.loginfo("Single-thread mode enabled: capture + publish in one loop")
        else:
            self.capture_thread = threading.Thread(target=self._capture_loop, name="rs_capture", daemon=True)
            self.publish_thread = threading.Thread(target=self._publish_loop, name="rs_publish", daemon=True)
            rospy.loginfo("Dual-thread mode enabled: capture and publish separated")

        self.capture_thread.start()
        if self.publish_thread:
            self.publish_thread.start()
            rospy.loginfo("Capture thread and publish thread started")
        else:
            rospy.loginfo("Capture+publish thread started")

    def stop(self):
        if self.stop_event.is_set():
            return

        self.stop_event.set()

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.5)
        if self.publish_thread and self.publish_thread.is_alive():
            self.publish_thread.join(timeout=1.5)

        try:
            self.rs2_pipe.stop()
            rospy.loginfo("RealSense pipeline stopped")
        except Exception as e:
            rospy.logwarn(f"Failed to stop RealSense pipeline cleanly: {e}")

    def _push_latest(self, item):
        try:
            self.frame_queue.put_nowait(item)
        except queue.Full:
            try:
                _ = self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            self.frame_queue.put_nowait(item)

    def _capture_one(self):
        frames = self.rs2_pipe.wait_for_frames(timeout_ms=1000)
        if self.depth and self.color:
            frames = self.align.process(frames)

        color_image = None
        depth_image = None
        if self.color:
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            color_image = np.asanyarray(color_frame.get_data()).copy()
        if self.depth:
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                return None
            depth_image = np.asanyarray(depth_frame.get_data()).copy()

        return rospy.Time.now(), color_image, depth_image

    def _capture_loop(self):
        frame_idx = 0
        cnt = None
        while not self.stop_event.is_set() and not rospy.is_shutdown():
            try:
                if frame_idx % self.ds_factor != 0:
                    frame_idx += 1
                    continue
                frame_idx += 1
                if cnt is None:
                    cnt = 0
                    start = pc()
                if cnt % 100 == 0 and cnt > 0:
                    cnt = 0
                    rospy.loginfo(f"captured {100 / (pc() - start):.2f} hz")
                    start = pc()
                cnt += 1
                frame_data = self._capture_one()
                if frame_data is None:
                    continue

                self._push_latest(frame_data)
            except Exception as e:
                if not self.stop_event.is_set() and not rospy.is_shutdown():
                    rospy.logwarn(f"Capture loop error: {e}")

    def _build_cam_info(self, stamp):
        cam_info_msg = CameraInfo()
        cam_info_msg.header.stamp = stamp
        cam_info_msg.width = self.img_sz[1]
        cam_info_msg.height = self.img_sz[0]

        if self.intrinsics:
            intrinsics = self.intrinsics
            cam_info_msg.K = [intrinsics.fx, 0, intrinsics.ppx,
                              0, intrinsics.fy, intrinsics.ppy,
                              0, 0, 1]
            cam_info_msg.D = list(intrinsics.coeffs) if hasattr(intrinsics, "coeffs") else [0] * 5

        return cam_info_msg

    def _publish_loop(self):
        cnt = None
        while not self.stop_event.is_set() and not rospy.is_shutdown():
            try:
                stamp, color_image, depth_image = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if cnt is None:
                cnt = 0
                start = pc()
            if self.depth and depth_image is not None:
                depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")
                depth_msg.header.stamp = stamp
                self.depth_pub.publish(depth_msg)

            if self.color and color_image is not None:
                color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
                color_msg.header.stamp = stamp
                self.color_pub.publish(color_msg)
            if cnt % 100 == 0 and cnt > 0:
                cnt = 0
                rospy.loginfo(f"published {100 / (pc() - start):.2f} hz")
                start = pc()
            cnt += 1
            self.cam_info_pub.publish(self._build_cam_info(stamp))

    def _publish_one(self, stamp, color_image, depth_image):
        if self.depth and depth_image is not None:
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")
            depth_msg.header.stamp = stamp
            self.depth_pub.publish(depth_msg)

        if self.color and color_image is not None:
            color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
            color_msg.header.stamp = stamp
            self.color_pub.publish(color_msg)

        self.cam_info_pub.publish(self._build_cam_info(stamp))

    def _capture_and_publish_loop(self):
        frame_idx = 0
        cap_cnt = None
        pub_cnt = None
        while not self.stop_event.is_set() and not rospy.is_shutdown():
            try:
                if frame_idx % self.ds_factor != 0:
                    frame_idx += 1
                    _ = self.rs2_pipe.wait_for_frames(timeout_ms=1000)
                    continue
                frame_idx += 1

                frame_data = self._capture_one()
                if frame_data is None:
                    continue

                stamp, color_image, depth_image = frame_data

                if cap_cnt is None:
                    cap_cnt = 0
                    cap_start = pc()
                if cap_cnt % 100 == 0 and cap_cnt > 0:
                    cap_cnt = 0
                    rospy.loginfo(f"captured {100 / (pc() - cap_start):.2f} hz")
                    cap_start = pc()
                cap_cnt += 1

                self._publish_one(stamp, color_image, depth_image)

                if pub_cnt is None:
                    pub_cnt = 0
                    pub_start = pc()
                if pub_cnt % 100 == 0 and pub_cnt > 0:
                    pub_cnt = 0
                    rospy.loginfo(f"published {100 / (pc() - pub_start):.2f} hz")
                    pub_start = pc()
                pub_cnt += 1
            except Exception as e:
                if not self.stop_event.is_set() and not rospy.is_shutdown():
                    rospy.logwarn(f"Capture+publish loop error: {e}")

if __name__ == '__main__':
    rospy.init_node('rs_camera')
    rospy.loginfo("Started rs_camera node")
    single_thread = rospy.get_param("~single_thread", True)
    rs_camera = D435iCamera(
        color=True,
        depth=False,
        raw_fps=60,
        ds_factor=1,
        single_thread=True,
    )
    rs_camera.start()
    rospy.spin()