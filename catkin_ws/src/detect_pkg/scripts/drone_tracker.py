#!/usr/bin/env python3
from PyCUDADetector import CUDADetector, draw_bbox_xyxy, func_timer
import numpy as np
import rospy
import cv2
import cv_bridge
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from typing import Optional
import threading
import signal
import atexit
import os
import sys
from one_euro import OneEuroFilter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

_2D_FILTER = True
HEADLESS_MODE = False
DEBUG = False

DRONE_SIZE: float = 0.18  # in meters, this is ground true
DRONE_SIZE: float = 0.23  # closer.
DRONE_SIZE: float = 0.30  # more robust but less accurate. This is used for the final test.

save_folder = "track2"
idx = 0

class DroneTracker:
    def __init__(
            self,
            drone_size: float = 0.18,
            max_contiguous_failures: int = 10,
            init_distance: float = 2,
            color_topic: str = '/camera/color/img_rect_raw',
            cam_info_topic: str = '/camera/camera_info',
            det_cfg: dict = None,
    ):
        self.debug = DEBUG
        self.drone_size = drone_size
        self.max_contiguous_failures = max_contiguous_failures
        self.init_distance = init_distance
        self.detector = CUDADetector(**det_cfg)
        self.bridge = cv_bridge.CvBridge()
        # 增加 queue_size=1 和大的 buff_size 以丢弃旧帧
        self.color_sub = rospy.Subscriber(color_topic, Image, self._color_callback, queue_size=1, buff_size=2**24)
        self.cam_info_sub = rospy.Subscriber(cam_info_topic, CameraInfo, self._cam_info_callback, queue_size=1)
        
        self.is_processing = False


        self.color = None
        self.intrinsics = None
        self.frame_id = None
        self.w = None
        self.h = None
        os.makedirs(save_folder, exist_ok=True)
        self.n_lost_cnt = 0
        self.prev_det = None
        self.prev_pose = None
        self.curr_det = None
        self.curr_pose = None

        self.target_pub = rospy.Publisher('/drone_tracker/local_target', Odometry, queue_size=10)
        self._shutdown_lock = threading.Lock()
        self._shutdown_done = False
        rospy.on_shutdown(self.shutdown_once)
        atexit.register(self.shutdown_once)

        if _2D_FILTER:
            self.detect_filter = OneEuroFilter(freq=30, min_cutoff=1.0, beta=0.1, dcutoff=1.0)
            self.filtered_color = None


    def shutdown_once(self):
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True

        rospy.loginfo("DroneTracker shutdown: releasing detector resources")
        try:
            self.detector.shutdown()
        except Exception as e:
            rospy.logerr(f"DroneTracker shutdown error: {e}")
        cv2.destroyAllWindows()
    
    @func_timer
    def _color_callback(self, color_msg):
        if self.is_processing:
            return  # 如果上一帧还在处理，直接跳过当前帧
        self.is_processing = True
        
        try:
            if self.w is not None and self.h is not None:
                extent = int(max(self.w, self.h))
                self.color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
                self.filtered_color = self.color.copy()
                rgb_color = cv2.cvtColor(self.color, cv2.COLOR_BGR2RGB)
                # Notice: The input image should be RGB instead of BGR, or the detection efficient will be extremely low.
                # rgb_color = self.color
                # cv2.imshow("color", self.color)
                # cv2.waitKey(1)
                canvas = np.zeros([1, 3, extent, extent], dtype=np.float32)
                input_color = (rgb_color.transpose((2, 0, 1)) / 255.0).astype(np.float32)[None]
                canvas[..., :self.h, :self.w] = input_color
                pose = self.get_target(self.detector.detect(canvas))
                if not HEADLESS_MODE:
                    self.color = self.filtered_color if _2D_FILTER else self.color
                    cv2.imshow("color", self.color)
                    global idx
                    idx += 1 
                    cv2.imwrite(os.path.join(save_folder, f"{idx:04d}.png"), self.color)
                    cv2.waitKey(1)
                self._publish_target(pose)
        finally:
            self.is_processing = False

    def _cam_info_callback(self, cam_info_msg):
        if self.w is None or self.h is None or self.intrinsics is None:
            self.intrinsics = np.asanyarray(cam_info_msg.K).reshape(3, 3)
            self.w = cam_info_msg.width
            self.h = cam_info_msg.height
            self.frame_id = cam_info_msg.header.frame_id
            self.reset()

    def get_target(self, det_ret) -> Optional[np.ndarray]:
        if self.n_lost_cnt >= self.max_contiguous_failures:
            self.n_lost_cnt = 0
            self.reset()

        if det_ret is None:
            self.n_lost_cnt += 1
            return None

        det_ret = np.asarray(det_ret, dtype=np.float32)
        if det_ret.size == 0:
            self.n_lost_cnt += 1
            return None

        if det_ret.ndim == 1:
            if det_ret.shape[0] < 5:
                self.n_lost_cnt += 1
                return None
            det_ret = det_ret[:5][None, :]
        elif det_ret.ndim != 2 or det_ret.shape[1] < 5:
            self.n_lost_cnt += 1
            return None

        if det_ret.shape[0] > 1:
            error = np.sum(np.abs(det_ret[:, :4] - self.prev_det[:4]), axis=1)
            det_ret = det_ret[np.argmin(error)]
        else:
            det_ret = det_ret[0]
        rospy.loginfo(det_ret)
        if self.n_lost_cnt == 0:
            self.prev_det = det_ret
        fx = self.intrinsics[0, 0]
        fy = self.intrinsics[1, 1]
        cx = self.intrinsics[0, 2]
        cy = self.intrinsics[1, 2]
        if not HEADLESS_MODE:
            self.color = draw_bbox_xyxy(self.color, det_ret[:4] / 640)
        # x1, y1, x2, y2, conf = det_ret
        # extent_x = (x2 - x1)
        # extent_y = (y2 - y1)
        # dcx = (x1 + x2) / 2
        # dcy = (y1 + y2) / 2
        if _2D_FILTER:
            det_ret[:4] = self.detect_filter.filter(det_ret[:4], rospy.Time.now().to_sec())
            if not HEADLESS_MODE:
                self.filtered_color = draw_bbox_xyxy(self.color, det_ret[:4] / 640, color=(0, 255, 0))
        # should find a robust way to calculate this
        cet, edge, ratio, area = self.get_geometry(det_ret)
        if self.is_lost(det_ret):
            rospy.logwarn('DroneTracker::get_target: Lost')
            self.n_lost_cnt += 1
            return None
        else:
            scx, scy = cet
            lx, ly = edge

            z_cam = (fx + fy) / 2 * self.drone_size / lx
            x_cam = (scx - cx) / fx * z_cam
            y_cam = (scy - cy) / fy * z_cam

            self.prev_pose = np.array([z_cam, -x_cam, -y_cam])
            self.prev_det = det_ret
            self.curr_det = det_ret.copy()
            self.curr_pose = self.prev_pose.copy()

        return self.prev_pose

    def is_lost(self, det_ret) -> bool:
        if det_ret is None or self.prev_det is None:
            return True

        prev_cet, prev_edge, prev_ratio, prev_area = self.get_geometry(self.prev_det)
        cet, edge, ratio, area = self.get_geometry(det_ret)

        eps = 1e-6
        if prev_ratio <= eps or prev_area <= eps:
            print(1)
            return True

        if np.sum(np.abs(prev_cet - cet)) > 100:
            print(cet)
            print(prev_cet)
            return True
        if np.sum(np.abs(prev_edge - edge)) > 100:
            print(3)
            return True
        if abs(prev_ratio - ratio) / max(prev_ratio, eps) > 0.5:
            print(4)
            return True
        if abs(prev_area - area) / max(prev_area, eps) > 0.5:
            print(5)
            return True

        return False

    @staticmethod
    def get_geometry(det_ret) -> (np.ndarray, np.ndarray, float, int):
        x1, y1, x2, y2, conf = np.asarray(det_ret, dtype=np.float32)[:5]
        lx = max(x2 - x1, 1e-6)
        ly = max(y2 - y1, 1e-6)
        scx = (x1 + x2) / 2
        scy = (y1 + y2) / 2

        return np.array([scx, scy]), np.array([lx, ly]), lx / ly, lx * ly
    
    def reset(self):
        self.prev_det = np.zeros(5, dtype=np.float32)
        self.prev_pose = np.zeros(3, dtype=np.float32)
        self.prev_pose[0] = self.init_distance  # in body frame
        half_box_width = self.intrinsics[0][0] * self.drone_size / self.init_distance / 2
        half_box_height = self.intrinsics[1][1] * self.drone_size / self.init_distance * self.h / self.w / 2
        self.prev_det[0] = self.w / 2 - half_box_width
        self.prev_det[2] = self.w / 2 + half_box_width
        self.prev_det[1] = self.h / 2 - half_box_height
        self.prev_det[3] = self.h / 2 + half_box_height
        self.prev_det[4] = 1.

    def _publish_target(self, pose: np.ndarray):
        if pose is None:
            pose = [3.0, 0, 0]
        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'camera'
        if self.frame_id is not None:
            msg.header.frame_id = self.frame_id
        msg.pose.pose.position.x = float(pose[0])
        msg.pose.pose.position.y = float(pose[1])
        msg.pose.pose.position.z = float(pose[2])
        rospy.loginfo(f"Publishing target pose: {pose}")
        self.target_pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('drone_tracker')
    detector_config = {
        "engine_path": "/home/drone/hyx/drone_pose_detect/stage1.engine",
        "debug": False,
        "enable_nms": True,
        "conf_thres": 0.25,
        "iou_thres": 0.45,
        "max_det": 300,
        "pre_nms_topk": 1000,
    }
    drone_tracker = DroneTracker(
        drone_size=DRONE_SIZE,
        max_contiguous_failures = 10,
        det_cfg=detector_config)

    def _signal_handler(signum, _frame):
        rospy.logwarn(f"Signal {signum} received, shutting down")
        drone_tracker.shutdown_once()
        rospy.signal_shutdown(f"signal {signum}")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        rospy.spin()
    finally:
        drone_tracker.shutdown_once()
