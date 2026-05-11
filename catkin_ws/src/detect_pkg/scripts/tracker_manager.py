#!/usr/bin/env python3
import rospy
import threading
import torch as th
from maths import Quaternion as q
from nav_msgs.msg import Odometry

max_velocity = 12.0
USE_VICON_ODOM = True

class TrackerManager:
    def __init__(self):
        self.drone_odom = None
        self.drone_local_target = None
        self._lock = threading.Lock()
        self.loop_hz = rospy.get_param("~loop_hz", 30)
        self.max_dt = rospy.get_param("~max_dt", 0.12)

        if USE_VICON_ODOM:
            self.drone_odom_sub = rospy.Subscriber("/vicon/hyx/odom", Odometry, self._drone_odom_callback)
        else:
            self.drone_odom_sub = rospy.Subscriber("/camera/odom/sample", Odometry, self._drone_odom_callback)
        self.camera_local_target_sub = rospy.Subscriber("/drone_tracker/local_target", Odometry, self._camera_local_target_callback)
        self.ekf_global_input_pub = rospy.Publisher("/ekf_node/input_global_target", Odometry, queue_size=1)

        rospy.set_param('visfly/max_velocity', max_velocity)
    
    def main(self):
        rate = rospy.Rate(self.loop_hz)
        while not rospy.is_shutdown():
            odom, local_target = self._snapshot_inputs()
            if odom is None or local_target is None:
                rate.sleep()
                continue

            if self._is_too_old_pair(odom, local_target):
                rate.sleep()
                continue

            global_target_pos = self.compute_global_target(odom, local_target)
            self.publish_global_target(global_target_pos, odom)
            rate.sleep()

    @staticmethod
    def _target_vec(odom):
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        z = odom.pose.pose.position.z

        return th.tensor([x, y, z], dtype=th.float32)

    @staticmethod
    def _drone_pose(odom):
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        z = odom.pose.pose.position.z
        qx = odom.pose.pose.orientation.x
        qy = odom.pose.pose.orientation.y
        qz = odom.pose.pose.orientation.z
        qw = odom.pose.pose.orientation.w

        orientation = q(qw, qx, qy, qz)
        position = th.tensor([x, y, z], dtype=th.float32)
        return orientation, position

    def compute_global_target(self, drone_odom, local_target_odom):
        orientation, drone_pos = self._drone_pose(drone_odom)
        local_target = self._target_vec(local_target_odom)

        # local->world: p_wt = p_wb + R_wb * p_bt
        rotated_local = orientation.rotate(local_target).reshape(-1)
        return drone_pos + rotated_local

    def publish_global_target(self, global_target_pos, drone_odom):
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = drone_odom.header.frame_id or "global"
        odom.pose.pose.position.x = float(global_target_pos[0])
        odom.pose.pose.position.y = float(global_target_pos[1])
        odom.pose.pose.position.z = float(global_target_pos[2])
        odom.pose.pose.orientation.w = 1.0
        self.ekf_global_input_pub.publish(odom)

    def _snapshot_inputs(self):
        with self._lock:
            return self.drone_odom, self.drone_local_target

    def _is_too_old_pair(self, drone_odom, local_target):
        t1 = drone_odom.header.stamp
        t2 = local_target.header.stamp
        if t1 == rospy.Time() or t2 == rospy.Time():
            return False

        dt = abs((t1 - t2).to_sec())
        if dt > self.max_dt:
            rospy.logwarn_throttle(1.0, f"tracker_manager skip unsynced pair dt={dt:.3f}s")
            return True
        return False

    def _drone_odom_callback(self, odom_msg):
        with self._lock:
            self.drone_odom = odom_msg

    def _camera_local_target_callback(self, local_target_msg):
        with self._lock:
            self.drone_local_target = local_target_msg

if __name__ == "__main__":
    rospy.init_node("tracker_manager")
    node = TrackerManager()
    node.main()
