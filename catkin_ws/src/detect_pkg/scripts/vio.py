#!/usr/bin/env python3
# I've given up.
# The old version pyrs2 is too hard to use since there are
# lots of hardware issues.
# Just use
# ``` bash
# roslaunch realsense2_camera rs_t265.launch
# ```
# TODO:
#   1.将local系下检测到的位姿变换到world系下进滤波器估计，然后变换到head系下输入策略推理
#   2.帧间信息？
#   3.可能使用更大的模型或更多训练。

import math as m
import pyrealsense2 as rs2
import sys
import os
import glob
import time


def _hidraw_access_ok():
    nodes = sorted(glob.glob('/dev/hidraw*'))
    if not nodes:
        return True, []
    denied = [p for p in nodes if not os.access(p, os.R_OK | os.W_OK)]
    return len(denied) == 0, denied


def _build_cfg(serial=None):
    cfg = rs2.config()
    if serial is not None:
        cfg.enable_device(serial)
    cfg.enable_stream(rs2.stream.pose)
    # T265 is often more stable when fisheye streams are requested too.
    cfg.enable_stream(rs2.stream.fisheye, 1)
    cfg.enable_stream(rs2.stream.fisheye, 2)
    return cfg


def _probe_pose(pipe, timeout_ms=1200, warmup_frames=3):
    # Some starts return successfully but need a few frames before pose appears.
    for _ in range(max(1, warmup_frames)):
        frames = pipe.wait_for_frames(timeout_ms=timeout_ms)
        pose = frames.get_pose_frame()
        if pose:
            return True
    return False


def _find_device_by_serial(ctx, serial):
    for dev in ctx.query_devices():
        try:
            if dev.get_info(rs2.camera_info.serial_number) == serial:
                return dev
        except Exception:
            continue
    return None


def _wait_device_reenumerated(serial, timeout_s=8.0, poll_s=0.2):
    end_t = time.monotonic() + timeout_s
    while time.monotonic() < end_t:
        ctx = rs2.context()
        if _find_device_by_serial(ctx, serial) is not None:
            return True
        time.sleep(poll_s)
    return False


def _hardware_reset_t265(serial):
    try:
        ctx = rs2.context()
        dev = _find_device_by_serial(ctx, serial)
        if dev is None:
            return False
        print("[VIO] Trigger hardware_reset()")
        dev.hardware_reset()
        return True
    except Exception as e:
        print(f"[WARN] hardware_reset failed: {e}")
        return False


def _start_pipeline_with_retries(ctx, serial, max_attempts=8, max_total_wait_s=25.0):
    last_err = None
    backoff_s = 0.1
    deadline = time.monotonic() + max_total_wait_s
    fail_streak = 0

    for attempt in range(1, max_attempts + 1):
        if time.monotonic() >= deadline:
            break
        # Prefer fixed-serial binding first; fallback to auto-pick.
        for use_serial in (True, False):
            if time.monotonic() >= deadline:
                break
            pipe = rs2.pipeline(ctx)
            cfg = _build_cfg(serial if use_serial else None)
            try:
                print(f"[VIO] start attempt={attempt}/{max_attempts}, use_serial={use_serial}")
                pipe.start(cfg)
                if not _probe_pose(pipe, timeout_ms=1200, warmup_frames=3):
                    raise RuntimeError("pipeline started but pose frame not ready")
                return pipe
            except Exception as e:
                last_err = e
                fail_streak += 1
                print(f"[WARN] {e}")
                try:
                    pipe.stop()
                except Exception:
                    pass

                # If failures keep repeating, emulate manual replug with hardware reset.
                if fail_streak >= 2:
                    reset_ok = _hardware_reset_t265(serial)
                    if reset_ok:
                        if not _wait_device_reenumerated(serial, timeout_s=8.0, poll_s=0.2):
                            last_err = RuntimeError("device did not re-enumerate after hardware_reset")
                        # Refresh context to avoid stale device handles after reset/re-enumeration.
                        ctx = rs2.context()
                    fail_streak = 0

        time.sleep(backoff_s)
        backoff_s = min(backoff_s * 1.5, 0.8)

    raise RuntimeError(f"pipeline start failed after retries/timeout: {last_err}")

def main():
    print(f"[VIO] Using pyrealsense2 from: {rs2.__file__}")
    print(f"[VIO] Creating context...")
    ctx = rs2.context()
    print(f"[VIO] Context created, querying devices...")
    devs = ctx.query_devices()
    print(f"[VIO] Found {len(devs)} device(s)")
    if len(devs) == 0:
        print("[ERROR] No RealSense device found in Python context", file=sys.stderr)
        print("[DEBUG] Available devices:", file=sys.stderr)
        for i in range(ctx.device_count):
            try:
                dev = ctx.get_device(i)
                print(f"  Device {i}: {dev.get_info(rs2.camera_info.name)}", file=sys.stderr)
            except Exception as e:
                print(f"  Device {i}: Error - {e}", file=sys.stderr)
        raise RuntimeError("No RealSense device found in Python context")

    # 显式选择第一个设备（T265）
    dev = devs[0]
    name = dev.get_info(rs2.camera_info.name)
    serial = dev.get_info(rs2.camera_info.serial_number)
    print(f"Using device: {name}, SN={serial}")

    try:
        pipe = _start_pipeline_with_retries(ctx, serial, max_attempts=8)
    except Exception:
        ok, denied = _hidraw_access_ok()
        if not ok:
            print("[ERROR] Failed to open T265 device interface.", file=sys.stderr)
            print("[HINT] Current user has no RW permission on hidraw nodes:", file=sys.stderr)
            for p in denied:
                print(f"  - {p}", file=sys.stderr)
            print("[HINT] Install/reload librealsense udev rules and replug T265.", file=sys.stderr)
        raise
    try:
        for _ in range(50):
            frames = pipe.wait_for_frames()
            pose = frames.get_pose_frame()
            if not pose:
                continue

            data = pose.get_pose_data()
            print(f"Frame #{pose.frame_number}")
            print(f"Position: {data.translation}")
            print(f"Velocity: {data.velocity}")
            print(f"Acceleration: {data.acceleration}")

            w = data.rotation.w
            x = -data.rotation.z
            y = data.rotation.x
            z = -data.rotation.y

            pitch = -m.asin(2.0 * (x * z - w * y)) * 180.0 / m.pi
            roll = m.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z) * 180.0 / m.pi
            yaw = m.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z) * 180.0 / m.pi
            print(f"RPY [deg]: Roll={roll:.3f}, Pitch={pitch:.3f}, Yaw={yaw:.3f}\n")
    finally:
        pipe.stop()

if __name__ == "__main__":
    main()
