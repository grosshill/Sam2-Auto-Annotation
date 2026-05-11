import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
from datetime import datetime


def record_realsense_video(duration=10, width=640, height=480, fps=24, output_dir="recordings"):
    """
    从 RealSense D435i 录制视频

    Args:
        duration: 录制时长（秒）
        width: 图像宽度
        height: 图像高度
        fps: 帧率
        output_dir: 输出目录
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成输出文件名（使用时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    color_output = os.path.join(output_dir, f"color_{timestamp}.mp4")

    print(f"=== RealSense D435i 录制器 ===")
    print(f"分辨率: {width}x{height}")
    print(f"帧率: {fps} fps")
    print(f"时长: {duration} 秒")
    print(f"输出目录: {output_dir}")
    print("-" * 40)

    # 配置 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()
    color_writer = None
    pipeline_started = False

    # 轻量化：仅启用彩色流，避免额外的深度对齐和带宽开销。
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    try:
        # 开始流
        profile = pipeline.start(config)
        pipeline_started = True
        print("相机已启动，正在预热...")

        # 获取设备信息（确认是 D435i）
        device = profile.get_device()
        print(f"设备型号: {device.get_info(rs.camera_info.name)}")
        print(f"序列号: {device.get_info(rs.camera_info.serial_number)}")

        # 等待相机稳定
        for i in range(30):
            pipeline.wait_for_frames()

        # 初始化视频写入器
        fourcc_color = cv2.VideoWriter_fourcc(*'mp4v')

        color_writer = cv2.VideoWriter(color_output, fourcc_color, fps, (width, height))

        if not color_writer.isOpened():
            print("错误：无法创建视频文件")
            return

        # 录制
        print(f"开始录制 {duration} 秒...")
        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < duration:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # 转换为 numpy 数组
            color_image = np.asanyarray(color_frame.get_data())

            # 写入视频
            color_writer.write(color_image)

            frame_count += 1

            # 显示进度（每秒钟更新一次）
            elapsed = time.time() - start_time
            if int(elapsed) > int(elapsed - 1 / fps):
                remaining = duration - elapsed
                print(f"\r进度: {elapsed:.1f}/{duration} 秒 | 帧数: {frame_count} | 剩余: {remaining:.1f} 秒", end="")

        print(f"\n录制完成！共录制 {frame_count} 帧")
        print(f"实际帧率: {frame_count / duration:.2f} fps")

    except Exception as e:
        print(f"\n错误: {e}")
        if "Couldn't resolve requests" in str(e):
            print("提示：当前相机不支持该分辨率/帧率组合，请尝试 640x480@30 或 848x480@30。")

    finally:
        # 释放资源
        if color_writer is not None:
            color_writer.release()
        if pipeline_started:
            pipeline.stop()
        print("资源已释放")
        print(f"\n输出文件:")
        print(f"  彩色视频: {color_output}")


if __name__ == "__main__":
    # 参数配置
    record_realsense_video(
        duration=10,  # 录制时长（秒）
        width=640,  # 宽度（轻量化设置）
        height=480,  # 高度（轻量化设置）
        fps=24,  # 帧率
        output_dir="recordings"  # 输出目录
    )