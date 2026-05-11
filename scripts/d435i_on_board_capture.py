import pyrealsense2 as rs2
import cv2
import numpy as np
import os
from argparse import ArgumentParser
import time

parser = ArgumentParser()
parser.add_argument("output", help="Path to the image")
output = parser.parse_args().output
os.makedirs(output, exist_ok=True)

idx = 0
rs2_pipe = rs2.pipeline()
rs2_config = rs2.config()
rs2_pipe_wrapper = rs2.pipeline_wrapper(rs2_pipe)
rs2_pipe_profile = rs2_config.resolve(rs2_pipe_wrapper)

rs2_config.enable_stream(rs2.stream.color, 640, 480, rs2.format.bgr8, 60)

cam_profile = rs2_pipe.start(rs2_config)
frames = rs2_pipe.wait_for_frames()
time.sleep(5)
down_sample = True
idx = 0
while True:
    frames = rs2_pipe.wait_for_frames()
    down_sample = not down_sample
    if down_sample: continue
    img = np.asanyarray(frames.get_color_frame().get_data())
    cv2.imshow("frame", img)
    cv2.imwrite(os.path.join(output, f'{idx:04d}.png'), img)
    idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

