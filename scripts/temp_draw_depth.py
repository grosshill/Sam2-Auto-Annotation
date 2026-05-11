import numpy as np
import cv2
from gen_label import draw_bbox_xyxy

depth_img = cv2.imread("/home/hyx/drone_pose_detect/captures/d435i_aligned_20260418_170419/depth/000039_1776503062221.png", cv2.IMREAD_UNCHANGED)

depth_img[depth_img > 3000] = 0
depth_img = (depth_img / 3000 * 255).astype(np.uint8)

depth_img = np.repeat(depth_img[:, :, np.newaxis], 3, axis=2)

depth_img = draw_bbox_xyxy(depth_img, [197.9844, 149.4244, 323.0334, 204.3512])

cv2.imwrite("box_depth.png", depth_img)