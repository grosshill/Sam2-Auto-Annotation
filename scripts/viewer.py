from extract_mask import show_image_with_dense_grid
import cv2

img = cv2.imread("/home/hyx/drone_pose_detect/video_data/camera_move/rgb_frames/0244.png")
show_image_with_dense_grid(
    img,
    100,
    10
)