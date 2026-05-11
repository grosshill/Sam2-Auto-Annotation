import os
import cv2
import numpy as np
from natsort import natsorted

## 0245
class KCF:
    def __init__(self):
        self.kcf = cv2.TrackerKCF_create()
        # self.kcf.init()

    def update(self, frame, bbox: np.ndarray):
        pass

class IsCrossHillStillInList:
    def __call__(self):
        return self

if __name__ == "__main__":
    # KCF = KCF()
    tracker = cv2.TrackerKCF_create()
    img_folder ="/home/hyx/drone_pose_detect/video_data/camera_move/rgb_frames"
    img_list = natsorted(os.listdir(img_folder))[244:]

    bbox = cv2.selectROI("image", cv2.imread(os.path.join(img_folder, img_list[0])), False)
    print(bbox)
    tracker.init(cv2.imread(os.path.join(img_folder, img_list[0])), bbox)

    for img_path in img_list:
        frame = cv2.imread(os.path.join(img_folder, img_path))
        suc, bbox = tracker.update(frame)
        if suc:
            # 跟踪成功
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # 跟踪失败（例如目标被完全遮挡或移出画面）
            cv2.putText(frame, "Tracking failed", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # 显示结果
        cv2.imshow('KCF Tracker', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 8. 释放资源
cv2.destroyAllWindows()
