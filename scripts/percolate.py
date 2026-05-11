import os
from numpy import random
from natsort import natsorted
from argparse import ArgumentParser

def percolate(base_dir, name, mov_to):
    remained = os.path.join(base_dir, name, "check_labels")
    label_dir = os.path.join(base_dir, name, "labels")
    rgb_dir = os.path.join(base_dir, name, "rgb_frames")
    remained_imgs = natsorted(os.listdir(remained))
    target_img_dir = os.path.join('datasets', f'{mov_to}', 'images', 'train')
    target_label_dir = os.path.join('datasets', f'{mov_to}', 'labels', 'train')
    target_img_val_dir = os.path.join('datasets', f'{mov_to}', 'images', 'val')
    target_label_val_dir = os.path.join('datasets', f'{mov_to}', 'labels', 'val')
    # print(f'rm -rf {target_img_dir + "/" + name + "*"}')
    os.system(f'rm -rf {target_img_dir + "/" + name + "*"}')
    os.system(f'rm -rf {target_label_dir + "/" + name + "*"}')
    for img in remained_imgs:
        if random.random() > 0.3:
            t_img_dir = target_img_dir
            t_label_dir = target_label_dir
        else:
            t_img_dir = target_img_val_dir
            t_label_dir = target_label_val_dir
        os.system(f"cp {os.path.join(rgb_dir, img)}"
                  f" {os.path.join(t_img_dir, name + img)}")
        os.system(f"cp {os.path.join(label_dir, img.split('.')[0] + '.txt')}"
                  f" {os.path.join(t_label_dir, name + img.split('.')[0] + '.txt')}")

    print(f'Saved {len(remained_imgs)} images.')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", "-v", type=str, default="drone_outdoor_2")
    parser.add_argument("--base_dir", "-b", type=str, default="/home/hyx/drone_pose_detect/video_data")
    parser.add_argument("--target", '-t', default="drone_outdoor_reinforce")
    args = parser.parse_args()
    percolate(args.base_dir, args.video, args.target)