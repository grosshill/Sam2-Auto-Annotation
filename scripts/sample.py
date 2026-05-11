import os
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser

def main(args):
    train_img = os.listdir(f"{args.base_dir}/{args.target}/images/train")
    img_name = [_.split(".")[0] for _ in train_img]
    # print(img_name)
    os.system(f"rm -rf {args.base_dir}/{args.target}/images/val/*.png")
    os.system(f"rm -rf {args.base_dir}/{args.target}/labels/val/*.txt")

    for name in tqdm(img_name):
        if np.random.uniform() < args.percentage:
            os.system(f"cp {args.base_dir}/{args.target}/images/train/{name}.png "
                      f"{args.base_dir}/{args.target}/images/val/{name}.png")
            os.system(f"cp /{args.base_dir}/{args.target}/labels/train/{name}.txt "
                      f"{args.base_dir}/{args.target}/labels/val/{name}.txt")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", "-v", type=str, default="drone_indoor_right")
    parser.add_argument("--base_dir", "-b", type=str, default="/home/hyx/drone_pose_detect/datasets")
    parser.add_argument("--target", '-t', default="drone")
    parser.add_argument("--percentage", "-p", type=float, default=0.3)
    args = parser.parse_args()
    main(args)