"""Any-view depth, rays, and recovered cameras with Depth Anything 3.

Runs out of the box: with no arguments it downloads the pretrained DA3-SMALL
weights and two Opera House views, then prints per-view depth shapes and the
camera pose recovered jointly from the unposed images.
"""
import argparse

import numpy as np

import paz
from paz.applications import EstimateDepthAnything3Small

import demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="pretrained")
    parser.add_argument("--images", nargs="+", default=None)
    parser.add_argument("--image_size", type=int, default=518)
    args = parser.parse_args()

    estimate = EstimateDepthAnything3Small(args.weights, args.image_size)
    if args.images is None:
        names = "opera_house_0", "opera_house_1"
        images = [demo.fetch_image(name) for name in names]
    else:
        images = [paz.image.load(path) for path in args.images]
    outputs = estimate(images)
    depth, confidence, extrinsics, intrinsics, rays, ray_confidence = outputs

    print("depth", tuple(depth.shape))
    extrinsics = np.array(extrinsics)[0]
    intrinsics = np.array(intrinsics)[0]
    for view in range(len(extrinsics)):
        focal = round(float(intrinsics[view][0, 0]), 1)
        translation = extrinsics[view][:, 3].round(3).tolist()
        print(f"view {view} focal", focal, "translation", translation)
