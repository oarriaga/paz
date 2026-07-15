"""Any-view depth, rays, and recovered cameras with Depth Anything 3.

Convert the Apache-2.0 DA3-SMALL checkpoint first:

    python -m paz.models.foundation.depth_anything3.convert
"""
import argparse

import numpy as np

import paz
from paz.applications import EstimateDepthAnything3Small

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--image_size", type=int, default=518)
    args = parser.parse_args()

    estimate = EstimateDepthAnything3Small(args.weights, args.image_size)
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
