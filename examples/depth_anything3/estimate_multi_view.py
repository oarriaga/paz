"""Any-view depth, rays, and recovered cameras with Depth Anything 3.

Convert the Apache-2.0 DA3-SMALL checkpoint first:

    python -m paz.models.foundation.depth_anything3.convert
"""
import argparse

import cv2
import numpy as np

from paz.applications import EstimateDepthAnything3Small

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--image_size", type=int, default=518)
    args = parser.parse_args()

    estimate = EstimateDepthAnything3Small(args.weights, args.image_size)
    images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
              for path in args.images]
    depth, confidence, extrinsics, intrinsics, rays, ray_confidence = estimate(images)

    print("depth", tuple(depth.shape))
    for view, (extrinsic, intrinsic) in enumerate(zip(np.array(extrinsics)[0],
                                                      np.array(intrinsics)[0])):
        print(f"view {view} focal", round(float(intrinsic[0, 0]), 1),
              "translation", extrinsic[:, 3].round(3).tolist())
