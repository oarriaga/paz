"""Monocular relative depth with Depth Anything 3.

Convert the Apache-2.0 DA3MONO-LARGE checkpoint first:

    DA3_MODEL=mono python -m paz.models.foundation.depth_anything3.convert
"""
import argparse

import cv2
import numpy as np

from paz.applications import EstimateDepthAnything3MonoLarge

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--output", default="depth.png")
    args = parser.parse_args()

    estimate = EstimateDepthAnything3MonoLarge(args.weights, args.image_size)
    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    depth, sky = estimate(image)

    depth = np.array(depth)[0]
    scaled = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    cv2.imwrite(args.output, (scaled * 255).astype("uint8"))
    print("saved", args.output)
