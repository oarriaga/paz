"""Monocular relative depth with Depth Anything 3.

Convert the DA3MONO-LARGE checkpoint first:

    DA3_MODEL=mono python -m paz.models.foundation.depth_anything3.convert
"""
import argparse

import numpy as np

import paz
from paz.applications import EstimateDepthAnything3MonoLarge

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--output", default="depth.png")
    args = parser.parse_args()

    estimate = EstimateDepthAnything3MonoLarge(args.weights, args.image_size)
    depth, sky = estimate(paz.image.load(args.image))

    depth = np.array(depth)[0]
    scaled = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    gray = (scaled[..., None] * np.ones(3) * 255).astype("uint8")
    paz.image.write(args.output, gray)
    print("saved", args.output)
