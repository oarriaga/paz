"""Monocular relative depth with Depth Anything 3.

Runs out of the box: with no arguments it downloads the pretrained
DA3MONO-LARGE weights and a demo image, then writes a depth map. The Opera
House scene shows the sharp near/far separation the monocular model recovers.
"""
import argparse

import numpy as np

import paz
from paz.applications import EstimateDepthAnything3MonoLarge

import demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="pretrained")
    parser.add_argument("--image", default=None)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--output", default="depth.png")
    args = parser.parse_args()

    estimate = EstimateDepthAnything3MonoLarge(args.weights, args.image_size)
    if args.image is None:
        image = demo.fetch_image("opera_house_0")
    else:
        image = paz.image.load(args.image)
    depth, sky = estimate(image)

    depth = np.array(depth)[0]
    scaled = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    gray = (scaled[..., None] * np.ones(3) * 255).astype("uint8")
    paz.image.write(args.output, gray)
    print("saved", args.output)
