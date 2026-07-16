"""Metric monocular depth in meters with Depth Anything 3.

Runs out of the box: with no arguments it downloads the pretrained
DA3METRIC-LARGE weights and an indoor demo image, then prints depth in meters.

Focal length is in pixels at the processed resolution; the model never infers
it. The default suits the demo image; pass --focal_length for your own camera.
"""
import argparse

import numpy as np

import paz
from paz.applications import EstimateDepthAnything3MetricLarge

import demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="pretrained")
    parser.add_argument("--image", default=None)
    parser.add_argument("--focal_length", type=float, default=470.0)
    parser.add_argument("--image_size", type=int, default=518)
    args = parser.parse_args()

    settings = args.focal_length, args.weights, args.image_size
    estimate = EstimateDepthAnything3MetricLarge(*settings)
    if args.image is None:
        image = demo.fetch_image("indoor")
    else:
        image = paz.image.load(args.image)
    depth_meters, sky = estimate(image)

    depth_meters = np.array(depth_meters)[0]
    lowest = round(float(depth_meters.min()), 3)
    highest = round(float(depth_meters.max()), 3)
    print("median depth (m):", round(float(np.median(depth_meters)), 3))
    print("range (m):", lowest, highest)
