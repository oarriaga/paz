"""Metric monocular depth in meters with Depth Anything 3.

Convert the DA3METRIC-LARGE checkpoint first:

    DA3_MODEL=metric python -m paz.models.foundation.depth_anything3.convert

Focal length is in pixels at the processed resolution and must be given
explicitly; it is never inferred.
"""
import argparse

import numpy as np

import paz
from paz.applications import EstimateDepthAnything3MetricLarge

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="pretrained")
    parser.add_argument("--image", required=True)
    parser.add_argument("--focal_length", type=float, required=True)
    parser.add_argument("--image_size", type=int, default=518)
    args = parser.parse_args()

    settings = args.focal_length, args.weights, args.image_size
    estimate = EstimateDepthAnything3MetricLarge(*settings)
    depth_meters, sky = estimate(paz.image.load(args.image))

    depth_meters = np.array(depth_meters)[0]
    lowest = round(float(depth_meters.min()), 3)
    highest = round(float(depth_meters.max()), 3)
    print("median depth (m):", round(float(np.median(depth_meters)), 3))
    print("range (m):", lowest, highest)
