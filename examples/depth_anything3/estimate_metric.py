"""Metric monocular depth in meters with Depth Anything 3.

Convert the Apache-2.0 DA3METRIC-LARGE checkpoint first:

    DA3_MODEL=metric python -m paz.models.foundation.depth_anything3.convert

Focal length is in pixels at the processed resolution and must be given
explicitly; it is never inferred.
"""
import argparse

import cv2
import numpy as np

from paz.applications import EstimateDepthAnything3MetricLarge

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--focal_length", type=float, required=True)
    parser.add_argument("--image_size", type=int, default=518)
    args = parser.parse_args()

    estimate = EstimateDepthAnything3MetricLarge(
        args.weights, args.focal_length, args.image_size)
    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    depth_meters, sky = estimate(image)

    depth_meters = np.array(depth_meters)[0]
    print("median depth (m):", round(float(np.median(depth_meters)), 3))
    print("range (m):", round(float(depth_meters.min()), 3),
          round(float(depth_meters.max()), 3))
