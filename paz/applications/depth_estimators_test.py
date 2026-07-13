import os

import numpy as np

from paz.applications.depth_estimators import EstimateDepthAnything3Small
from paz.applications.depth_estimators import EstimateDepthAnything3MonoLarge
from paz.applications.depth_estimators import preprocess_image
from paz.models.foundation.depth_anything3.models import build_da3_small
from paz.models.foundation.depth_anything3.models import build_da3_mono_large

IMAGE_SIZE = 70


def random_image():
    return np.random.RandomState(0).randint(0, 256, (48, 64, 3), "uint8")


def test_preprocess_image_shape_and_normalization():
    processed = np.array(preprocess_image(random_image(), IMAGE_SIZE))
    assert processed.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert processed.min() < 0.0 and processed.max() > 0.0


def test_any_view_estimator_returns_six_tensors(tmp_path):
    weights = os.path.join(tmp_path, "small.weights.h5")
    build_da3_small(2, (IMAGE_SIZE, IMAGE_SIZE, 3)).save_weights(weights)
    estimate = EstimateDepthAnything3Small(weights, image_size=IMAGE_SIZE)
    outputs = estimate([random_image(), random_image()])
    assert len(outputs) == 6
    depth, confidence, extrinsics, intrinsics, rays, ray_confidence = outputs
    assert tuple(depth.shape) == (1, 2, IMAGE_SIZE, IMAGE_SIZE)
    assert tuple(extrinsics.shape) == (1, 2, 3, 4)
    assert tuple(intrinsics.shape) == (1, 2, 3, 3)


def test_mono_estimator_returns_depth_and_sky(tmp_path):
    weights = os.path.join(tmp_path, "mono.weights.h5")
    build_da3_mono_large((IMAGE_SIZE, IMAGE_SIZE, 3)).save_weights(weights)
    estimate = EstimateDepthAnything3MonoLarge(weights, image_size=IMAGE_SIZE)
    depth, sky = estimate(random_image())
    assert tuple(depth.shape) == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert tuple(sky.shape) == (1, IMAGE_SIZE, IMAGE_SIZE)
