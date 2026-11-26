import os

import numpy as np
import pytest
import paz
from paz.applications import human_pose_estimators as estimators

MODELS = os.path.expanduser("~/.keras/paz/models")
WEIGHTS = os.path.join(MODELS, "higher_hrnet_paz_jax.weights.h5")
VOC = "~/.keras/paz/datasets/VOC2007-test/VOCdevkit/VOC2007/JPEGImages"
PORTRAIT_IMAGE = os.path.join(os.path.expanduser(VOC), "000368.jpg")


def transform_corners(H, W):
    image = np.zeros((H, W, 3), dtype=np.uint8)
    center, scale, size = estimators.compute_transform_params(image)
    transform = estimators.build_affine(center, scale, size, inverse=False)
    corners = np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]])
    return corners @ transform.T, size


@pytest.mark.parametrize("H, W", [(500, 333), (333, 500), (400, 400)])
def test_transform_fits_image_to_network_input(H, W):
    corners, size = transform_corners(H, W)
    zoom = 512 / min(H, W)
    assert np.all(corners > -zoom) and np.all(corners < size + zoom)
    span = corners.max(axis=0) - corners.min(axis=0)
    assert np.allclose(span, [W * zoom, H * zoom], atol=1e-3)


def test_portrait_scale_is_width_then_height():
    image = np.zeros((500, 333, 3), dtype=np.uint8)
    scale = estimators.compute_transform_scale(image, [512, 832], 200)
    assert np.allclose(scale, [333 / 200, 832 / 512 * 333 / 200])


def test_landscape_scale_is_width_then_height():
    image = np.zeros((333, 500, 3), dtype=np.uint8)
    scale = estimators.compute_transform_scale(image, [832, 512], 200)
    assert np.allclose(scale, [832 / 512 * 333 / 200, 333 / 200])


def test_portrait_keypoints_map_back_to_image():
    image = np.zeros((500, 333, 3), dtype=np.uint8)
    center, scale, size = estimators.compute_transform_params(image)
    transform = estimators.build_affine(center, scale, size, inverse=False)
    points = np.array([[10.0, 20.0, 1.0], [300.0, 480.0, 1.0]])
    people = [np.concatenate([points @ transform.T, np.ones((2, 1))], axis=1)]
    restored = estimators.transform_to_image(people, center, scale, size)[0]
    assert np.allclose(restored[:, :2], points[:, :2], atol=1e-2)


@pytest.mark.skipif(not os.path.exists(WEIGHTS), reason="weights not cached")
@pytest.mark.skipif(not os.path.exists(PORTRAIT_IMAGE), reason="no image")
def test_portrait_image_detects_person_inside_image():
    image = np.array(paz.image.load(PORTRAIT_IMAGE))
    estimate = paz.applications.HigherHRNetHumanPose2D(draw=False)
    keypoints, scores = estimate(image)
    person = keypoints[int(np.argmax(scores))]
    H, W = image.shape[:2]
    assert len(keypoints) < 5
    assert np.all((person >= 0) & (person < [W, H]))
