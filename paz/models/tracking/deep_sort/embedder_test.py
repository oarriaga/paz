import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import functools

import cv2
import numpy as np
import pytest

from paz.models.tracking.deep_sort.mobilenet_v2.model import MobileNetV2
from paz.models.tracking.deep_sort.mobilenet_v2 import model_test
from paz.models.tracking.deep_sort.embedder import build_embedder
from paz.models.tracking.deep_sort.embedder import crop_boxes
from paz.models.tracking.deep_sort.embedder import preprocess_crops
from paz.models.tracking.deep_sort.embedder import build_forward
from paz.models.tracking.deep_sort.embedder import compute_features

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STDV = np.array([0.229, 0.224, 0.225], np.float32)
EMPTY_BOXES = [
    [25.0, 3.0, 4.0, 4.0],
    [3.0, 21.0, 4.0, 4.0],
    [3.0, 3.0, 0.5, 4.0],
    [3.0, 3.0, 4.0, 0.9],
]


@functools.cache
def build_model():
    return model_test.randomize_weights(MobileNetV2(weights=None), 0)


@functools.cache
def build_shared_forward():
    return build_forward(build_model())


def build_image(height, width, seed):
    random = np.random.RandomState(seed)
    return random.randint(0, 256, (height, width, 3)).astype(np.uint8)


def build_images(num_images, seed):
    images = np.random.RandomState(seed).randn(num_images, 224, 224, 3)
    return images.astype(np.float32)


def to_boxes(boxes2D):
    return np.array(boxes2D, np.float32)


def preprocess_manually(crop):
    resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
    return (resized.astype(np.float32) / 255.0 - MEAN) / STDV


def test_crop_truncates_towards_zero_then_clips():
    image = build_image(20, 20, 0)
    crops = crop_boxes(image, to_boxes([[-3.7, 2.9, 10.5, 5.2]]))
    assert len(crops) == 1
    assert crops[0].dtype == np.uint8
    assert crops[0].shape == (5, 7, 3)
    assert np.array_equal(crops[0], image[2:7, 0:7])


def test_crop_clips_right_and_bottom_edges():
    image = build_image(20, 20, 1)
    crops = crop_boxes(image, to_boxes([[15.2, 17.9, 10.0, 10.0]]))
    assert np.array_equal(crops[0], image[17:20, 15:20])


def test_crops_keep_box_order():
    image = build_image(30, 40, 2)
    boxes2D = [[1.0, 2.0, 5.0, 6.0], [20.0, 10.0, 8.0, 9.0]]
    crops = crop_boxes(image, to_boxes(boxes2D))
    assert len(crops) == 2
    assert np.array_equal(crops[0], image[2:8, 1:6])
    assert np.array_equal(crops[1], image[10:19, 20:28])


@pytest.mark.parametrize("box2D", EMPTY_BOXES)
def test_empty_crop_raises(box2D):
    image = build_image(20, 20, 3)
    with pytest.raises(ValueError):
        crop_boxes(image, to_boxes([[1.0, 1.0, 5.0, 5.0], box2D]))


def test_crop_clips_negative_top():
    image = build_image(20, 20, 7)
    crops = crop_boxes(image, to_boxes([[2.0, -4.6, 5.0, 9.8]]))
    assert np.array_equal(crops[0], image[0:5, 2:7])


def test_box_left_of_frame_raises_instead_of_wrapping():
    # the oracle slices [2:6, 0:-4] here, a wrapped crop of the wrong area
    image = build_image(20, 20, 3)
    with pytest.raises(ValueError):
        crop_boxes(image, to_boxes([[-9.0, 2.0, 5.5, 4.0]]))


def test_box_above_frame_raises_instead_of_wrapping():
    # the oracle slices [0:-5, 2:7] here, a wrapped crop of the wrong area
    image = build_image(20, 20, 8)
    with pytest.raises(ValueError):
        crop_boxes(image, to_boxes([[2.0, -9.0, 5.0, 4.5]]))


def test_preprocessing_matches_manual_opencv():
    crops = [build_image(37, 53, 4), build_image(300, 260, 5)]
    images = np.asarray(preprocess_crops(crops))
    expected = np.stack([preprocess_manually(crop) for crop in crops])
    assert images.shape == (2, 224, 224, 3)
    assert images.dtype == np.float32
    assert np.allclose(images, expected, rtol=0, atol=1e-6)


def test_preprocessing_keeps_RGB_order():
    crop = np.zeros((8, 8, 3), np.uint8)
    crop[..., 0] = 255
    image = np.asarray(preprocess_crops([crop]))[0]
    expected = (np.array([1.0, 0.0, 0.0], np.float32) - MEAN) / STDV
    assert np.allclose(image[100, 100], expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("num_images", [1, 16, 17])
def test_compute_features_returns_one_row_per_image(num_images):
    images = build_images(num_images, 1)
    features = np.asarray(compute_features(build_shared_forward(), images))
    assert features.shape == (num_images, 1280)
    assert features.dtype == np.float32


def test_forward_compiles_once_for_any_number_of_images():
    forward = build_forward(build_model())
    images = build_images(17, 2)
    for num_images in (1, 16, 17):
        compute_features(forward, images[:num_images])
    assert forward._cache_size() == 1


def test_features_do_not_depend_on_batching():
    forward = build_shared_forward()
    images = build_images(17, 3)
    batched = np.asarray(compute_features(forward, images))
    model_test.assert_informative(batched)
    single = []
    for image in images:
        single.append(np.asarray(compute_features(forward, image[None]))[0])
    assert np.allclose(batched, np.stack(single), rtol=1e-5, atol=1e-5)


def raise_if_called(images, training):
    raise AssertionError("the model ran on a frame without boxes")


@pytest.mark.parametrize("boxes2D", [[], np.zeros((0, 4), np.float32)])
def test_embed_without_boxes_skips_the_model(boxes2D):
    embed = build_embedder(raise_if_called)
    features = np.asarray(embed(build_image(20, 20, 9), boxes2D))
    assert features.shape == (0, 1280)
    assert features.dtype == np.float32


def test_embedder_uses_current_model_weights():
    model = model_test.randomize_weights(MobileNetV2(weights=None), 0)
    embed = build_embedder(model)
    image_RGB = build_image(60, 80, 10)
    boxes2D = to_boxes([[-2.5, 3.9, 30.2, 40.7], [50.1, 20.0, 40.0, 50.0]])
    first = np.asarray(embed(image_RGB, boxes2D))
    model_test.randomize_weights(model, 1)
    second = np.asarray(embed(image_RGB, boxes2D))
    expected = np.asarray(build_embedder(model)(image_RGB, boxes2D))
    model_test.assert_informative(first)
    model_test.assert_informative(expected)
    assert not np.allclose(first, expected, rtol=0, atol=1e-2)
    assert np.allclose(second, expected, rtol=0, atol=1e-5)


def test_embed_matches_manual_pipeline():
    image_RGB = build_image(60, 80, 6)
    boxes2D = [[-2.5, 3.9, 30.2, 40.7], [50.1, 20.0, 40.0, 50.0]]
    features = build_embedder(build_model())(image_RGB, to_boxes(boxes2D))
    crops = [image_RGB[3:43, 0:28], image_RGB[20:60, 50:80]]
    images = np.stack([preprocess_manually(crop) for crop in crops])
    expected = np.asarray(build_model()(images))
    features = np.asarray(features)
    model_test.assert_informative(expected)
    assert features.shape == (2, 1280)
    assert features.dtype == np.float32
    assert np.allclose(features, expected, rtol=1e-4, atol=1e-5)
