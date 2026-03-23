import os

os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import pytest
from keras.models import Model

from paz.models.keypoint.detnet import DetNet


@pytest.fixture(scope="module")
def model():
    return DetNet()


@pytest.fixture(scope="module")
def outputs(model):
    image = np.zeros((1, 128, 128, 3), dtype=np.uint8)
    return model(image)


def test_detnet_creation(model):
    assert isinstance(model, Model)


def test_detnet_output_shape(model):
    assert model.output_shape == [(21, 3), (21, 2)]


def test_detnet_xyz_output_shape(outputs):
    xyz, _ = outputs
    assert tuple(np.asarray(xyz).shape) == (21, 3)


def test_detnet_uv_output_shape(outputs):
    _, uv = outputs
    assert tuple(np.asarray(uv).shape) == (21, 2)


def test_detnet_xyz_values_are_finite(outputs):
    xyz, _ = outputs
    assert np.isfinite(np.asarray(xyz)).all()


def test_detnet_uv_values_stay_inside_heatmap(outputs):
    _, uv = outputs
    uv = np.asarray(uv)
    assert ((uv >= 0) & (uv < 32)).all()
