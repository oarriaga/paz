import os

os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import pytest
from keras.models import Model

from paz.models.keypoint.iknet import IKNet


@pytest.fixture(scope="module")
def model():
    return IKNet()


@pytest.fixture(scope="module")
def outputs(model):
    keypoints = np.zeros((1, 84, 3), dtype=np.float32)
    return model(keypoints)


def test_iknet_creation(model):
    assert isinstance(model, Model)


def test_iknet_output_shape(model):
    assert model.output_shape == (None, 21, 4)


def test_iknet_runtime_output_shape(outputs):
    quaternions = outputs
    assert tuple(np.asarray(quaternions).shape) == (1, 21, 4)


def test_iknet_output_is_normalized(outputs):
    quaternions = np.asarray(outputs)
    norms = np.linalg.norm(quaternions, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_iknet_scalar_component_is_non_negative(outputs):
    quaternions = np.asarray(outputs)
    assert (quaternions[..., -1] >= 0.0).all()
