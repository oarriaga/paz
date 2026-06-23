import os

os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import pytest
from keras.models import Model

from paz.models.pose_estimation.higher_hrnet import HigherHRNet


@pytest.fixture(scope="module")
def model():
    return HigherHRNet(weights=None, input_shape=(256, 256, 3))


@pytest.fixture(scope="module")
def outputs(model):
    return model(np.zeros((1, 256, 256, 3), dtype="float32"))


def test_higher_hrnet_creation(model):
    assert isinstance(model, Model)


def test_higher_hrnet_weighted_layer_count(model):
    weighted = [layer for layer in model.layers if layer.weights]
    assert len(weighted) == 604


def test_higher_hrnet_two_outputs(outputs):
    assert len(outputs) == 2


def test_higher_hrnet_heatmaps_and_tags_shape(outputs):
    assert tuple(np.asarray(outputs[0]).shape) == (1, 64, 64, 34)


def test_higher_hrnet_upsampled_heatmaps_shape(outputs):
    assert tuple(np.asarray(outputs[1]).shape) == (1, 128, 128, 17)


def test_higher_hrnet_outputs_are_finite(outputs):
    assert np.isfinite(np.asarray(outputs[0])).all()
    assert np.isfinite(np.asarray(outputs[1])).all()
