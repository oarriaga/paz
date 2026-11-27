import os

os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import pytest
from keras.models import Model

from paz.models.keypoint.simplebaselines import SimpleBaseline


@pytest.fixture(scope="module")
def model():
    return SimpleBaseline(weights=None)


def test_simplebaseline_creation(model):
    assert isinstance(model, Model)


def test_simplebaseline_output_shape(model):
    assert model.output_shape == (None, 16, 3)


def test_simplebaseline_weighted_layer_count(model):
    weighted = [layer for layer in model.layers if layer.weights]
    assert len(weighted) == 11


def test_simplebaseline_runtime_output(model):
    outputs = np.asarray(model(np.zeros((4, 32), dtype="float32")))
    assert outputs.shape == (4, 16, 3)
    assert np.isfinite(outputs).all()


def test_simplebaseline_invalid_weights():
    with pytest.raises(ValueError):
        SimpleBaseline(weights="invalid")
