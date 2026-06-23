import os

os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import pytest
from keras.models import Model

from paz.models.pose_estimation.efficientpose import EfficientPosePhi0


@pytest.fixture(scope="module")
def model():
    return EfficientPosePhi0(base_weights=None, head_weights=None)


def test_efficientpose_creation(model):
    assert isinstance(model, Model)


def test_efficientpose_has_anchors(model):
    assert np.asarray(model.prior_boxes).shape == (49104, 4)
    assert np.asarray(model.translation_priors).shape == (49104, 3)


def test_efficientpose_outputs(model):
    detections, transformation = model(np.zeros((1, 512, 512, 3), "float32"))
    assert tuple(np.asarray(detections).shape) == (1, 49104, 12)
    assert tuple(np.asarray(transformation).shape) == (1, 49104, 6)


def test_efficientpose_invalid_weights():
    with pytest.raises(ValueError):
        EfficientPosePhi0(base_weights="invalid", head_weights=None)
