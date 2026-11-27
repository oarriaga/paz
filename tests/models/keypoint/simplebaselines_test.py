import numpy as np
import pytest

from paz.models.keypoint.simplebaselines import SimpleBaseline


@pytest.fixture(scope="module")
def model():
    return SimpleBaseline(weights=None)


def test_simplebaseline_output_shape(model):
    assert model.output_shape == (None, 16, 3)


def test_simplebaseline_runtime_output(model):
    outputs = np.asarray(model(np.zeros((4, 32), dtype="float32")))
    assert outputs.shape == (4, 16, 3)
    assert np.isfinite(outputs).all()


def test_simplebaseline_invalid_weights():
    with pytest.raises(ValueError):
        SimpleBaseline(weights="invalid")
