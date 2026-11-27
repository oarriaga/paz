import numpy as np
import pytest

torch = pytest.importorskip("torch")

from paz.models.foundation.dinov3.layers.layer_scale import LayerScale
from torch_layers_for_testing import PT_LayerScale


def build_reference(dimension, init_values):
    reference = PT_LayerScale(dimension, init_values=init_values)
    # The reference allocates gamma uninitialized until it is reset.
    reference.reset_parameters()
    return reference


@pytest.mark.parametrize("init_values", [1e-5, 0.1])
@pytest.mark.parametrize("shape", [(4, 128), (4, 64, 128)])
def test_output_matches_reference(init_values, shape):
    x = np.random.RandomState(0).rand(*shape).astype("float32")
    reference = build_reference(shape[-1], init_values)
    expected = reference(torch.from_numpy(x)).detach().numpy()
    output = np.array(LayerScale(shape[-1], init_values=init_values)(x))
    np.testing.assert_allclose(output, expected, rtol=1e-6)


def test_gamma_is_the_only_trainable_weight():
    weights = LayerScale(128).trainable_weights
    assert len(weights) == 1
    assert weights[0].name.startswith("gamma")
