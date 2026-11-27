import jax
import jax.numpy as jp
import pytest
import paz
from paz.models.classification.xception import MiniXception


def get_layer_types(model):
    return [type(layer).__name__ for layer in model.layers]


def test_rescale_is_applied_before_the_stem_convolution():
    plain = MiniXception((48, 48, 1), 7)
    rescaled = MiniXception((48, 48, 1), 7, preprocess="rescale")
    rescaled.set_weights(plain.get_weights())
    images = jax.random.uniform(jax.random.PRNGKey(0), (2, 48, 48, 1)) * 255
    expected = jax.jit(plain)(images / 127.5 - 1)
    assert jp.allclose(jax.jit(rescaled)(images), expected, atol=1e-5)


def test_rescale_only_adds_the_rescaling_layer():
    plain = get_layer_types(MiniXception((48, 48, 1), 7))
    rescaled = get_layer_types(MiniXception((48, 48, 1), 7, None, "rescale"))
    assert plain[:2] == ["InputLayer", "Conv2D"]
    assert rescaled == [plain[0], "Rescaling", *plain[1:]]


def test_unknown_preprocess_raises():
    with pytest.raises(ValueError, match="None or 'rescale'$"):
        MiniXception((48, 48, 1), 7, preprocess="normalize")


@pytest.mark.weights
def test_MiniXceptionFER_loads_released_weights():
    # scores recorded on a constant 0.5 image before the stem was moved
    expected = jp.array([0.13431, 0.25521, 0.32726])
    model = paz.models.MiniXceptionFER()
    scores = jax.jit(model)(jp.full((1, 48, 48, 1), 0.5))
    assert len(model.layers) == 45
    assert jp.allclose(scores[0, [0, 3, 6]], expected, atol=1e-4)


@pytest.mark.weights
def test_MiniXceptionIMDB_loads_released_weights():
    model = paz.models.MiniXceptionIMDB()
    scores = jax.jit(model)(jp.full((1, 64, 64, 1), 0.5))
    assert len(model.layers) == 47
    assert jp.allclose(scores[0], jp.array([0.65942, 0.34058]), atol=1e-4)
