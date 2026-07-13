import numpy as np
import jax
from keras import Input, Model

from paz.models.foundation.dinov2.blocks import build_dinov2_block


def build_block_model():
    tokens = Input((26, 384), name="tokens")
    output = build_dinov2_block(tokens, 384, 6, 4.0, 1e-5, "block_0")
    return Model(tokens, output)


def test_block_preserves_shape():
    model = build_block_model()
    output = model(np.zeros((2, 26, 384), "float32"))
    assert tuple(output.shape) == (2, 26, 384)


def test_block_jit_matches_eager():
    model = build_block_model()
    data = np.random.RandomState(0).randn(2, 26, 384).astype("float32")
    eager = np.array(model(data))
    jitted = np.array(jax.jit(lambda x: model(x))(data))
    assert np.allclose(eager, jitted, atol=1e-5)
