import os

os.environ.setdefault("KERAS_BACKEND", "jax")

from keras import Input, Model

from paz.models.transformers import feedforward


def test_gelu_shape_and_layer_names():
    x = Input((4, 8))
    y = feedforward.gelu(x, 16, 8, "ff_intermediate", "ff_output")
    model = Model(x, y)
    assert model.output_shape == (None, 4, 8)
    names = [layer.name for layer in model.layers]
    assert "ff_intermediate" in names and "ff_output" in names


def test_glu_shape_and_layer_names():
    x = Input((4, 8))
    y = feedforward.glu(x, 16, 8, "ff_gate", "ff_up", "ff_down")
    model = Model(x, y)
    assert model.output_shape == (None, 4, 8)
    names = [layer.name for layer in model.layers]
    assert all(n in names for n in ("ff_gate", "ff_up", "ff_down"))


def test_swiglu_matches_silu_gated_product():
    import numpy as np
    from keras import ops

    x = Input((4, 8))
    y = feedforward.swiglu(x, 16, 8, "sw_gate", "sw_up", "sw_down")
    model = Model(x, y)
    assert model.output_shape == (None, 4, 8)
    inputs = np.random.default_rng(0).normal(size=(2, 4, 8)).astype("float32")
    gate_kernel = model.get_layer("sw_gate").get_weights()[0]
    up_kernel = model.get_layer("sw_up").get_weights()[0]
    down_kernel = model.get_layer("sw_down").get_weights()[0]
    gated = np.asarray(ops.silu(inputs @ gate_kernel)) * (inputs @ up_kernel)
    expected = gated @ down_kernel
    assert np.allclose(np.asarray(model(inputs)), expected, atol=1e-5)
