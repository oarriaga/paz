import numpy as np
from keras import Input, Model, ops
from scipy.special import erf

from paz.models.transformers import feedforward


def build_inputs():
    inputs = np.random.default_rng(0).normal(size=(2, 4, 8))
    return inputs.astype("float32")


def compute_exact_gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def randomize_weights(model):
    random = np.random.default_rng(1)
    weights = []
    for weight in model.get_weights():
        weights.append(random.normal(size=weight.shape).astype("float32"))
    model.set_weights(weights)


def test_gelu_matches_dense_gelu_dense():
    x = Input((4, 8))
    y = feedforward.gelu(x, 16, 8, "ff_intermediate", "ff_output")
    model = Model(x, y)
    randomize_weights(model)
    inputs = build_inputs()
    kernel, bias = model.get_layer("ff_intermediate").get_weights()
    hidden = compute_exact_gelu(inputs @ kernel + bias)
    kernel, bias = model.get_layer("ff_output").get_weights()
    expected = hidden @ kernel + bias
    assert np.allclose(np.asarray(model(inputs)), expected, atol=1e-4)


def test_glu_matches_gelu_gated_product():
    x = Input((4, 8))
    y = feedforward.glu(x, 16, 8, "ff_gate", "ff_up", "ff_down")
    model = Model(x, y)
    randomize_weights(model)
    inputs = build_inputs()
    gate_kernel = model.get_layer("ff_gate").get_weights()[0]
    up_kernel = model.get_layer("ff_up").get_weights()[0]
    down_kernel = model.get_layer("ff_down").get_weights()[0]
    gated = compute_exact_gelu(inputs @ gate_kernel) * (inputs @ up_kernel)
    expected = gated @ down_kernel
    assert np.allclose(np.asarray(model(inputs)), expected, atol=1e-4)


def test_swiglu_matches_silu_gated_product():
    x = Input((4, 8))
    y = feedforward.swiglu(x, 16, 8, "sw_gate", "sw_up", "sw_down")
    model = Model(x, y)
    inputs = build_inputs()
    gate_kernel = model.get_layer("sw_gate").get_weights()[0]
    up_kernel = model.get_layer("sw_up").get_weights()[0]
    down_kernel = model.get_layer("sw_down").get_weights()[0]
    gated = np.asarray(ops.silu(inputs @ gate_kernel)) * (inputs @ up_kernel)
    expected = gated @ down_kernel
    assert np.allclose(np.asarray(model(inputs)), expected, atol=1e-5)
