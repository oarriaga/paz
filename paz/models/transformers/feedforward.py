"""Position-wise feedforward sub-layers (no norm, no residual).

Flavors: ``gelu`` (vanilla Dense -> GELU -> Dense), ``glu`` (gated GeGLU),
and ``swiglu`` (SiLU-gated, Llama-style).
Callers pass layer names so weight loading keeps working, and wrap their own
normalization / residual around the returned tensor.
"""
import keras
from keras import activations
from keras.layers import Dense


def gelu(x, inner_dim, output_dim, intermediate_name, output_name):
    inner = build_dense(inner_dim, intermediate_name, activations.gelu, True)
    outer = build_dense(output_dim, output_name, None, True)
    return outer(inner(x))


def glu(x, inner_dim, output_dim, gate_name, up_name, down_name):
    gate = build_dense(inner_dim, gate_name, activations.gelu, False)
    up = build_dense(inner_dim, up_name, None, False)
    down = build_dense(output_dim, down_name, None, False)
    return down(gate(x) * up(x))


def swiglu(x, inner_dim, output_dim, gate_name, up_name, down_name):
    gate = build_dense(inner_dim, gate_name, activations.silu, False)
    up = build_dense(inner_dim, up_name, None, False)
    down = build_dense(output_dim, down_name, None, False)
    return down(gate(x) * up(x))


def build_dense(units, name, activation, use_bias):
    kernel = keras.initializers.TruncatedNormal(stddev=0.02)
    keys = ("activation", "use_bias", "kernel_initializer", "name")
    values = (activation, use_bias, kernel, name)
    kwargs = dict(zip(keys, values))
    return Dense(units, **kwargs)
