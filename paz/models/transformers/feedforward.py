"""Position-wise feedforward sub-layers (no norm, no residual).

Flavors: ``gelu`` (vanilla Dense -> GELU -> Dense) and ``glu`` (gated GeGLU).
Callers pass layer names so weight loading keeps working, and wrap their own
normalization / residual around the returned tensor.
"""
import keras
from keras import activations
from keras.layers import Dense


def gelu(x, inner_dim, output_dim, intermediate_name, output_name):
    kernel = keras.initializers.TruncatedNormal(stddev=0.02)
    hidden = Dense(inner_dim, activation=activations.gelu,
                   kernel_initializer=kernel, bias_initializer="zeros",
                   name=intermediate_name)(x)
    return Dense(output_dim, kernel_initializer=kernel,
                 bias_initializer="zeros", name=output_name)(hidden)


def glu(x, inner_dim, output_dim, gate_name, up_name, down_name):
    kernel = keras.initializers.TruncatedNormal(stddev=0.02)
    gate = Dense(inner_dim, activation=activations.gelu, use_bias=False,
                 kernel_initializer=kernel, name=gate_name)(x)
    up = Dense(inner_dim, use_bias=False, kernel_initializer=kernel,
               name=up_name)(x)
    return Dense(output_dim, use_bias=False, kernel_initializer=kernel,
                 name=down_name)(gate * up)
