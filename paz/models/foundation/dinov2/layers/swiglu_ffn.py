import keras
from keras import ops
from keras.layers import Activation, Dense, Dropout


def swiglu_ffn_fused(x, hidden_dim, out_dim, use_bias, drop_rate, name, activation):  # fmt: skip
    hidden = fused_hidden_dim(hidden_dim)
    output = compute_swiglu(x, hidden, out_dim, use_bias, activation, name)
    return Dropout(drop_rate, name=f"{name}_drop")(output)


def compute_swiglu(x, hidden_dim, out_dim, use_bias, activation, name):
    fused_name = f"{name}_fused_gate_and_value_projection"
    gate_and_value = project(x, 2 * hidden_dim, use_bias, fused_name)
    value, gate = split_value_and_gate(gate_and_value)
    value = Activation(activation, name=f"{name}_act")(value)
    output_name = f"{name}_output_projection"
    return project(value * gate, out_dim, use_bias, output_name)


def split_value_and_gate(gate_and_value):
    parts = ops.split(gate_and_value, 2, axis=-1)
    return parts[0], parts[1]


def project(x, units, use_bias, name):
    init = keras.initializers.TruncatedNormal(stddev=0.02)
    kwargs = dict(use_bias=use_bias, kernel_initializer=init, name=name)
    return Dense(units, **kwargs)(x)


def fused_hidden_dim(hidden_dim):
    return (int(hidden_dim * 2 / 3) + 7) // 8 * 8
