import keras
from keras import ops
from keras.layers import Activation, Dense, Dropout


def swiglu_ffn_fused(
    x,
    hidden_dim,
    out_dim,
    use_bias,
    drop_rate,
    name,
    activation,
):
    hidden_units = compute_fused_hidden(hidden_dim)
    args = (x, hidden_units, out_dim, use_bias, activation, name)
    projected = compute_swiglu(*args)
    output = Dropout(drop_rate, name=f"{name}_drop")(projected)
    return output


def compute_swiglu(x, hidden, output_units, use_bias, activation, name):
    fused_name = f"{name}_fused_gate_and_value_projection"
    fused = project(x, 2 * hidden, use_bias, fused_name)
    value, gate = split_value_and_gate(fused)
    activated = activate(value, activation, f"{name}_act")
    multiplied = activated * gate
    output_name = f"{name}_output_projection"
    output = project(multiplied, output_units, use_bias, output_name)
    return output


def split_value_and_gate(fused):
    parts = ops.split(fused, 2, axis=-1)
    return parts[0], parts[1]


def project(x, units, use_bias, name):
    init = keras.initializers.TruncatedNormal(stddev=0.02)
    kwargs = dict(use_bias=use_bias, kernel_initializer=init, name=name)
    return Dense(units, **kwargs)(x)


def activate(x, activation, name):
    return Activation(activation, name=name)(x)


def compute_fused_hidden(hidden):
    return (int(hidden * 2 / 3) + 7) // 8 * 8

