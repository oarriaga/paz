import keras
from keras.layers import Activation, Dense, Dropout


# no layers should be passed only string of the activation function, e.g. "gelu", "relu", etc.
def mlp(
    x,
    hidden_features,
    output_features,
    use_bias,
    drop_rate,
    name,
    activation="gelu",
):
    hidden = project(x, hidden_features, use_bias, f"{name}_fc1")
    # activated = activate(hidden, activation_layer, f"{name}_act")
    activated = Activation(activation, name=f"{name}_act")(hidden)
    dropped = Dropout(drop_rate, name=f"{name}_drop1")(activated)
    projected = project(dropped, output_features, use_bias, f"{name}_fc2")
    output = Dropout(drop_rate, name=f"{name}_drop2")(projected)
    return output


def project(x, units, use_bias, name):
    init = keras.initializers.TruncatedNormal(stddev=0.02)
    kwargs = dict(use_bias=use_bias, kernel_initializer=init, name=name)
    return Dense(units, **kwargs)(x)


# no layers should be passed only string of the activation function, e.g. "gelu", "relu", etc.
# def activate(x, activation, name):
#     return Activation(activation, name=name)(x)
