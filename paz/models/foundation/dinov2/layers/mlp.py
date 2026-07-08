import keras
from keras.layers import Activation, Dense, Dropout


def mlp(x, hidden_dim, out_dim, use_bias, drop_rate, name, activation="gelu"):
    x = project(x, hidden_dim, use_bias, f"{name}_fc1")
    x = Activation(activation, name=f"{name}_act")(x)
    x = Dropout(drop_rate, name=f"{name}_drop1")(x)
    x = project(x, out_dim, use_bias, f"{name}_fc2")
    x = Dropout(drop_rate, name=f"{name}_drop2")(x)
    return x


def project(x, units, use_bias, name):
    init = keras.initializers.TruncatedNormal(stddev=0.02)
    kwargs = dict(use_bias=use_bias, kernel_initializer=init, name=name)
    return Dense(units, **kwargs)(x)
