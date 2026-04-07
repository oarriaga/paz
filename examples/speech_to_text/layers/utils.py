import keras
from keras.activations import gelu
from keras.layers import Dense


def Kernel(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


def build_gelu_dense(dim, name):
    kwargs = {"activation": gelu,
              "kernel_initializer": Kernel(),
              "bias_initializer": "zeros",
              "name": name}
    return Dense(dim, **kwargs)


def build_dense(dim, name):
    kwargs = {"kernel_initializer": Kernel(),
              "bias_initializer": "zeros",
              "name": name}
    return Dense(dim, **kwargs)
