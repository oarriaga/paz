import keras
from keras.activations import gelu
from keras.layers import Dense


def build_gelu_dense(dim, name):
    keys = ("activation", "kernel_initializer", "bias_initializer", "name")
    values = (gelu, Kernel(), "zeros", name)
    kwargs = dict(zip(keys, values))
    return Dense(dim, **kwargs)


def build_dense(dim, name):
    keys = ("kernel_initializer", "bias_initializer", "name")
    values = (Kernel(), "zeros", name)
    kwargs = dict(zip(keys, values))
    return Dense(dim, **kwargs)


def Kernel(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)
