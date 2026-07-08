import keras
from keras.layers import EinsumDense, Identity


def apply_layer_scale(x, dim, init_values, name):
    if init_values:
        return scale(x, dim, init_values, name)
    return Identity(name=name)(x)


def scale(x, dim, init_values, name):
    initializer = keras.initializers.Constant(init_values)
    # EinsumDense "...d,d->...d" is a learnable per-channel scale on the last dim
    kwargs = dict(bias_axes=None, kernel_initializer=initializer, name=name)
    return EinsumDense("...d,d->...d", output_shape=(dim,), **kwargs)(x)
