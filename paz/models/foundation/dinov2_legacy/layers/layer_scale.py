import keras
from keras.layers import EinsumDense, Identity


def apply_layer_scale(x, dim, init_values, name):
    if init_values:
        return scale(x, dim, init_values, name)
    return Identity(name=name)(x)


def scale(x, dim, init_values, name):
    initializer = keras.initializers.Constant(init_values)
    # The EinsumDense layer is used here to perform element-wise scaling and "...d,d->...d" applies a learnable per-channel scale to the last dim.
    return EinsumDense(
        equation="...d,d->...d",
        output_shape=(dim,),
        bias_axes=None,
        kernel_initializer=initializer,
        name=name,
    )(x)
