import keras
from keras import ops


def correlate(x, kernel_size=(5, 5)):
    B, H, W, C = ops.shape(x)
    x = ops.relu(x)
    x = identity = ops.normalize(x, axis=-1)
    patches_args = (kernel_size, (1, 1), 1, "same")
    patches = ops.image.extract_patches(x, *patches_args)
    patches = ops.reshape(patches, (B, H, W, *kernel_size, C))
    identity = ops.expand_dims(identity, [3, 4])
    correlations = patches * identity
    return correlations


def block_3D(x, filters, kernel_size, use_bias=False):
    kwargs = {"use_bias": use_bias, "padding": "valid"}
    x = keras.layers.Conv3D(filters, (1, kernel_size, kernel_size), **kwargs)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)


def block_2D(x, filters, kernel_size=1, use_bias=False, activation="relu"):
    kwargs = {"use_bias": use_bias, "padding": "valid"}
    x = keras.layers.Conv2D(filters, kernel_size, **kwargs)(x)
    x = keras.layers.BatchNormalization()(x)
    if activation is not None:
        x = keras.layers.Activation(activation)(x)
    return x


def self_correlate(x, filters=[64, 64, 64, 640], kernel_size=3, bias=False):
    x = correlate(x, kernel_size=(5, 5))
    B, H, W, U, V, C = ops.shape(x)
    x = ops.reshape(x, (B, H * W, U * V, C))
    x = block_2D(x, filters[0], 1, bias, "relu")
    x = ops.reshape(x, (B, H * W, U, V, filters[0]))
    x = block_3D(x, filters[1], kernel_size, bias)
    x = block_3D(x, filters[2], kernel_size, bias)
    x = ops.reshape(x, (B, H, W, filters[2]))
    x = block_2D(x, filters[3], 1, bias, None)
    return x
