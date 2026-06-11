from keras.layers import Dropout, Identity


def apply_drop_path(x, rate, name):
    if rate > 0.0:
        result = drop_path(x, rate, name)
    else:
        result = apply_identity(x, name)
    return result


def drop_path(x, rate, name):
    noise_shape = build_noise_shape(x)
    layer = Dropout(rate, noise_shape=noise_shape, name=name)
    return layer(x)


def apply_identity(x, name):
    layer = Identity(name=name)
    return layer(x)


def build_noise_shape(x):
    rank = len(x.shape)
    return (None,) + (1,) * (rank - 1)
