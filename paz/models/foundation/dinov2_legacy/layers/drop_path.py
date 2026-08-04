from keras.layers import Dropout, Identity


def apply_drop_path(x, rate, name):
    if rate > 0.0:
        output = Dropout(rate, noise_shape=build_noise_shape(x), name=name)(x)
    else:
        output = Identity(name=name)(x)
    return output


def build_noise_shape(x):
    rank = len(x.shape)
    return (None,) + (1,) * (rank - 1)
