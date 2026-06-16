import keras
from keras.layers import Conv2D, Reshape


def build_patch_embedding(x, patch_size, embedding_dimension, name):
    patch_HW = to_pair(patch_size)
    projected = project_patches(x, embedding_dimension, patch_HW, name)
    flattened = flatten_patches(projected, embedding_dimension)
    return flattened


def project_patches(x, embedding_dimension, patch_HW, name):
    # args = dict(
    #     filters=embedding_dimension,
    #     kernel_size=patch_HW,
    #     strides=patch_HW,
    #     padding="valid",
    #     name=f"{name}_proj",
    # )
    # return Conv2D(**args)(x)
    return Conv2D(
        filters=embedding_dimension,
        kernel_size=patch_HW,
        strides=patch_HW,
        padding="valid",
        name=f"{name}_proj",
    )(x)


def flatten_patches(x, embedding_dimension):
    return Reshape((-1, embedding_dimension))(x)


def to_pair(size_or_shape_patch):
    is_pair = isinstance(size_or_shape_patch, (list, tuple))
    if is_pair:
        shape_patch = tuple(size_or_shape_patch)
    else:
        shape_patch = (size_or_shape_patch, size_or_shape_patch)
    return shape_patch
