import tensorflow as tf


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def max_pooling2D(pool_size, strides, padding):
    return tf.keras.layers.MaxPooling2D(pool_size, strides, padding)


def transpose_tensor(tensor, permutes):
    return tf.transpose(tensor, permutes)


def elementwise_equality(tensor_a, tensor_b):
    return tf.math.equal(tensor_a, tensor_b)


def cast_tensor(tensor, dtype):
    return tf.cast(tensor, dtype)


def reshape_tensor(tensor, shape):
    return tf.reshape(tensor, shape)


def where_true(condition, tensor=None):
    return tf.where(condition, tensor)


def fill_tensor(shape, value):
    return tf.fill(shape, value)


def stack_tensors(list_of_tensors, axis=0):
    return tf.stack(list_of_tensors, axis)


def gather_nd(params, indices):
    return tf.gather_nd(params, indices)


def find_k_largest_entries(tensor, k):
    return tf.math.top_k(tensor, k)


def up_sampling2D(size, interpolation):
    return tf.keras.layers.UpSampling2D(size=size, interpolation=interpolation)


def concatenate_tensors(list_of_tensors, axis=0):
    return tf.concat(list_of_tensors, axis)
