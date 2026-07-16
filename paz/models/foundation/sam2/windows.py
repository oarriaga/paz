"""Non-overlapping window partition and unpartition for Hiera attention.

Operates on channels-last ``(batch, height, width, channels)`` tensors and
pads to a whole number of windows, matching the official SAM 2 backbone. Pure
``keras.ops`` so the functions trace inside functional models.
"""
from keras import ops


def window_partition(x, window):
    height, width, channels = x.shape[1], x.shape[2], x.shape[3]
    pad_h = (window - height % window) % window
    pad_w = (window - width % window) % window
    x = ops.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
    padded_h, padded_w = height + pad_h, width + pad_w
    shape = (-1, padded_h // window, window, padded_w // window, window,
             channels)
    x = ops.reshape(x, shape)
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = ops.reshape(x, (-1, window, window, channels))
    return windows, (padded_h, padded_w)


def window_unpartition(windows, window, padded_size, size):
    padded_h, padded_w = padded_size
    height, width = size
    channels = windows.shape[-1]
    shape = (-1, padded_h // window, padded_w // window, window, window,
             channels)
    x = ops.reshape(windows, shape)
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    x = ops.reshape(x, (-1, padded_h, padded_w, channels))
    return x[:, :height, :width, :]
