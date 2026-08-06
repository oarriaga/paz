"""Window layout for vision-transformer tokens.

``partition`` splits a patch grid into ``num_windows x num_windows`` windows and
folds them into the batch, so a plain attention block only mixes tokens inside
one window. ``merge`` temporarily folds the windows back into the sequence for
blocks that must attend globally, and ``split`` undoes that.
"""
from keras import ops


def partition(patches, grid, num_windows):
    height, width = grid
    hidden_size = patches.shape[-1]
    rows, columns = height // num_windows, width // num_windows
    shape = (-1, num_windows, rows, num_windows, columns, hidden_size)
    windows = ops.transpose(ops.reshape(patches, shape), (0, 1, 3, 2, 4, 5))
    return ops.reshape(windows, (-1, rows * columns, hidden_size))


def unpartition(patches, grid, num_windows):
    height, width = grid
    hidden_size = patches.shape[-1]
    rows, columns = height // num_windows, width // num_windows
    shape = (-1, num_windows, num_windows, rows, columns, hidden_size)
    grouped = ops.transpose(ops.reshape(patches, shape), (0, 1, 3, 2, 4, 5))
    return ops.reshape(grouped, (-1, height, width, hidden_size))


def merge(tokens, num_windows):
    length = tokens.shape[1] * num_windows**2
    return ops.reshape(tokens, (-1, length, tokens.shape[-1]))


def split(tokens, num_windows):
    length = tokens.shape[1] // num_windows**2
    return ops.reshape(tokens, (-1, length, tokens.shape[-1]))
