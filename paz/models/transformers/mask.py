"""Attention masks."""
from keras import ops


def causal(query_positions, key_positions):
    query = ops.expand_dims(query_positions, axis=-1)
    key = ops.expand_dims(key_positions, axis=-2)
    return ops.less_equal(key, query)


def sliding_window(query_positions, key_positions, window_size):
    query = ops.expand_dims(query_positions, axis=-1)
    key = ops.expand_dims(key_positions, axis=-2)
    return ops.less(query - key, window_size)
