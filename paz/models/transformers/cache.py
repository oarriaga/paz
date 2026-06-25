"""Key/value cache for cached attention decoding.

Layout ``(batch, num_layers, 2, max_length, num_heads, head_dim)`` where axis 2
stacks (key, value). ``update`` writes one per-layer entry in place.
"""
from keras import ops


def build(batch, num_layers, max_length, num_heads, head_dim, dtype="float32"):
    shape = (batch, num_layers, 2, max_length, num_heads, head_dim)
    return ops.zeros(shape, dtype=dtype)


def update(state, index, key, value):
    entry = ops.stack((key, value), axis=1)
    return ops.slice_update(state, [0, 0, index, 0, 0], entry)
