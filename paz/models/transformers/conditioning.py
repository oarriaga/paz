"""Adaptive layer-norm (adaLN) conditioning signals.

Shift, scale, and gate signals have shape [batch, dim] and are broadcast
over the sequence axis of [batch, sequence, dim] activations.
"""
from keras import ops


def modulate(x, shift, scale):
    shift = ops.expand_dims(shift, axis=1)
    scale = ops.expand_dims(scale, axis=1)
    return x * (1.0 + scale) + shift


def gate(x, values):
    return x * ops.expand_dims(values, axis=1)
