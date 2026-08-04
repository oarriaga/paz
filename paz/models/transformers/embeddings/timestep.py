"""Sinusoidal timestep embeddings for diffusion-style models."""
import math

from keras import ops


def sinusoidal(times, dim, max_period, frequency_scale):
    half = dim // 2
    exponents = ops.arange(half, dtype="float32") / half
    frequencies = frequency_scale * ops.exp(-math.log(max_period) * exponents)
    angles = ops.expand_dims(times, axis=-1) * frequencies
    return ops.concatenate((ops.cos(angles), ops.sin(angles)), axis=-1)
