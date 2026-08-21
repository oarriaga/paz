"""Sine embeddings of normalized coordinates, in the DETR convention.

Each coordinate is scaled by ``2 * pi``, divided by a geometric frequency
ladder, and turned into interleaved sine and cosine pairs. This differs from
``timestep.sinusoidal``, which concatenates a cosine half and a sine half.
"""
import math

from keras import ops


def embed_boxes(boxes, dim):
    """Embeds normalized ``(cx, cy, w, h)`` boxes into ``4 * dim`` features."""
    frequencies = build_frequencies(dim, boxes.dtype)
    parts = [embed_axis(boxes[..., arg], frequencies) for arg in (1, 0, 2, 3)]
    return ops.concatenate(parts, axis=-1)


def embed_axis(coordinates, frequencies):
    scaled = ops.expand_dims(coordinates * 2 * math.pi, -1) / frequencies
    sines, cosines = ops.sin(scaled[..., 0::2]), ops.cos(scaled[..., 1::2])
    interleaved = ops.stack([sines, cosines], axis=-1)
    shape = (-1, coordinates.shape[1], frequencies.shape[0])
    return ops.reshape(interleaved, shape)


def build_frequencies(dim, dtype):
    """Builds the geometric ladder of wavelengths the embedding divides by.

    ``temperature`` is the ratio between the longest and shortest wavelength.
    Its value is the one the original transformer picked and DETR kept, so it
    is fixed by the published weights rather than free to tune.

    Consecutive entries are paired because ``embed_axis`` reads a sine from the
    even ones and a cosine from the odd ones: halving the step index makes both
    halves of a pair share a wavelength, which is what lets the pair encode one
    angle.
    """
    temperature = 10000.0
    steps = ops.arange(dim, dtype=dtype)
    pair_index = 2 * ops.floor(steps / 2)
    return temperature ** (pair_index / dim)
