import math

import numpy as np

from paz.models.transformers.embeddings import sine


def test_embed_boxes_returns_four_axes():
    boxes = np.zeros((2, 5, 4), "float32")
    assert tuple(sine.embed_boxes(boxes, 8).shape) == (2, 5, 32)


def test_embed_boxes_at_zero_alternates_sine_and_cosine():
    boxes = np.zeros((1, 1, 4), "float32")
    embedded = np.array(sine.embed_boxes(boxes, 4))[0, 0]
    assert np.allclose(embedded, [0.0, 1.0, 0.0, 1.0] * 4)


def test_embed_axis_matches_closed_form():
    coordinates = np.array([[0.25]], "float32")
    frequencies = sine.build_frequencies(4, "float32")
    embedded = np.array(sine.embed_axis(coordinates, frequencies))[0, 0]
    angles = 0.25 * 2 * math.pi / np.array(frequencies)
    assert np.allclose(embedded[0::2], np.sin(angles[0::2]), atol=1e-6)
    assert np.allclose(embedded[1::2], np.cos(angles[1::2]), atol=1e-6)


def test_embed_boxes_orders_axes_as_y_x_width_height():
    boxes = np.array([[[0.1, 0.2, 0.3, 0.4]]], "float32")
    embedded = np.array(sine.embed_boxes(boxes, 4))[0, 0]
    frequencies = sine.build_frequencies(4, "float32")
    for index, value in enumerate([0.2, 0.1, 0.3, 0.4]):
        axis = np.array([[value]], "float32")
        expected = np.array(sine.embed_axis(axis, frequencies))[0, 0]
        assert np.allclose(embedded[index * 4:(index + 1) * 4], expected)
