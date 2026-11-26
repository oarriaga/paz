import jax.numpy as jp
import numpy as np

import paz


def test_to_RGB_spans_the_full_uint8_range():
    depth = jp.array([[1.0, 2.0, 3.0]])
    image = paz.depth.to_RGB(depth)
    assert image.dtype == jp.uint8
    assert np.array_equal(image, np.array([[0, 127, 255]]))
