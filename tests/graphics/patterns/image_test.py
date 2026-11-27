import jax.numpy as jp

from paz.graphics.patterns import image


def build_four_color_image():
    top_row = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    bottom_row = [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    return jp.array([top_row, bottom_row])


def test_compute_image_colors_samples_corners():
    four_colors = build_four_color_image()
    H, W, _ = four_colors.shape
    u = jp.array([[[0.0]], [[1.0]], [[0.0]], [[1.0]]])
    v = jp.array([[[0.0]], [[0.0]], [[1.0]], [[1.0]]])
    corners = [four_colors[H - 1, 0], four_colors[H - 1, W - 1]]
    corners += [four_colors[0, 0], four_colors[0, W - 1]]
    expected_colors = jp.vstack(corners)
    actual_colors = image.compute_image_colors(u, v, four_colors)
    assert jp.allclose(jp.squeeze(actual_colors, axis=1), expected_colors)
