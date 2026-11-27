import jax.numpy as jp

from paz.graphics.patterns import spherical_checker


def build_cell_center(u_cell, v_cell):
    theta = 2.0 * jp.pi * (0.5 - (u_cell + 0.5) / 16.0)
    phi = jp.pi * (1.0 - (v_cell + 0.5) / 16.0)
    x, z = jp.sin(phi) * jp.sin(theta), jp.sin(phi) * jp.cos(theta)
    return jp.array([x, jp.cos(phi), z])


def test_compute_colors_alternates_between_neighbor_cells():
    color_A = jp.array([1.0, 0.0, 0.0])
    color_B = jp.array([0.0, 0.0, 1.0])
    cells = [(4, 8), (5, 8), (4, 9), (5, 9)]
    points = jp.array([build_cell_center(*cell) for cell in cells])
    expected_colors = jp.vstack([color_A, color_B, color_B, color_A])
    actual_colors = spherical_checker.compute_colors(points, color_A, color_B)
    assert jp.allclose(actual_colors, expected_colors)
