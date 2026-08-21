import jax.numpy as jp

from paz.graphics.patterns import planar


def test_planar_map_wraps_outside_unit_range():
    points_inside = jp.array([[0.2, 10.0, 0.8]])
    points_outside = jp.array([[1.2, -5.0, 2.8]])
    u_inside, v_inside = planar.planar_map(points_inside)
    u_outside, v_outside = planar.planar_map(points_outside)
    assert jp.allclose(u_inside, u_outside)
    assert jp.allclose(v_inside, v_outside)
