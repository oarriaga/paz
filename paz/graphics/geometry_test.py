import pytest
import jax.numpy as jp
from pytest import approx

from paz.graphics import geometry
from paz.backend.graphics.constants import FARAWAY


@pytest.fixture
def quadratic_solutions():
    """Provides sample quadratic depth solutions."""
    # [hit, miss, hit_inverted_order]
    depths_A = jp.array([5.0, -1.0, 10.0])
    depths_B = jp.array([10.0, -5.0, 5.0])
    return depths_A, depths_B


def test_compute_quadratic_is_hit_with_positive_depth(quadratic_solutions):
    """Tests that a positive depth solution results in a hit."""
    depths_A, depths_B = quadratic_solutions
    is_valid = jp.array([True, True, True])
    is_hit = geometry.compute_quadratic_is_hit(depths_A, depths_B, is_valid)
    assert is_hit[0]


def test_compute_quadratic_is_hit_with_negative_depths(quadratic_solutions):
    """Tests that two negative depth solutions result in a miss."""
    depths_A, depths_B = quadratic_solutions
    is_valid = jp.array([True, True, True])
    is_hit = geometry.compute_quadratic_is_hit(depths_A, depths_B, is_valid)
    assert not is_hit[1]


def test_compute_quadratic_is_hit_with_invalid_mask(quadratic_solutions):
    """Tests that an invalid hit is correctly masked."""
    depths_A, depths_B = quadratic_solutions
    is_valid = jp.array([False, True, True])
    is_hit = geometry.compute_quadratic_is_hit(depths_A, depths_B, is_valid)
    assert not is_hit[0]


def test_apply_hit_mask_when_hit():
    """Tests that depth is unchanged when the mask is True."""
    depths = jp.array([10.0])
    mask = jp.array([True])
    result = geometry.apply_hit_mask(mask, depths)
    assert result[0] == approx(10.0)


def test_apply_hit_mask_when_miss():
    """Tests that depth becomes FARAWAY when the mask is False."""
    depths = jp.array([10.0])
    mask = jp.array([False])
    result = geometry.apply_hit_mask(mask, depths)
    assert result[0] == approx(FARAWAY)


def test_compute_quadratic_depths_chooses_smaller_positive(quadratic_solutions):
    """Tests that the smaller of two positive depths is chosen."""
    depths_A, depths_B = quadratic_solutions
    depth = geometry.compute_quadratic_depths(depths_A, depths_B)
    assert depth[0] == approx(5.0)


def test_compute_quadratic_depths_chooses_positive_over_negative():
    """Tests that a positive depth is chosen over a negative one."""
    depths_A = jp.array([-5.0])
    depths_B = jp.array([5.0])
    depth = geometry.compute_quadratic_depths(depths_A, depths_B)
    assert depth[0] == approx(5.0)


def test_compute_points3D_calculates_correct_position():
    """Tests the 3D point calculation from origin, direction, and depth."""
    origins = jp.array([[1.0, 2.0, 3.0]])
    directions = jp.array([[0.0, 0.0, -1.0]])
    depth = jp.array([10.0])

    points = geometry.compute_points3D(origins, directions, depth)
    expected = jp.array([[1.0, 2.0, -7.0]])
    assert jp.allclose(points, expected)


def test_compute_hits_to_light_is_normalized():
    """Tests that the vector from hit to light is a unit vector."""
    light_position = jp.array([10.0, 0.0, 0.0])
    hits = jp.array([[5.0, 0.0, 0.0]])
    hits_to_light = geometry.compute_hits_to_light(light_position, hits)
    norm = jp.linalg.norm(hits_to_light)
    assert norm == approx(1.0)


def test_reflect_vector():
    """Tests the reflection formula with a 45-degree incident vector."""
    incident_vector = jp.array([[0.7071, -0.7071, 0.0]])
    normal_vector = jp.array([[0.0, 1.0, 0.0]])

    reflection = geometry.reflect(incident_vector, normal_vector)

    expected = jp.array([[0.7071, 0.7071, 0.0]])
    assert jp.allclose(reflection, expected, atol=1e-4)


def test_sort_depths_returns_sorted_array():
    """Tests that an array of depths is correctly sorted."""
    depths_list = [jp.array([3.0, 1.0]), jp.array([2.0, 4.0])]
    sorted_depths = geometry.sort_depths(depths_list)
    expected = jp.array([[2.0, 1.0], [3.0, 4.0]])
    assert jp.all(sorted_depths == expected)
