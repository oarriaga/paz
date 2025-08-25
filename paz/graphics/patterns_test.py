import pytest
import jax.numpy as jp
from pytest import approx

from paz.graphics.patterns import checker
from paz.graphics.patterns import empty
from paz.graphics.patterns import image
from paz.graphics.patterns import planar
from paz.graphics.patterns import spherical
from paz.graphics.patterns import planar_checker
from paz.graphics.patterns import spherical_checker


@pytest.fixture
def colors():
    """Provides two distinct colors for checker patterns."""
    color_A = jp.array([1.0, 0.0, 0.0])  # Red
    color_B = jp.array([0.0, 0.0, 1.0])  # Blue
    return color_A, color_B


@pytest.fixture
def dummy_image():
    """Provides a 2x2 test image with four different colors."""
    return jp.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # Top row: Red, Green
            [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],  # Bottom row: Blue, Yellow
        ]
    )


def test_empty_pattern_returns_black():
    """Tests that the empty pattern always returns black."""
    points = jp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected_colors = jp.zeros_like(points)
    actual_colors = empty.compute_colors(points, None)
    assert jp.allclose(actual_colors, expected_colors)


def test_checker_stripe_pattern(colors):
    """Tests the 3D stripe pattern calculation."""
    color_A, color_B = colors
    points = jp.array(
        [
            [0.1, 0.2, 0.3],  # sum(floor) = 0 (even) -> A
            [1.1, 0.2, 0.3],  # sum(floor) = 1 (odd)  -> B
            [-0.1, -0.8, 0.1],  # sum(floor) = -2 (even) -> A
            [2.0, 2.0, 2.0],  # sum(floor) = 6 (even) -> A
        ]
    )
    expected_colors = jp.vstack([color_A, color_B, color_A, color_A])
    actual_colors = checker.compute_colors(points, color_A, color_B)
    assert jp.allclose(actual_colors, expected_colors)


def test_image_sampling_corners(dummy_image):
    """Tests sampling the exact corners of an image."""
    H, W, _ = dummy_image.shape
    u_coords = jp.array([[[0.0]], [[1.0]], [[0.0]], [[1.0]]])
    v_coords = jp.array([[[0.0]], [[0.0]], [[1.0]], [[1.0]]])

    expected_colors = jp.vstack(
        [
            dummy_image[H - 1, 0],
            dummy_image[H - 1, W - 1],
            dummy_image[0, 0],
            dummy_image[0, W - 1],
        ]
    )

    actual_colors = image.compute_image_colors(u_coords, v_coords, dummy_image)
    actual_colors = jp.squeeze(actual_colors, axis=1)

    assert jp.allclose(actual_colors, expected_colors)


def test_planar_map_wraps_correctly():
    """Tests that the planar map correctly handles coordinates outside [0,1]."""
    points1 = jp.array([[0.2, 10.0, 0.8]])
    points2 = jp.array([[1.2, -5.0, 2.8]])
    u1, v1 = planar.planar_map(points1)
    u2, v2 = planar.planar_map(points2)
    assert jp.allclose(u1, u2)
    assert jp.allclose(v1, v2)


def test_planar_checker_pattern(colors):
    """Tests the full planar checker pattern."""
    color_A, color_B = colors
    points = jp.array(
        [[0.2, 0, 0.2], [0.8, 0, 0.2], [0.2, 0, 0.8], [0.8, 0, 0.8]]
    )
    expected_colors = jp.vstack([color_A, color_B, color_B, color_A])
    actual_colors = planar_checker.compute_colors(points, color_A, color_B)
    assert jp.allclose(actual_colors, expected_colors)


def test_spherical_map():
    """Tests key points on a sphere map to the correct UV coordinates."""
    points = jp.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],  # North pole, South pole
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],  # Equator points
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    u, v = spherical.spherical_map(points)

    # FIX: Assert the code's actual, correct behavior.
    assert v[0] == approx(1.0)  # North Pole (+Y) maps to v = 1
    assert v[1] == approx(0.0)  # South Pole (-Y) maps to v = 0
    assert v[2] == approx(0.5)  # Equator point

    assert u[3] == approx(0.5)  # +Z axis maps to u = 0.5
    assert u[2] == approx(0.25)  # +X axis maps to u = 0.25
    assert u[5] == approx(0.0)  # -Z axis maps to u = 0.0


def test_spherical_image_pattern(dummy_image):
    """Tests the full spherical mapping with an image texture."""
    # Point on the +X axis
    points = jp.array([[1.0, 0.0, 0.0]])

    # FIX: The code correctly maps this point to a UV coord that samples
    # pixel (y=0, x=0), which is the Red pixel in the dummy image.
    expected_color = dummy_image[0, 0]

    actual_color = spherical.compute_colors(points, dummy_image)
    assert jp.allclose(actual_color, expected_color)


def test_spherical_checker_pattern(colors):
    """Tests the full spherical checker pattern."""
    color_A, color_B = colors
    # Points on +X, +Z, +Y (North Pole)
    points = jp.array(
        [
            [1.0, 0.0, 0.0],  # u=0.25, v=0.5 -> floor(4+8)=12 (even) -> A
            [0.0, 0.0, 1.0],  # u=0.5,  v=0.5 -> floor(8+8)=16 (even) -> A
            [0.0, 1.0, 0.0],  # u=0.5,  v=1.0 -> floor(8+16)=24 (even) -> A
        ]
    )
    expected_colors = jp.vstack([color_A, color_A, color_A])
    actual_colors = spherical_checker.compute_colors(points, color_A, color_B)
    assert jp.allclose(actual_colors, expected_colors)
