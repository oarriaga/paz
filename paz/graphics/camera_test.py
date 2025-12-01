import pytest
import jax.numpy as jp

from paz.graphics import camera
from pytest import approx


@pytest.fixture
def square_setup():
    """Parameters for a square image."""
    return {"H": 100, "W": 100, "y_FOV": jp.pi / 2}


@pytest.fixture
def landscape_setup():
    """Parameters for a landscape image."""
    return {"H": 100, "W": 200, "y_FOV": jp.pi / 2}


@pytest.fixture
def portrait_setup():
    """Parameters for a portrait image."""
    return {"H": 200, "W": 100, "y_FOV": jp.pi / 2}


def test_compute_aspect_ratio_square(square_setup):
    """Tests aspect ratio for a square image."""
    H, W = square_setup["H"], square_setup["W"]
    assert camera.compute_aspect_ratio(H, W) == 1.0


def test_compute_aspect_ratio_landscape(landscape_setup):
    """Tests aspect ratio for a landscape image."""
    H, W = landscape_setup["H"], landscape_setup["W"]
    assert camera.compute_aspect_ratio(H, W) == 2.0


def test_compute_aspect_ratio_portrait(portrait_setup):
    """Tests aspect ratio for a portrait image."""
    H, W = portrait_setup["H"], portrait_setup["W"]
    assert camera.compute_aspect_ratio(H, W) == 0.5


def test_compute_focal_length_90_degrees():
    """Tests focal length for a 90-degree FOV."""
    y_FOV = jp.pi / 2.0
    assert camera.compute_focal_length(y_FOV) == approx(1.0)


def test_compute_half_view_square_aspect():
    """Tests half_view with a square aspect ratio."""
    aspect_ratio = 1.0
    y_FOV = jp.pi / 2.0
    assert camera.compute_half_view(y_FOV, aspect_ratio) == approx(1.0)


def test_compute_half_view_landscape_aspect():
    """Tests half_view with a landscape aspect ratio."""
    aspect_ratio = 2.0
    y_FOV = jp.pi / 2.0
    assert camera.compute_half_view(y_FOV, aspect_ratio) == approx(2.0)


def test_compute_half_W_for_landscape_aspect(landscape_setup):
    """Tests half_W calculation for landscape images."""
    aspect_ratio = camera.compute_aspect_ratio(
        landscape_setup["H"], landscape_setup["W"]
    )
    half_view = camera.compute_half_view(landscape_setup["y_FOV"], aspect_ratio)
    assert camera.compute_half_W(aspect_ratio, half_view) == approx(half_view)


def test_compute_half_H_for_landscape_aspect(landscape_setup):
    """Tests half_H calculation for landscape images."""
    aspect_ratio = camera.compute_aspect_ratio(
        landscape_setup["H"], landscape_setup["W"]
    )
    half_view = camera.compute_half_view(landscape_setup["y_FOV"], aspect_ratio)
    assert camera.compute_half_H(aspect_ratio, half_view) == approx(
        half_view / aspect_ratio
    )


def test_compute_half_W_for_portrait_aspect(portrait_setup):
    """Tests half_W calculation for portrait images."""
    aspect_ratio = camera.compute_aspect_ratio(
        portrait_setup["H"], portrait_setup["W"]
    )
    half_view = camera.compute_half_view(portrait_setup["y_FOV"], aspect_ratio)
    assert camera.compute_half_W(aspect_ratio, half_view) == approx(
        half_view * aspect_ratio
    )


def test_compute_half_H_for_portrait_aspect(portrait_setup):
    """Tests half_H calculation for portrait images."""
    aspect_ratio = camera.compute_aspect_ratio(
        portrait_setup["H"], portrait_setup["W"]
    )
    half_view = camera.compute_half_view(portrait_setup["y_FOV"], aspect_ratio)
    assert camera.compute_half_H(aspect_ratio, half_view) == approx(half_view)


def test_compute_pixel_size():
    """Tests the pixel size calculation."""
    half_W = 1.0
    width_in_pixels = 200
    expected_pixel_size = 1.0 * 2.0 / 200
    assert camera.compute_pixel_size(half_W, width_in_pixels) == approx(
        expected_pixel_size
    )


@pytest.fixture
def ray_directions_square(square_setup):
    """Pre-calculates ray directions for a square image."""
    p = square_setup
    aspect_ratio = camera.compute_aspect_ratio(p["H"], p["W"])
    half_view = camera.compute_half_view(p["y_FOV"], aspect_ratio)
    half_W = camera.compute_half_W(aspect_ratio, half_view)
    half_H = camera.compute_half_H(aspect_ratio, half_view)
    pixel_size = camera.compute_pixel_size(half_W, p["W"])
    return camera.build_ray_directions(
        p["H"], p["W"], pixel_size, half_W, half_H
    )


def test_build_ray_directions_output_shape(ray_directions_square, square_setup):
    """Tests the output shape of ray directions."""
    H, W = square_setup["H"], square_setup["W"]
    assert ray_directions_square.shape == (H * W, 4)


def test_build_ray_directions_z_coordinate(ray_directions_square):
    """Tests the Z component of ray directions is -1."""
    z_coords = ray_directions_square[:, 2]
    assert jp.all(z_coords == -1.0)


def test_build_ray_directions_w_coordinate(ray_directions_square):
    """Tests the W component of ray directions is 1."""
    w_coords = ray_directions_square[:, 3]
    assert jp.all(w_coords == 1.0)


def test_build_ray_origins_shape(square_setup):
    """Tests the output shape of ray origins."""
    H, W = square_setup["H"], square_setup["W"]
    origins = camera.build_ray_origins(H, W)
    assert origins.shape == (H * W, 4)


def test_build_ray_origins_values(square_setup):
    """Tests that all ray origins are [0, 0, 0, 1]."""
    H, W = square_setup["H"], square_setup["W"]
    origins = camera.build_ray_origins(H, W)
    expected_origin = jp.array([0.0, 0.0, 0.0, 1.0])
    assert jp.all(origins == expected_origin)


def test_transform_rays_with_identity_matrix():
    """Tests transform_rays with an identity matrix."""
    origin = jp.array([[0.0, 0.0, 0.0, 1.0]])
    direction = jp.array([[0.5, 0.5, -1.0, 1.0]])
    identity_matrix = jp.eye(4)

    expected_direction_vec = jp.array([0.5, 0.5, -1.0])
    expected_norm_direction = expected_direction_vec / jp.linalg.norm(
        expected_direction_vec
    )

    out_origin, out_dir = camera.transform_rays(
        identity_matrix, origin, direction
    )

    assert jp.allclose(out_origin, jp.array([[0.0, 0.0, 0.0]]))
    assert jp.allclose(out_dir, expected_norm_direction.reshape(1, 3))


def test_transform_rays_origin_with_translation():
    """Tests ray origin transformation with a translation matrix."""
    origin = jp.array([[0.0, 0.0, 0.0, 1.0]])
    direction = jp.array([[0.0, 0.0, -1.0, 1.0]])
    world_to_camera = jp.eye(4).at[2, 3].set(-5)
    expected_origin = jp.array([[0.0, 0.0, -5.0]])

    out_origin, _ = camera.transform_rays(world_to_camera, origin, direction)
    assert jp.allclose(out_origin, expected_origin)


def test_transform_rays_direction_with_translation():
    """Tests ray direction transformation with a translation matrix."""
    origin = jp.array([[0.0, 0.0, 0.0, 1.0]])
    direction = jp.array([[0.0, 0.0, -1.0, 1.0]])
    world_to_camera = jp.eye(4).at[2, 3].set(-5)
    expected_direction = jp.array([[0.0, 0.0, -1.0]])

    _, out_dir = camera.transform_rays(world_to_camera, origin, direction)
    assert jp.allclose(out_dir, expected_direction, rtol=1e-4)
