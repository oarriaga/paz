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


def test_compute_focal_length_90_degrees():
    """Tests focal length for a 90-degree FOV."""
    y_FOV = jp.pi / 2.0
    assert camera.compute_focal_length(y_FOV) == approx(1.0)


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


def test_compute_image_sizes_square():
    y_FOV = jp.pi / 2.0
    aspect_ratio = 1.0
    H, W = camera.compute_image_sizes(y_FOV, aspect_ratio)
    assert H == approx(2.0)
    assert W == approx(2.0)


def test_compute_image_sizes_landscape():
    y_FOV = jp.pi / 2.0
    aspect_ratio = 2.0
    H, W = camera.compute_image_sizes(y_FOV, aspect_ratio)
    assert H == approx(2.0)
    assert W == approx(4.0)


def test_compute_pixel_size():
    """Tests the pixel size calculation."""
    width_in_world = 2.0
    width_in_pixels = 200
    expected_pixel_size = 2.0 / 200
    assert camera.compute_pixel_size(width_in_world, width_in_pixels) == approx(
        expected_pixel_size
    )


@pytest.fixture
def ray_directions_square(square_setup):
    """Pre-calculates ray directions for a square image."""
    p = square_setup
    aspect_ratio = camera.compute_aspect_ratio(p["H"], p["W"])
    H_world, W_world = camera.compute_image_sizes(p["y_FOV"], aspect_ratio)
    return camera.build_ray_directions(p["H"], p["W"], H_world, W_world)


def test_build_ray_directions_output_shape(ray_directions_square, square_setup):
    """Tests the output shape of ray directions."""
    H, W = square_setup["H"], square_setup["W"]
    assert ray_directions_square.shape == (H * W, 3)


def test_build_ray_directions_z_coordinate(ray_directions_square):
    """Tests the Z component of ray directions is -1."""
    z_coords = ray_directions_square[:, 2]
    assert jp.all(z_coords == -1.0)


def test_build_ray_origins_shape(square_setup):
    """Tests the output shape of ray origins."""
    H, W = square_setup["H"], square_setup["W"]
    origins = camera.build_ray_origins(H, W)
    assert origins.shape == (H * W, 3)


def test_build_ray_origins_values(square_setup):
    """Tests that all ray origins are [0, 0, 0]."""
    H, W = square_setup["H"], square_setup["W"]
    origins = camera.build_ray_origins(H, W)
    expected_origin = jp.array([0.0, 0.0, 0.0])
    assert jp.all(origins == expected_origin)


def test_build_rays_identity():
    """Test build_rays with identity camera matrix."""
    size = (10, 10)
    y_FOV = jp.pi / 2.0
    origins, directions = camera.build_rays(size, y_FOV, jp.eye(4))

    assert origins.shape == (100, 3)
    assert directions.shape == (100, 3)
    assert jp.all(origins == 0.0)
    norms = jp.linalg.norm(directions, axis=-1)
    assert jp.allclose(norms, 1.0, atol=1e-5)
    assert jp.all(directions[:, 2] < 0.0)


def test_build_rays_center_ray_points_forward():
    origins, directions = camera.build_rays((1, 1), jp.pi / 2.0, jp.eye(4))
    expected = jp.array([[0.0, 0.0, -1.0]])

    assert jp.all(origins == 0.0)
    assert jp.allclose(directions, expected, atol=1e-5)


def test_compute_intrinsics():
    y_FOV = jp.pi / 2.0
    H, W = 100, 200
    K = camera.compute_intrinsics(y_FOV, H, W)

    # Focal length y = 1/tan(45) = 1.
    # fy in pixels = 1 * H/2 = 50.
    # fx = fy (square pixels) = 50.
    # cx = W/2 = 100.
    # cy = H/2 = 50.

    assert K[0, 0] == approx(50.0)
    assert K[1, 1] == approx(50.0)
    assert K[0, 2] == approx(100.0)
    assert K[1, 2] == approx(50.0)
    assert K[2, 2] == 1.0
