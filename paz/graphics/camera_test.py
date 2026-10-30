import jax
import jax.numpy as jp
import paz
from paz.graphics import camera
from pytest import approx


def build_fov_rays(H, W, y_FOV):
    focal_length = H / (2.0 * jp.tan(y_FOV / 2.0))
    u_grid, v_grid = jp.meshgrid(jp.arange(W), jp.arange(H))
    x = (u_grid.ravel() + 0.5 - W / 2.0) / focal_length
    y = (H / 2.0 - v_grid.ravel() - 0.5) / focal_length
    directions = jp.stack([x, y, -jp.ones_like(x)], axis=-1)
    return paz.algebra.normalize(directions)


def build_calibrated_intrinsics():
    rows = [[611.1, 0.0, 315.5], [0.0, 610.0, 237.7], [0.0, 0.0, 1.0]]
    return jp.array(rows)


def test_compute_focal_length_90_degrees():
    assert camera.compute_focal_length(jp.pi / 2.0) == approx(1.0)


def test_compute_intrinsics():
    K = camera.compute_intrinsics(jp.pi / 2.0, 100, 200)
    assert K.shape == (3, 3)
    assert K[0, 0] == approx(50.0)
    assert K[1, 1] == approx(50.0)
    assert K[0, 2] == approx(99.5)
    assert K[1, 2] == approx(49.5)
    assert K[2, 2] == 1.0


def test_build_rays_identity():
    intrinsics = camera.compute_intrinsics(jp.pi / 2.0, 10, 10)
    origins, directions = camera.build_rays((10, 10), intrinsics, jp.eye(4))
    assert origins.shape == (100, 3)
    assert directions.shape == (100, 3)
    assert jp.all(origins == 0.0)
    norms = jp.linalg.norm(directions, axis=-1)
    assert jp.allclose(norms, 1.0, atol=1e-5)
    assert jp.all(directions[:, 2] < 0.0)


def test_build_rays_center_ray_points_forward():
    intrinsics = camera.compute_intrinsics(jp.pi / 2.0, 1, 1)
    origins, directions = camera.build_rays((1, 1), intrinsics, jp.eye(4))
    assert jp.all(origins == 0.0)
    assert jp.allclose(directions, jp.array([[0.0, 0.0, -1.0]]), atol=1e-5)


def test_build_rays_matches_field_of_view_camera():
    H, W, y_FOV = 33, 17, jp.pi / 3.0
    intrinsics = camera.compute_intrinsics(y_FOV, H, W)
    with jax.default_matmul_precision("highest"):
        _, directions = camera.build_rays((H, W), intrinsics)
    expected = build_fov_rays(H, W, y_FOV)
    assert jp.allclose(directions, expected, atol=1e-6)


def test_build_pixel_rays_shift_follows_principal_point():
    intrinsics = build_calibrated_intrinsics()
    shifted = intrinsics.at[0, 2].add(10.0)
    pixels = jp.array([[0.0, 0.0], [100.0, 37.0], [639.0, 479.0]])
    _, directions = camera.build_pixel_rays(intrinsics, pixels, jp.eye(4))
    _, moved = camera.build_pixel_rays(shifted, pixels, jp.eye(4))
    x = directions[:, 0] / -directions[:, 2]
    x_moved = moved[:, 0] / -moved[:, 2]
    assert jp.allclose(x - x_moved, 10.0 / intrinsics[0, 0], atol=1e-6)


def test_build_pixel_rays_agree_with_pointcloud_from_depth():
    intrinsics = build_calibrated_intrinsics()
    flip = jp.array([1.0, -1.0, -1.0])
    points = paz.pointcloud.from_depth(jp.full((4, 6), 2.5), intrinsics)
    pixels = camera.build_pixel_grid(4, 6)
    with jax.default_matmul_precision("highest"):
        _, directions = camera.build_pixel_rays(intrinsics, pixels, jp.eye(4))
    assert jp.allclose(jp.cross(points * flip, directions), 0.0, atol=1e-6)
