import pytest
import jax.numpy as jp

import paz
from paz.graphics.mesh.tile import assemble
from paz.graphics.mesh.tile import assert_exact_tile_side
from paz.graphics.mesh.tile import build_tile_rays
from paz.graphics.mesh.tile import make_tile_coordinates


def build_calibrated_intrinsics():
    rows = [[70.0, 0.0, 30.0], [0.0, 65.0, 20.0], [0.0, 0.0, 1.0]]
    return jp.array(rows)


def test_assert_exact_tile_side_valid():
    assert_exact_tile_side(100, 2)
    assert_exact_tile_side(100, 5)
    assert_exact_tile_side(100, 10)


def test_assert_exact_tile_side_invalid():
    with pytest.raises(ValueError):
        assert_exact_tile_side(100, 3)


def test_make_tile_coordinates_shape():
    coordinates = make_tile_coordinates(3, 4)
    assert coordinates.shape == (12, 2)


def test_build_tile_rays_match_full_image_block():
    H, W = 48, 64
    intrinsics = build_calibrated_intrinsics()
    pose = paz.SE3.translation(jp.array([0.5, -1.0, 2.0]))
    _, directions = paz.graphics.camera.build_rays((H, W), intrinsics, pose)
    args = H, W, 2, 4, intrinsics, pose, jp.array([3, 1])
    origins, tile_directions = build_tile_rays(*args)
    block = directions.reshape(H, W, 3)[24:48, 48:64].reshape(-1, 3)
    assert origins.shape == (24 * 16, 3)
    assert jp.allclose(tile_directions, block, atol=1e-6)


def test_build_tile_rays_normalized():
    args = 10, 15, 1, 1, build_calibrated_intrinsics(), jp.eye(4)
    _, directions = build_tile_rays(*args, jp.array([0, 0]))
    norms = jp.linalg.norm(directions, axis=-1)
    assert jp.allclose(norms, 1.0, atol=1e-4)


def test_assemble_reconstructs_image():
    blocks = jp.arange(24).reshape(4, 6, 1).astype(float)
    image = assemble(4, 6, 2, 2, blocks)
    assert image.shape == (4, 6, 1)
