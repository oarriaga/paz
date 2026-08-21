import pytest
import jax.numpy as jp

from paz.graphics.mesh.tile import assemble
from paz.graphics.mesh.tile import assert_exact_tile_side
from paz.graphics.mesh.tile import make_ray_origins
from paz.graphics.mesh.tile import make_ray_targets
from paz.graphics.mesh.tile import make_tile_coordinates
from paz.graphics.mesh.tile import transform_tile_rays


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


def test_make_ray_origins_shape():
    origins = make_ray_origins(10, 15)
    assert origins.shape == (150, 4)
    assert jp.allclose(origins[:, 3], 1.0)


def test_make_ray_targets_shape():
    tile_arg = jp.array([0, 0])
    targets = make_ray_targets(10, 15, 0.01, 0.5, 0.5, tile_arg)
    assert targets.shape == (150, 4)
    assert jp.allclose(targets[:, 2], -1.0)
    assert jp.allclose(targets[:, 3], 1.0)


def test_transform_tile_rays_output_3d():
    origins = jp.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    targets = jp.array([[0.1, 0.2, -1.0, 1.0], [0.3, 0.1, -1.0, 1.0]])
    args = jp.eye(4), origins, targets
    world_origins, world_directions = transform_tile_rays(*args)
    assert world_origins.shape == (2, 3)
    assert world_directions.shape == (2, 3)


def test_transform_tile_rays_normalized():
    origins = jp.array([[0.0, 0.0, 0.0, 1.0]])
    targets = jp.array([[0.5, 0.5, -1.0, 1.0]])
    _, directions = transform_tile_rays(jp.eye(4), origins, targets)
    norms = jp.linalg.norm(directions, axis=-1)
    assert jp.allclose(norms, 1.0, atol=1e-4)


def test_assemble_reconstructs_image():
    blocks = jp.arange(24).reshape(4, 6, 1).astype(float)
    image = assemble(4, 6, 2, 2, blocks)
    assert image.shape == (4, 6, 1)
