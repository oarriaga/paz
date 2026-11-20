import jax
import jax.numpy as jp
import pytest

import paz
from paz.graphics import SPHERE, CUBE, CYLINDER, CONE, PLANE
from paz.graphics.shapes import solid

NUM_RAYS = 512


# GPU matmul defaults to a precision that moves sampled points by about 1e-3.
@pytest.fixture(autouse=True)
def full_matmul_precision():
    with jax.default_matmul_precision("highest"):
        yield


def build_shape(build):
    scale = paz.SE3.scaling(jp.array([0.8, 1.2, 0.6]))
    rotation = paz.SE3.rotation_x(0.4) @ paz.SE3.rotation_z(-0.7)
    shift = paz.SE3.translation(jp.array([0.2, -0.1, 0.3]))
    return build(shift @ rotation @ scale)


def sample_rays(key):
    key_origins, key_targets = jax.random.split(key)
    origins = jax.random.normal(key_origins, (NUM_RAYS, 3))
    origins = 4.0 * paz.algebra.normalize(origins)
    targets = 0.8 * jax.random.normal(key_targets, (NUM_RAYS, 3))
    return origins, paz.algebra.normalize(targets - origins)


def is_inside_canonical(shape_type, points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    radius = jp.sqrt(x**2 + z**2)
    is_inside = {SPHERE: jp.sum(points**2, axis=1) < 1.0}
    is_inside[CUBE] = jp.max(jp.abs(points), axis=1) < 1.0
    is_inside[CYLINDER] = (radius < 1.0) & (jp.abs(y) < 1.0)
    is_inside[CONE] = (radius < -y) & (y > -1.0)
    is_inside[PLANE] = y < 0.0
    return is_inside[shape_type]


def sample_is_inside(shape, origins, directions, depths):
    points = origins[:, None] + depths[None, :, None] * directions[:, None]
    world_to_shape = jp.linalg.inv(shape.transform)
    points = paz.algebra.transform_points(world_to_shape, points.reshape(-1, 3))
    is_inside = is_inside_canonical(shape.type, points)
    return is_inside.reshape(len(origins), len(depths))


SHAPES = [paz.graphics.Sphere, paz.graphics.Cube, paz.graphics.Cylinder]
SHAPES += [paz.graphics.Cone, paz.graphics.Plane]


@pytest.mark.parametrize("build", SHAPES)
def test_interval_matches_sampled_inside(build):
    shape = build_shape(build)
    origins, directions = sample_rays(jax.random.PRNGKey(0))
    enters, exits = solid.compute_interval(shape, origins, directions)
    depths = jp.linspace(-2.0, 10.0, 2401)
    sampled = sample_is_inside(shape, origins, directions, depths)
    predicted = (enters[:, None] < depths) & (depths < exits[:, None])
    step = depths[1] - depths[0]
    near_enter = jp.abs(depths - enters[:, None]) < step
    near_exit = jp.abs(depths - exits[:, None]) < step
    assert bool(jp.any(predicted))
    assert bool(jp.all((sampled == predicted) | near_enter | near_exit))


@pytest.mark.parametrize("build", SHAPES)
def test_enter_matches_closest_intersection(build):
    shape = build_shape(build)
    origins, directions = sample_rays(jax.random.PRNGKey(1))
    intersection = paz.graphics.shapes.intersect(shape, origins, directions)
    hit_mask, depth = intersection[0], intersection[1][:, 0]
    enters, _ = solid.compute_interval(shape, origins, directions)
    is_front = hit_mask & (enters > 0.0)
    assert bool(jp.any(is_front))
    assert jp.allclose(jp.where(is_front, enters - depth, 0.0), 0.0, atol=1e-4)


@pytest.mark.parametrize("build", SHAPES)
def test_normals_match_closest_intersection(build):
    shape = build_shape(build)
    origins, directions = sample_rays(jax.random.PRNGKey(2))
    intersection = paz.graphics.shapes.intersect(shape, origins, directions)
    hit_mask, _, points, expected, eyes = intersection
    normals = solid.compute_normals(shape, points, eyes)
    errors = jp.max(jp.abs(normals - expected), axis=1)
    is_equal = jp.where(hit_mask, errors < 1e-3, True)
    # Rays grazing a rim may pick the cap on one side and the body on other.
    assert float(jp.mean(is_equal)) > 0.99


def test_miss_is_empty_interval():
    origins = jp.array([[0.0, 3.0, -5.0]])
    directions = jp.array([[0.0, 0.0, 1.0]])
    for build in SHAPES[:4]:
        enters, exits = solid.compute_interval(build(), origins, directions)
        assert not bool(jp.any(enters < exits))


def test_ray_starting_inside_keeps_negative_enter():
    origins = jp.zeros((1, 3))
    directions = jp.array([[0.0, 0.0, 1.0]])
    for build in SHAPES[:3]:
        enters, exits = solid.compute_interval(build(), origins, directions)
        assert jp.allclose(enters, -1.0, atol=1e-5)
        assert jp.allclose(exits, 1.0, atol=1e-5)
