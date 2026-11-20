import pytest
import jax
import jax.numpy as jp
import paz
from paz.graphics import FARAWAY, Sphere, Cube, Cylinder, Cone, Plane
from paz.graphics.types import Material, Shape, Pattern


@pytest.fixture
def material():
    return Material()


@pytest.fixture
def pattern_256():
    return Pattern(image=jp.zeros((256, 256, 3)))


@pytest.fixture
def pattern_512():
    return Pattern(image=jp.zeros((512, 512, 3)))


@pytest.fixture
def shape_256_A(material, pattern_256):
    return Shape(
        transform=jp.eye(4),
        type=paz.graphics.SPHERE,
        material=material,
        pattern=pattern_256,
    )


@pytest.fixture
def shape_256_B(material, pattern_256):
    return Shape(
        transform=jp.eye(4).at[0, 3].set(1.0),
        type=paz.graphics.CUBE,
        material=material,
        pattern=pattern_256,
    )


@pytest.fixture
def shape_512(material, pattern_512):
    return Shape(
        transform=jp.eye(4),
        type=paz.graphics.PLANE,
        material=material,
        pattern=pattern_512,
    )


@pytest.fixture
def shape_default(material):
    """Provides a shape that uses the default 1x1 pattern."""
    return Shape(
        transform=jp.eye(4),
        type=paz.graphics.CONE,
        material=material,
        pattern=Pattern(),
    )


def test_group_by_pattern_size_with_empty_list():
    result = paz.graphics.shapes.group_by_pattern_size([])
    assert result == {}


def test_group_by_pattern_size_with_homogenous_list(shape_256_A, shape_256_B):
    shapes_list = [shape_256_A, shape_256_B]
    result = paz.graphics.shapes.group_by_pattern_size(shapes_list)

    assert len(result) == 1
    assert (256, 256) in result

    result_list = result[(256, 256)]
    assert len(result_list) == 2

    # `is` avoids the ValueError raised by comparing shapes holding arrays.
    assert result_list[0] is shape_256_A
    assert result_list[1] is shape_256_B


def test_group_by_pattern_size_with_heterogeneous_list(shape_256_A, shape_512):
    shapes_list = [shape_256_A, shape_512]
    result = paz.graphics.shapes.group_by_pattern_size(shapes_list)

    assert len(result) == 2
    assert (256, 256) in result
    assert (512, 512) in result
    assert result[(256, 256)] == [shape_256_A]
    assert result[(512, 512)] == [shape_512]


def test_group_by_pattern_size_with_default_pattern(shape_default):
    shapes_list = [shape_default]
    result = paz.graphics.shapes.group_by_pattern_size(shapes_list)

    assert len(result) == 1
    assert (1, 1) in result
    assert result[(1, 1)] == [shape_default]


def test_group_by_pattern_size_with_complex_mixed_list(
    shape_256_A, shape_512, shape_default, shape_256_B
):
    shapes_list = [shape_256_A, shape_512, shape_default, shape_256_B]
    result = paz.graphics.shapes.group_by_pattern_size(shapes_list)

    assert len(result) == 3
    assert (256, 256) in result
    assert (512, 512) in result
    assert (1, 1) in result

    group_256 = result[(256, 256)]
    assert len(group_256) == 2
    assert len(result[(512, 512)]) == 1
    assert len(result[(1, 1)]) == 1

    assert group_256[0] is shape_256_A
    assert group_256[1] is shape_256_B
    assert result[(512, 512)][0] is shape_512
    assert result[(1, 1)][0] is shape_default


def sample_rays(origin_radius):
    keys = jax.random.split(jax.random.PRNGKey(0))
    origins = jax.random.normal(keys[0], (512, 3))
    origins = origin_radius * paz.algebra.normalize(origins)
    targets = 0.8 * jax.random.normal(keys[1], (512, 3))
    return origins, paz.algebra.normalize(targets - origins)


def is_inside_canonical(shape_type, points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    radius = jp.sqrt(x**2 + z**2)
    is_inside = {paz.graphics.SPHERE: jp.sum(points**2, axis=1) < 1.0}
    is_inside[paz.graphics.CUBE] = jp.max(jp.abs(points), axis=1) < 1.0
    is_inside[paz.graphics.CYLINDER] = (radius < 1.0) & (jp.abs(y) < 1.0)
    is_inside[paz.graphics.CONE] = (radius < -y) & (y > -1.0)
    is_inside[paz.graphics.PLANE] = y < 0.0
    return is_inside[shape_type]


def sample_is_inside(shape, origins, directions, depths):
    points = origins[:, None] + depths[None, :, None] * directions[:, None]
    world_to_shape = jp.linalg.inv(shape.transform)
    points = paz.algebra.transform_points(world_to_shape, points.reshape(-1, 3))
    is_inside = is_inside_canonical(shape.type, points)
    return is_inside.reshape(len(origins), len(depths))


# A ray is inside a shape while an odd number of its roots are still ahead.
@pytest.mark.parametrize("origin_radius", [4.0, 0.3])
@pytest.mark.parametrize("build", [Sphere, Cube, Cylinder, Cone, Plane])
def test_roots_ahead_count_whether_a_ray_is_inside(build, origin_radius):
    scale = paz.SE3.scaling(jp.array([0.8, 1.2, 0.6]))
    shape = build(paz.SE3.rotation_x(0.4) @ scale)
    origins, directions = sample_rays(origin_radius)
    depths = jp.linspace(0.01, 10.0, 2001)
    # GPU matmul defaults to a precision that moves points by about 1e-3.
    with jax.default_matmul_precision("highest"):
        roots = paz.graphics.shapes.compute_roots(shape, origins, directions)[1]
        sampled = sample_is_inside(shape, origins, directions, depths)
    is_beyond = (roots[:, :, None] > depths) & (roots[:, :, None] != FARAWAY)
    predicted = jp.sum(is_beyond, axis=0) % 2 == 1
    step = depths[1] - depths[0]
    is_near_root = jp.any(jp.abs(roots[:, :, None] - depths) < step, axis=0)
    assert bool(jp.any(predicted))
    assert bool(jp.all((sampled == predicted) | is_near_root))


def compute_exit_normal(origin, direction):
    shape, rays = Cylinder(), (origin[None], direction[None])
    _, roots, _ = paz.graphics.shapes.compute_roots(shape, *rays)
    exit_depth = jp.max(jp.where(roots != FARAWAY, roots, -jp.inf), axis=0)
    args = shape, roots, exit_depth[:, None], *rays
    return paz.graphics.shapes.compute_surface(*args)[1][0]


def test_surface_at_a_cap_exit_has_the_cap_normal():
    origin, direction = jp.array([0.3, 3.0, 0.0]), jp.array([0.0, -1.0, 0.0])
    normal = compute_exit_normal(origin, direction)
    assert jp.allclose(normal, jp.array([0.0, 1.0, 0.0]), atol=1e-4)


def test_surface_at_a_side_exit_has_the_side_normal():
    origin, direction = jp.array([-3.0, 0.2, 0.0]), jp.array([1.0, 0.0, 0.0])
    normal = compute_exit_normal(origin, direction)
    assert jp.allclose(normal, jp.array([-1.0, 0.0, 0.0]), atol=1e-4)


def test_cap_roots_beyond_faraway_are_kept():
    shape = Cylinder(paz.SE3.scaling(jp.array([1.0, 3000.0, 1.0])))
    rays = jp.array([[0.3, 3005.0, 0.0]]), jp.array([[0.0, -1.0, 0.0]])
    _, roots, _ = paz.graphics.shapes.compute_roots(shape, *rays)
    assert jp.allclose(jp.sort(roots[:, 0]), jp.array([5, 1000, 1000, 6005]))
