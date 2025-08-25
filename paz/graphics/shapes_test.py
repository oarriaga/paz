import pytest
import jax.numpy as jp

from paz.graphics.shapes import sphere
from paz.graphics.shapes import plane
from paz.graphics.shapes import cube
from paz.graphics.shapes import cylinder
from paz.graphics.shapes import cone

from paz.graphics.constants import FARAWAY


@pytest.fixture
def ray():
    """Provides a sample ray origin and direction."""
    origin = jp.array([[0.0, 0.0, -5.0]])
    direction = jp.array([[0.0, 0.0, 1.0]])
    return origin, direction


def test_sphere_intersection_hit(ray):
    """Tests a ray that directly intersects the sphere."""
    origin, direction = ray
    hit_mask, _, depth = sphere.intersect_canonical_sphere(origin, direction)
    assert hit_mask[0]
    assert jp.allclose(depth[0], 4.0)


def test_sphere_intersection_miss():
    """Tests a ray that misses the sphere."""
    origin = jp.array([[0.0, 2.0, -5.0]])
    direction = jp.array([[0.0, 0.0, 1.0]])
    hit_mask, _, depth = sphere.intersect_canonical_sphere(origin, direction)
    assert not hit_mask[0]
    assert jp.allclose(depth[0], FARAWAY)


def test_sphere_normal_at_pole():
    """Tests the normal on the north pole of the sphere."""
    point = jp.array([[0.0, 1.0, 0.0]])
    expected_normal = jp.array([0.0, 1.0, 0.0])
    actual_normal = sphere.compute_canonical_normals_sphere(point)
    assert jp.allclose(actual_normal, expected_normal)


def test_sphere_normal_at_equator():
    """Tests the normal on the equator of the sphere."""
    point = jp.array([[1.0, 0.0, 0.0]])
    expected_normal = jp.array([1.0, 0.0, 0.0])
    actual_normal = sphere.compute_canonical_normals_sphere(point)
    assert jp.allclose(actual_normal, expected_normal)


def test_plane_intersection_hit():
    """Tests a ray intersecting the plane from above."""
    origin = jp.array([[0.0, 10.0, 0.0]])
    direction = jp.array([[0.0, -1.0, 0.0]])
    hit_mask, _, depth = plane.intersect_canonical_plane(origin, direction)
    assert hit_mask[0]
    assert jp.allclose(depth[0], 10.0)


def test_plane_intersection_miss_parallel():
    """Tests a ray parallel to the plane."""
    origin = jp.array([[0.0, 10.0, 0.0]])
    direction = jp.array([[1.0, 0.0, 0.0]])
    hit_mask, _, depth = plane.intersect_canonical_plane(origin, direction)
    assert not hit_mask[0]


def test_plane_normal():
    """Tests the normal of the canonical plane."""
    point = jp.array([[10.0, 0.0, -5.0]])
    expected_normal = jp.array([0.0, 1.0, 0.0])
    actual_normal = plane.compute_canonical_normals_plane(point)
    assert jp.allclose(actual_normal, expected_normal)


def test_cube_intersection_hit(ray):
    """Tests a ray that goes through the center of the cube."""
    origin, direction = ray
    hit_mask, _, depth = cube.intersect_canonical_cube(origin, direction)
    assert hit_mask[0]
    assert jp.allclose(depth[0], 4.0)


def test_cube_intersection_miss():
    """Tests a ray that misses the cube."""
    origin = jp.array([[0.0, 2.0, -5.0]])
    direction = jp.array([[0.0, 0.0, 1.0]])
    hit_mask, _, depth = cube.intersect_canonical_cube(origin, direction)
    assert not hit_mask[0]


def test_cube_normal_on_positive_x_face():
    """Tests the normal on the face where X=1."""
    point = jp.array([[1.0, 0.5, -0.5]])
    expected_normal = jp.array([1.0, 0.0, 0.0])
    actual_normal = cube.compute_canonical_normals_cube(point)
    assert jp.allclose(actual_normal, expected_normal)


def test_cube_normal_on_negative_y_face():
    """Tests the normal on the face where Y=-1."""
    point = jp.array([[0.2, -1.0, -0.8]])
    expected_normal = jp.array([0.0, -1.0, 0.0])
    actual_normal = cube.compute_canonical_normals_cube(point)
    assert jp.allclose(actual_normal, expected_normal)


def test_cylinder_intersection_hit_wall():
    """Tests a ray that intersects the cylinder wall."""
    origin = jp.array([[0.0, 0.5, -5.0]])
    direction = jp.array([[0.0, 0.0, 1.0]])
    hit_mask, _, depth = cylinder.intersect_canonical_cylinder(
        origin, direction
    )
    assert hit_mask[0]
    assert jp.allclose(depth[0], 4.0)


def test_cylinder_intersection_hit_top_cap():
    """Tests a ray that intersects the top cap."""
    origin = jp.array([[0.5, 5.0, 0.0]])
    direction = jp.array([[0.0, -1.0, 0.0]])
    hit_mask, _, depth = cylinder.intersect_canonical_cylinder(
        origin, direction
    )
    assert hit_mask[0]
    assert jp.allclose(depth[0], 4.0)


def test_cylinder_normal_on_top_cap():
    """Tests the normal on the cylinder's top cap."""
    point = jp.array([[0.5, 1.0, 0.5]])
    expected_normal = jp.array([0.0, 1.0, 0.0])
    actual_normal = cylinder.compute_canonical_normals_cylinder(point)
    assert jp.allclose(actual_normal, expected_normal)


def test_cylinder_normal_on_wall():
    """Tests the normal on the cylinder's side wall."""
    point = jp.array([[1.0, 0.5, 0.0]])
    expected_normal = jp.array([1.0, 0.0, 0.0])
    actual_normal = cylinder.compute_canonical_normals_cylinder(point)
    assert jp.allclose(actual_normal, expected_normal)


def test_cone_intersection_hit_wall():
    """Tests a ray that intersects the cone wall."""
    origin = jp.array([[0.0, -0.5, -5.0]])
    direction = jp.array([[0.0, 0.0, 1.0]])
    hit_mask, _, depth = cone.intersect_canonical_cone(origin, direction)
    assert hit_mask[0]
    assert jp.allclose(depth[0], 4.5)


def test_cone_intersection_hit_cap():
    """Tests a ray that intersects the cone's cap."""
    origin = jp.array([[0.0, 5.0, 0.0]])
    direction = jp.array([[0.0, -1.0, 0.0]])
    hit_mask, _, depth = cone.intersect_canonical_cone(origin, direction)
    assert hit_mask[0]
    assert jp.allclose(depth[0], 5.0)


def test_cone_normal_on_cap():
    """Tests the normal on the cone's cap."""
    point = jp.array([[0.0, 0.0, 0.0]])
    expected_normal = jp.array([0.0, 1.0, 0.0])
    actual_normal = cone.compute_canonical_normals_cone(point)
    assert jp.allclose(actual_normal, expected_normal)


def test_cone_normal_on_wall():
    """Tests the normal on the cone's wall."""
    point = jp.array([[0.3, -0.5, 0.4]])
    expected_normal = jp.array([0.3, 0.5, 0.4])
    actual_normal = cone.compute_canonical_normals_cone(point)
    assert jp.allclose(
        actual_normal / jp.linalg.norm(actual_normal),
        expected_normal / jp.linalg.norm(expected_normal),
    )
