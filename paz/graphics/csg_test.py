import jax
import jax.numpy as jp
import pytest

import paz
from paz import SE3
from paz.graphics import CSG, Group, Scene, Mesh, PointLight
from paz.graphics import Cube, Cylinder, Plane, Sphere
from paz.graphics import Material, CookTorranceMaterial
from paz.graphics import UNION, INTERSECTION, DIFFERENCE

IMAGE_SHAPE = 32, 32


# GPU matmul defaults to a precision that moves hit points by about 1e-3.
@pytest.fixture(autouse=True)
def full_matmul_precision():
    with jax.default_matmul_precision("highest"):
        yield


def build_camera():
    origin = jp.array([3.2, 3.0, 5.0])
    pose = SE3.view_transform(origin, jp.zeros(3), jp.array([0.0, 1.0, 0.0]))
    intrinsics = paz.graphics.camera.compute_intrinsics(0.6, *IMAGE_SHAPE)
    return intrinsics, pose


def render(scene, mask=None, shadows=False):
    intrinsics, pose = build_camera()
    lights = [PointLight(jp.full(3, 0.9), jp.array([-4.0, 6.0, 5.0]))]
    args = IMAGE_SHAPE, intrinsics, pose, scene, mask, lights, (1, 1), 1024
    return paz.graphics.render(*args, shadows)


def build_cube(build_material=Material):
    return Cube(jp.eye(4), build_material(jp.array([0.9, 0.2, 0.2])))


def build_ball(build_material=Material):
    scale = SE3.scaling(jp.full(3, 1.35))
    return Sphere(scale, build_material(jp.array([0.1, 0.2, 0.9])))


def build_side_ball(build_material=Material):
    shift = SE3.translation(jp.array([0.9, 0.5, 0.4]))
    return Sphere(shift, build_material(jp.array([0.1, 0.2, 0.9])))


def build_rods():
    scale = SE3.scaling(jp.array([0.55, 2.0, 0.55]))
    material = Material(jp.array([0.1, 0.8, 0.1]))
    rod_x = Cylinder(SE3.rotation_z(jp.pi / 2.0) @ scale, material)
    rod_z = Cylinder(SE3.rotation_x(jp.pi / 2.0) @ scale, material)
    return Group([Cylinder(scale, material), rod_x, rod_z])


def build_csg(operation=DIFFERENCE):
    body = CSG(build_cube(), build_ball(), INTERSECTION)
    return CSG(body, build_rods(), operation)


def build_floor():
    shift = SE3.translation(jp.array([0.0, -1.0, 0.0]))
    return Plane(shift, Material(jp.array([0.8, 0.8, 0.8])))


def is_inside_canonical(shape, points):
    world_to_shape = jp.linalg.inv(shape.transform)
    points = paz.algebra.transform_points(world_to_shape, points)
    radius = jp.sqrt(points[:, 0] ** 2 + points[:, 2] ** 2)
    is_capped = jp.abs(points[:, 1]) < 1.0
    is_inside = {paz.graphics.SPHERE: jp.sum(points**2, axis=1) < 1.0}
    is_inside[paz.graphics.CUBE] = jp.max(jp.abs(points), axis=1) < 1.0
    is_inside[paz.graphics.CYLINDER] = (radius < 1.0) & is_capped
    return is_inside[shape.type]


def is_inside_csg(points):
    is_inside = is_inside_canonical(build_cube(), points)
    is_inside = is_inside & is_inside_canonical(build_ball(), points)
    for rod in build_rods().shapes:
        is_inside = is_inside & ~is_inside_canonical(rod, points)
    return is_inside


def build_rays():
    """Builds camera rays and the cosine that turns distance into z-depth."""
    intrinsics, pose = build_camera()
    rays = paz.graphics.camera.build_rays(IMAGE_SHAPE, intrinsics, pose)
    origins, directions = rays
    camera_axis = -jp.linalg.inv(pose)[:3, 2]
    return origins, directions, directions @ camera_axis


def march_depth(distances):
    origins, directions, cosines = build_rays()
    points = origins[:, None] + distances[None, :, None] * directions[:, None]
    is_inside = is_inside_csg(points.reshape(-1, 3))
    is_inside = is_inside.reshape(len(origins), len(distances))
    depth = distances[jp.argmax(is_inside, axis=1)] * cosines
    depth = jp.where(jp.any(is_inside, axis=1), depth, 0.0)
    return depth.reshape(IMAGE_SHAPE)


def test_no_surface_is_missed_by_ray_marching():
    _, depth = render(Scene([build_csg()]))
    distances = jp.linspace(3.0, 10.0, 4001)
    marched = march_depth(distances)
    step = distances[1] - distances[0]
    assert jp.array_equal(depth > 0.0, marched > 0.0)
    # Marching can step over a sliver, so it may only land behind the depth.
    assert bool(jp.all(marched > depth - 2.0 * step))


def test_depth_lies_on_csg_boundary():
    _, depth = render(Scene([build_csg()]))
    origins, directions, cosines = build_rays()
    distances = (depth.reshape(-1) / cosines)[:, None]
    is_hit = depth.reshape(-1) > 0.0
    before = is_inside_csg(origins + (distances - 1e-4) * directions)
    behind = is_inside_csg(origins + (distances + 1e-4) * directions)
    assert not bool(jp.any(before & is_hit))
    assert bool(jp.all(behind | ~is_hit))


@pytest.mark.parametrize("build_material", [Material, CookTorranceMaterial])
def test_union_matches_plain_shapes(build_material):
    shapes = [build_cube(build_material), build_side_ball(build_material)]
    image, depth = render(Scene([CSG(*shapes, UNION)]))
    expected_image, expected_depth = render(Scene(shapes))
    assert jp.allclose(depth, expected_depth, atol=1e-4)
    assert jp.allclose(image, expected_image, atol=1e-3)


def test_union_shadows_match_plain_shapes():
    shapes = [build_cube(), build_side_ball()]
    image, _ = render(Scene([build_floor(), CSG(*shapes, UNION)]), None, True)
    expected, _ = render(Scene([build_floor()] + shapes), None, True)
    is_equal = jp.all(jp.abs(image - expected) < 1e-3, axis=-1)
    # The crease where both shapes meet may shadow one pixel differently.
    assert float(jp.mean(is_equal)) > 0.99


@pytest.mark.parametrize("operation", [UNION, INTERSECTION, DIFFERENCE])
def test_masked_leaf_is_skipped(operation):
    scene = Scene([CSG(build_cube(), build_side_ball(), operation)])
    _, depth = render(scene, [[True, False]])
    _, expected = render(Scene([build_cube()]))
    assert jp.allclose(depth, expected, atol=1e-4)


def test_difference_without_its_left_leaf_is_empty():
    scene = Scene([CSG(build_cube(), build_side_ball(), DIFFERENCE)])
    _, depth = render(scene, [[False, True]])
    assert not bool(jp.any(depth > 0.0))


def test_nested_mask_reaches_group_inside_csg():
    _, depth = render(Scene([build_csg()]), [[True, [True, False, False]]])
    body = CSG(build_cube(), build_ball(), INTERSECTION)
    rod = build_rods().shapes[0]
    _, expected = render(Scene([CSG(body, rod, DIFFERENCE)]))
    assert jp.allclose(depth, expected, atol=1e-4)


def test_flush_faces_are_carved():
    shift = SE3.translation(jp.array([0.0, 0.5, 0.0]))
    cutter = Cube(shift @ SE3.scaling(jp.full(3, 0.5)), Material())
    _, depth = render(Scene([CSG(build_cube(), cutter, DIFFERENCE)]))
    _, uncut = render(Scene([build_cube()]))
    assert float(jp.max(depth - uncut)) > 0.5


def test_empty_group_operand_is_rejected():
    scene = Scene([CSG(build_cube(), Group([]), DIFFERENCE)])
    with pytest.raises(ValueError):
        render(scene)


def test_single_mask_value_hides_whole_csg():
    scene = Scene([build_csg(), build_floor()])
    _, depth = render(scene, [False, True])
    _, expected = render(Scene([build_floor()]))
    assert jp.allclose(depth, expected, atol=1e-4)


def test_operations_vmap_in_one_compilation():
    def render_operation(operation):
        return render(Scene([build_csg(operation)]))[1]

    render_operations = jax.jit(jax.vmap(render_operation))
    depths = render_operations(jp.array([UNION, INTERSECTION, DIFFERENCE]))
    render_operations(jp.array([DIFFERENCE, UNION, INTERSECTION]))
    assert render_operations._cache_size() == 1
    for operation in [UNION, INTERSECTION, DIFFERENCE]:
        expected = render_operation(operation)
        assert jp.allclose(depths[operation], expected, atol=1e-4)


def test_leaf_masks_vmap_in_one_compilation():
    def render_mask(mask):
        return render(Scene([build_csg()]), [[True, mask]])[1]

    masks = jp.array([[True, True, True], [True, False, False]])
    render_masks = jax.jit(jax.vmap(render_mask))
    depths = render_masks(masks)
    assert render_masks._cache_size() == 1
    assert jp.allclose(depths[0], render(Scene([build_csg()]))[1], atol=1e-4)
    assert not jp.allclose(depths[0], depths[1], atol=1e-4)


def test_subtracted_leaf_casts_no_shadow():
    shift = SE3.translation(jp.array([-1.5, 2.5, 1.5]))
    cutter = Sphere(shift @ SE3.scaling(jp.full(3, 0.5)), Material())
    csg = CSG(build_cube(), cutter, DIFFERENCE)
    image, _ = render(Scene([build_floor(), csg]), None, True)
    expected, _ = render(Scene([build_floor(), build_cube()]), None, True)
    assert jp.allclose(image, expected, atol=1e-3)


def test_csg_and_mesh_occlude_each_other():
    vertices, faces, edges = paz.graphics.mesh.build_sphere(0.9, 2)
    colors = jp.full_like(vertices, 0.5)
    pose = SE3.translation(jp.array([1.2, 0.4, 1.0]))
    mesh = Mesh(vertices, colors, pose, Material(), faces, edges)
    _, depth = render(Scene([build_csg(), mesh]))
    _, csg_depth = render(Scene([build_csg()]))
    _, mesh_depth = render(Scene([mesh]))
    csg_depth = jp.where(csg_depth > 0.0, csg_depth, jp.inf)
    mesh_depth = jp.where(mesh_depth > 0.0, mesh_depth, jp.inf)
    expected = jp.minimum(csg_depth, mesh_depth)
    expected = jp.where(jp.isfinite(expected), expected, 0.0)
    assert jp.allclose(depth, expected, atol=1e-4)


def test_depth_gradient_reaches_leaf_pose():
    def compute_mean_depth(shift):
        cutter = build_side_ball()
        pose = SE3.translation(shift) @ cutter.transform
        cutter = cutter._replace(transform=pose)
        return jp.mean(render(Scene([CSG(build_cube(), cutter, 2)]))[1])

    gradient = jax.grad(compute_mean_depth)(jp.zeros(3))
    assert bool(jp.all(jp.isfinite(gradient)))
    assert bool(jp.any(gradient != 0.0))


def test_mesh_operand_is_rejected():
    vertices, faces, edges = paz.graphics.mesh.build_cube()
    colors = jp.ones_like(vertices)
    mesh = Mesh(vertices, colors, jp.eye(4), Material(), faces, edges)
    scene = Scene([CSG(build_cube(), mesh, DIFFERENCE)])
    with pytest.raises(ValueError):
        render(scene)
