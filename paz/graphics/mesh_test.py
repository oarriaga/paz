import jax
import jax.numpy as jp
from pathlib import Path

import pytest
import paz
from paz.graphics.constants import NO_PATTERN
from paz.graphics.types import PointLight, Material, Pattern
from paz.backend.lie import SE3, SO3
from paz.graphics.mesh import (
    Mesh,
    BinArgs,
    build_cube,
    build_sphere,
    tile_render_binned_soft_mask,
)


def build_scene(*meshes):
    return paz.graphics.Scene(list(meshes)), None


def render(*args):
    return paz.graphics.render(*args)


def compute_max_abs_difference(array_A, array_B):
    return float(jp.max(jp.abs(array_A - array_B)))


def snapshot_path(filename):
    return str(Path(__file__).parent / "snapshots" / filename)


def assert_snapshot(array, filename, atol):
    paz.assert_snapshot(array, snapshot_path(filename), atol=atol)


def build_vertex_colors(vertices, color):
    return jp.repeat(jp.array([color]), len(vertices), axis=0)


def make_scene(image_shape=(20, 20)):
    camera_origin = jp.array([0.0, 1.0, -1.5])
    y_FOV = jp.pi / 4.0
    camera_pose = SE3.view_transform(
        camera_origin, jp.zeros(3), jp.array([0.0, 0.0, 1.0])
    )
    lights = [PointLight(jp.full((3,), 10.0), camera_origin)]
    vertices, faces, edges = build_cube(1.0)
    color = jp.array([[0.7, 0.3, 0.1]])
    vertex_colors = jp.repeat(color, len(vertices), axis=0)
    transform = SE3.translation(jp.zeros(3))
    material = Material(jp.zeros(3), 0.1, 0.9, 0.1, 100)
    mesh = Mesh(vertices, vertex_colors, transform, material, faces, edges)
    scene, mask = build_scene(mesh)
    return image_shape, y_FOV, camera_pose, scene, mask, lights


def render_scene(image_shape=(20, 20), tiles=(1, 1), chunk_size=1024):
    args = make_scene(image_shape)
    return render(*args, tiles, chunk_size)


def test_render_returns_correct_shapes():
    image, depth = render_scene((20, 20))
    assert image.shape == (20, 20, 3)
    assert depth.shape == (20, 20)


def test_render_produces_nonzero_image():
    image, depth = render_scene()
    assert jp.any(image < 1.0)


def test_render_jit_compatible():
    args = make_scene()
    args = args + ((1, 1), 1024)
    render_fn = jax.jit(render, static_argnums=(0, 6, 7))
    image, depth = render_fn(*args)
    assert image.shape == (20, 20, 3)


def test_render_rect_tiles_match_single_tile():
    expected_image, expected_depth = render_scene((20, 20), (1, 1), 1024)
    actual_image, actual_depth = render_scene((20, 20), (2, 4), 13)
    assert compute_max_abs_difference(actual_image, expected_image) <= 1e-4
    assert compute_max_abs_difference(actual_depth, expected_depth) <= 1e-4


def test_render_depth_is_chunk_invariant():
    _, expected_depth = render_scene((20, 20), (2, 2), 1024)
    _, actual_depth = render_scene((20, 20), (2, 2), 7)
    assert compute_max_abs_difference(actual_depth, expected_depth) <= 1e-4


def test_render_depth_respects_mask():
    image_shape, y_FOV, camera_pose, scene, _, lights = make_scene()
    mask = jp.zeros(len(scene.nodes), dtype=bool)
    args = image_shape, y_FOV, camera_pose, scene, mask, lights
    _, depth = render(*args, (1, 1), 1024)
    assert jp.allclose(depth, 0.0)


def test_render_masks_returns_mesh_masks():
    shape, y_FOV, pose, scene, mask, lights = make_scene()
    args = shape, y_FOV, pose, scene, lights, (0.1, 10.0), (2, 2), 1024
    masks = paz.graphics.render_masks(*args)
    assert masks.shape == (1, 20, 20, 1)
    assert jp.any(masks > 0.0)


def test_render_gradient_through_vertices():
    camera_origin = jp.array([0.0, 1.0, -1.5])
    y_FOV = jp.pi / 4.0
    image_shape = (10, 10)
    camera_pose = SE3.view_transform(
        camera_origin, jp.zeros(3), jp.array([0.0, 0.0, 1.0])
    )
    lights = [PointLight(jp.full((3,), 10.0), camera_origin)]
    vertices, faces, edges = build_cube(1.0)
    color = jp.array([[0.7, 0.3, 0.1]])
    vertex_colors = jp.repeat(color, len(vertices), axis=0)
    transform = SE3.translation(jp.zeros(3))
    material = Material(jp.zeros(3), 0.1, 0.9, 0.1, 100)

    def loss_fn(verts):
        mesh = Mesh(verts, vertex_colors, transform, material, faces, edges)
        scene, mask = build_scene(mesh)
        args = image_shape, y_FOV, camera_pose, scene, mask, lights
        image, _ = render(*args, (1, 1), 1024)
        return jp.sum(image)

    grad = jax.grad(loss_fn)(vertices)
    assert grad.shape == vertices.shape
    assert jp.any(grad != 0.0)


def build_cube_mesh():
    vertices, faces, edges = build_cube(1.0)
    colors = build_vertex_colors(vertices, [0.7, 0.3, 0.1])
    transform = SE3.translation(jp.array([-0.45, 0.0, 0.0]))
    material = Material(jp.zeros(3), 0.1, 0.9, 0.1, 100)
    args = vertices, colors, transform, material, faces, edges
    return Mesh(*args)


def build_sphere_mesh():
    vertices, faces, edges = build_sphere(0.6, 1)
    colors = build_vertex_colors(vertices, [0.1, 0.4, 0.8])
    transform = SE3.translation(jp.array([0.5, 0.15, -0.3]))
    material = Material(jp.zeros(3), 0.15, 0.8, 0.15, 50)
    args = vertices, colors, transform, material, faces, edges
    return Mesh(*args)


def make_multi_mesh_scene(image_shape=(24, 24)):
    camera_origin = jp.array([0.0, 0.7, -2.5])
    camera_pose = SE3.view_transform(
        camera_origin, jp.zeros(3), jp.array([0.0, 1.0, 0.0])
    )
    lights = [PointLight(jp.full((3,), 1.0), camera_origin)]
    scene, mask = build_scene(build_cube_mesh(), build_sphere_mesh())
    return image_shape, jp.pi / 4.0, camera_pose, scene, mask, lights


def render_multi_mesh(image_shape=(24, 24), tiles=(1, 1), chunk_size=1024):
    args = make_multi_mesh_scene(image_shape)
    return render(*args, tiles, chunk_size)


def test_multi_mesh_render_matches_snapshot():
    image, depth = render_multi_mesh()
    assert_snapshot(image, "mesh_multi_image.npy", 1e-3)
    assert_snapshot(depth, "mesh_multi_depth.npy", 3e-3)


def test_multi_mesh_render_is_tile_invariant():
    expected_image, expected_depth = render_multi_mesh()
    actual_image, actual_depth = render_multi_mesh((24, 24), (2, 4), 13)
    assert compute_max_abs_difference(actual_image, expected_image) <= 1e-4
    assert compute_max_abs_difference(actual_depth, expected_depth) <= 1e-4


def test_multi_mesh_render_masks_matches_snapshot():
    shape, y_FOV, pose, scene, _, lights = make_multi_mesh_scene()
    args = shape, y_FOV, pose, scene, lights, (0.1, 10.0), (1, 1), 1024
    masks = paz.graphics.render_masks(*args)
    assert masks.shape == (2, 24, 24, 1)
    assert_snapshot(masks, "mesh_multi_masks.npy", 1e-3)


def test_multi_mesh_gradient_matches_snapshot():
    shape, y_FOV, pose, _, _, lights = make_multi_mesh_scene((12, 12))
    cube, sphere = build_cube_mesh(), build_sphere_mesh()

    def loss_fn(vertices):
        moved = cube._replace(vertices=vertices)
        scene, mask = build_scene(moved, sphere)
        args = shape, y_FOV, pose, scene, mask, lights
        return jp.sum(render(*args, (1, 1), 1024)[0])

    gradient = jax.grad(loss_fn)(cube.vertices)
    assert jp.any(gradient != 0.0)
    assert_snapshot(gradient, "mesh_multi_gradient.npy", 2e-3)


def build_checkered_image(box_size=4, rows=4, cols=4):
    checkered = jp.indices((rows, cols)).sum(axis=0) % 2
    tiled = jp.kron(checkered, jp.ones((box_size, box_size)))
    channels = [0.9 * tiled, 0.45 * tiled, 0.8 * (1.0 - tiled)]
    return jp.stack(channels, axis=-1)


def build_textured_quad_mesh():
    vertices = jp.array([
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ])
    faces = jp.array([[0, 1, 2], [0, 2, 3]])
    edges = jp.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    vertex_uvs = jp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    pattern = Pattern(jp.eye(4), NO_PATTERN, build_checkered_image())
    colors = build_vertex_colors(vertices, [0.0, 0.0, 0.0])
    material = Material(jp.zeros(3), 0.2, 0.9, 0.0, 100)
    args = vertices, colors, jp.eye(4), material, faces, edges
    return Mesh(*args, pattern, vertex_uvs)


def make_textured_quad_scene(image_shape=(24, 24)):
    camera_origin = jp.array([0.0, 0.0, -3.0])
    camera_pose = SE3.view_transform(
        camera_origin, jp.zeros(3), jp.array([0.0, 1.0, 0.0])
    )
    lights = [PointLight(jp.full((3,), 0.7), camera_origin)]
    scene, mask = build_scene(build_textured_quad_mesh())
    return image_shape, jp.pi / 4.0, camera_pose, scene, mask, lights


def test_textured_quad_render_matches_snapshot():
    image, _ = render(*make_textured_quad_scene(), (1, 1), 1024)
    assert_snapshot(image, "mesh_uv_texture_image.npy", 1e-3)


def test_textured_quad_samples_texture_not_vertex_colors():
    image, _ = render(*make_textured_quad_scene(), (1, 1), 1024)
    center_rows = image[8:16, 8:16]
    assert float(jp.max(center_rows) - jp.min(center_rows)) > 0.1


def make_tile_scene(image_shape=(20, 20), tiles=(2, 2), chunk_size=1024):
    camera_origin = jp.array([0.0, 1.0, -1.5])
    y_FOV = jp.pi / 4.0
    camera_pose = SE3.view_transform(
        camera_origin, jp.zeros(3), jp.array([0.0, 0.0, 1.0])
    )
    lights = [PointLight(jp.full((3,), 10.0), camera_origin)]
    vertices, faces, edges = build_cube(1.0)
    color = jp.array([[0.7, 0.3, 0.1]])
    vertex_colors = jp.repeat(color, len(vertices), axis=0)
    transform = SE3.translation(jp.zeros(3))
    material = Material(jp.zeros(3), 0.1, 0.9, 0.1, 100)
    mesh = Mesh(vertices, vertex_colors, transform, material, faces, edges)
    scene, mask = build_scene(mesh)
    args = image_shape, y_FOV, camera_pose, scene, mask, lights
    return args + (tiles, chunk_size)


def test_render_rect_tiles_returns_correct_shapes():
    args = make_tile_scene((20, 20))
    image, depth = render(*args)
    assert image.shape == (20, 20, 3)
    assert depth.shape == (20, 20)


def test_render_rect_tiles_produces_nonzero_image():
    args = make_tile_scene()
    image, depth = render(*args)
    assert jp.any(image < 1.0)


def test_render_rect_tiles_jit_compatible():
    args = make_tile_scene()
    render_fn = jax.jit(render, static_argnums=(0, 6, 7))
    image, depth = render_fn(*args)
    assert image.shape == (20, 20, 3)


def test_render_rect_depth_matches_single_tile():
    expected = make_tile_scene((20, 20), (1, 1), 1024)
    actual = make_tile_scene((20, 20), (2, 2), 1024)
    _, expected_depth = render(*expected)
    _, actual_depth = render(*actual)
    assert actual_depth.shape == expected_depth.shape
    assert jp.allclose(actual_depth, expected_depth, atol=1e-5)


# Gradient faithfulness of object pose for analysis-by-synthesis fitting.
# The mesh is a non-spherical ellipsoid so rotation changes the silhouette.
# sigma is kept small so the soft-mask blur radius stays sub-pixel and the
# FACES_PER_PIXEL fragment cap never truncates contributing faces.
POSE_FIT_SIGMA = 1e-3


def make_ellipsoid_mesh(transform):
    vertices, faces, edges = build_sphere(0.6, 2)
    vertices = vertices * jp.array([1.0, 1.8, 1.0])
    color = jp.repeat(jp.array([[0.7, 0.3, 0.1]]), len(vertices), axis=0)
    material = Material(jp.zeros(3), 0.1, 0.9, 0.1, 100)
    return Mesh(vertices, color, transform, material, faces, edges)


def make_pose_camera():
    camera_origin = jp.array([0.0, 0.0, 3.0])
    world_up = jp.array([0.0, 1.0, 0.0])
    pose = SE3.view_transform(camera_origin, jp.zeros(3), world_up)
    return camera_origin, pose


def render_pose_soft_mask(transform):
    _, pose = make_pose_camera()
    mesh = make_ellipsoid_mesh(transform)
    bins = BinArgs(8, mesh.faces.shape[0])
    args = (bins, jp.pi / 3.0, 32, 32, pose, mesh, POSE_FIT_SIGMA, 256)
    return tile_render_binned_soft_mask(*args)


def render_pose_depth(transform):
    camera_origin, pose = make_pose_camera()
    mesh = make_ellipsoid_mesh(transform)
    scene, mask = build_scene(mesh)
    lights = [PointLight(jp.full(3, 10.0), camera_origin)]
    args = (32, 32), jp.pi / 3.0, pose, scene, mask, lights
    return render(*args, (1, 1), 1024)[1]


def x_translation(value):
    return SE3.translation(jp.array([value, 0.0, 0.0]))


def z_rotation(angle):
    rotation = SO3.exp(SO3.hat(jp.array([0.0, 0.0, angle])))
    return SE3.to_affine_matrix(rotation, jp.zeros(3))


def central_difference(loss_fn, value, epsilon=1e-3):
    return (loss_fn(value + epsilon) - loss_fn(value - epsilon)) / (2 * epsilon)


def mse_to_target_loss(render_fn, build_transform, target):
    return lambda x: jp.mean((render_fn(build_transform(x)) - target) ** 2)


def assert_matches_central_difference(render_fn, build_transform, offset):
    target = render_fn(build_transform(offset))
    loss_fn = mse_to_target_loss(render_fn, build_transform, target)
    gradient = jax.grad(loss_fn)(0.0)
    finite = central_difference(loss_fn, 0.0)
    assert jp.abs(gradient) > 1e-4
    assert jp.allclose(gradient, finite, rtol=0.05)
    return gradient


def test_soft_mask_translation_gradient_matches_finite_difference():
    assert_matches_central_difference(render_pose_soft_mask, x_translation, 0.3)


def test_soft_mask_rotation_gradient_matches_finite_difference():
    gradient = assert_matches_central_difference(
        render_pose_soft_mask, z_rotation, 0.3
    )
    assert jp.abs(gradient) > 1e-3


def test_depth_translation_gradient_matches_finite_difference():
    assert_matches_central_difference(render_pose_depth, x_translation, 0.3)


def test_depth_rotation_gradient_matches_finite_difference():
    assert_matches_central_difference(render_pose_depth, z_rotation, 0.3)


def render_scene_meshes(image_shape=(24, 24), tiles=(1, 1), chunk_size=1024):
    shape, y_FOV, pose, _, _, lights = make_multi_mesh_scene(image_shape)
    scene = paz.graphics.Scene([build_cube_mesh(), build_sphere_mesh()])
    args = shape, y_FOV, pose, scene, None, lights
    return paz.graphics.render(*args, tiles, chunk_size)


def test_scene_meshes_render_returns_correct_shapes():
    image, depth = render_scene_meshes()
    assert image.shape == (24, 24, 3)
    assert depth.shape == (24, 24)


def test_scene_meshes_cover_same_pixels_as_mesh_render():
    _, expected_depth = render_multi_mesh()
    _, actual_depth = render_scene_meshes()
    assert jp.array_equal(actual_depth > 0, expected_depth > 0)


def test_scene_meshes_match_mesh_render_within_precision():
    expected_image, expected_depth = render_multi_mesh()
    actual_image, actual_depth = render_scene_meshes()
    assert compute_max_abs_difference(actual_depth, expected_depth) <= 5e-2
    assert compute_max_abs_difference(actual_image, expected_image) <= 1e-1


def test_scene_meshes_render_is_tile_invariant():
    expected_image, expected_depth = render_scene_meshes()
    actual_image, actual_depth = render_scene_meshes((24, 24), (2, 4), 13)
    assert compute_max_abs_difference(actual_image, expected_image) <= 1e-4
    assert compute_max_abs_difference(actual_depth, expected_depth) <= 1e-4


def test_scene_meshes_respect_mask():
    shape, y_FOV, pose, _, _, lights = make_multi_mesh_scene()
    scene = paz.graphics.Scene([build_cube_mesh(), build_sphere_mesh()])
    mask = jp.zeros(2, dtype=bool)
    args = shape, y_FOV, pose, scene, mask, lights
    _, depth = paz.graphics.render(*args, (1, 1), 1024)
    assert jp.allclose(depth, 0.0)


def test_scene_mixes_meshes_and_shapes():
    shape, y_FOV, pose, _, _, lights = make_multi_mesh_scene()
    sphere = paz.graphics.Sphere(SE3.translation(jp.array([0.0, 0.0, 2.0])))
    scene = paz.graphics.Scene([build_cube_mesh(), sphere])
    args = shape, y_FOV, pose, scene, None, lights
    image, depth = paz.graphics.render(*args, (1, 1), 1024)
    assert image.shape == (24, 24, 3)
    assert jp.any(depth > 0)


def build_cube_mesh_with_material(material):
    vertices, faces, edges = build_cube(1.0)
    colors = build_vertex_colors(vertices, [0.7, 0.3, 0.1])
    transform = SE3.translation(jp.array([-0.45, 0.0, 0.0]))
    args = vertices, colors, transform, material, faces, edges
    return Mesh(*args)


def render_mesh_with_material(material):
    shape, y_FOV, pose, _, _, lights = make_multi_mesh_scene()
    mesh = build_cube_mesh_with_material(material)
    scene = paz.graphics.Scene([mesh, build_sphere_mesh()])
    args = shape, y_FOV, pose, scene, None, lights, (1, 1), 1024
    return paz.graphics.render(*args, False, None, 2)


def test_mesh_reflective_material_changes_render():
    matte = Material(jp.zeros(3), 0.1, 0.9, 0.1, 100, 0.0)
    mirror = Material(jp.zeros(3), 0.1, 0.9, 0.1, 100, 0.9)
    matte_image, _ = render_mesh_with_material(matte)
    mirror_image, _ = render_mesh_with_material(mirror)
    assert compute_max_abs_difference(matte_image, mirror_image) > 1e-3


def test_mesh_transparent_material_changes_render():
    opaque = Material(jp.zeros(3), 0.1, 0.9, 0.1, 100, 0.0, 0.0)
    glass = Material(jp.zeros(3), 0.1, 0.9, 0.1, 100, 0.0, 0.8, 1.5)
    opaque_image, _ = render_mesh_with_material(opaque)
    glass_image, _ = render_mesh_with_material(glass)
    assert compute_max_abs_difference(opaque_image, glass_image) > 1e-3


def build_mesh_over_floor_scene():
    vertices, faces, edges = build_cube(1.0)
    colors = build_vertex_colors(vertices, [0.7, 0.3, 0.1])
    material = Material(jp.zeros(3), 0.1, 0.9, 0.1, 100)
    transform = SE3.translation(jp.array([0.0, 1.0, 0.0]))
    args = vertices, colors, transform, material, faces, edges
    floor = paz.graphics.Plane(SE3.translation(jp.array([0.0, -0.5, 0.0])))
    return paz.graphics.Scene([Mesh(*args), floor])


def render_mesh_over_floor(shadows, shadow_mask=None):
    camera_pose = SE3.view_transform(
        jp.array([0.0, 2.0, -4.0]), jp.zeros(3), jp.array([0.0, 1.0, 0.0])
    )
    lights = [PointLight(jp.ones(3), jp.array([0.0, 5.0, -1.0]))]
    scene = build_mesh_over_floor_scene()
    args = (32, 32), jp.pi / 3.0, camera_pose, scene, None, lights
    return paz.graphics.render(*args, (1, 1), 1024, shadows, shadow_mask)


def test_mesh_casts_shadow_on_shape():
    lit, _ = render_mesh_over_floor(False)
    shadowed, _ = render_mesh_over_floor(True)
    assert compute_max_abs_difference(lit, shadowed) > 1e-2
    assert jp.all(shadowed <= lit + 1e-4)


def test_mesh_shadow_mask_stops_mesh_casting():
    casting, _ = render_mesh_over_floor(True)
    blocked, _ = render_mesh_over_floor(True, jp.array([False, True]))
    assert compute_max_abs_difference(casting, blocked) > 1e-2
    assert jp.all(casting <= blocked + 1e-4)


def build_mesh_only_shadow_scene():
    vertices, faces, edges = build_cube(1.0)
    colors = build_vertex_colors(vertices, [0.7, 0.4, 0.2])
    material = Material(jp.zeros(3), 0.1, 0.9, 0.1, 100)
    blocker_pose = SE3.translation(jp.array([0.0, 1.4, 0.0]))
    blocker = Mesh(vertices, colors, blocker_pose, material, faces, edges)
    slab_pose = SE3.translation(jp.array([0.0, -0.6, 0.0]))
    slab_pose = slab_pose @ SE3.scaling(jp.array([6.0, 0.2, 6.0]))
    slab = Mesh(vertices, colors, slab_pose, material, faces, edges)
    return paz.graphics.Scene([blocker, slab])


def render_mesh_only_shadow(shadows):
    camera_pose = SE3.view_transform(
        jp.array([0.0, 2.4, -5.0]), jp.zeros(3), jp.array([0.0, 1.0, 0.0])
    )
    lights = [PointLight(jp.ones(3), jp.array([0.0, 6.0, -1.5]))]
    scene = build_mesh_only_shadow_scene()
    args = (32, 32), jp.pi / 3.0, camera_pose, scene, None, lights
    return paz.graphics.render(*args, (1, 1), 1024, shadows)


def test_mesh_receives_shadow_from_mesh():
    lit, _ = render_mesh_only_shadow(False)
    shadowed, _ = render_mesh_only_shadow(True)
    assert compute_max_abs_difference(lit, shadowed) > 1e-2
    assert jp.all(shadowed <= lit + 1e-4)


def test_scene_rejects_meshes_with_mixed_pattern_sizes():
    plain = build_cube_mesh()
    textured = build_textured_quad_mesh()
    with pytest.raises(ValueError, match="pattern images"):
        scene = paz.graphics.Scene([plain, textured])
        paz.graphics.scene.compile(scene, [], None)


def build_cook_torrance_mesh():
    vertices, faces, edges = build_cube(1.0)
    colors = build_vertex_colors(vertices, [0.7, 0.3, 0.1])
    material = paz.graphics.CookTorranceMaterial(
        jp.zeros(3), 0.1, 0.04, 0.4, 0.0
    )
    args = vertices, colors, jp.eye(4), material, faces, edges
    return Mesh(*args)


def render_cook_torrance_scene(meshes):
    shape, y_FOV, pose, _, _, lights = make_multi_mesh_scene()
    scene = paz.graphics.Scene(list(meshes))
    args = shape, y_FOV, pose, scene, None, lights
    return paz.graphics.render(*args, (1, 1), 1024)


def test_cook_torrance_mesh_renders():
    image, depth = render_cook_torrance_scene([build_cook_torrance_mesh()])
    assert image.shape == (24, 24, 3)
    assert jp.any(depth > 0.0)
    assert jp.all(jp.isfinite(image))


def test_cook_torrance_mesh_roughness_changes_shading():
    smooth = build_cook_torrance_mesh()
    rough = smooth._replace(material=smooth.material._replace(roughness=0.9))
    smooth_image, _ = render_cook_torrance_scene([smooth])
    rough_image, _ = render_cook_torrance_scene([rough])
    assert compute_max_abs_difference(smooth_image, rough_image) > 1e-3


def test_scene_rejects_meshes_with_mixed_material_types():
    phong = build_cube_mesh()
    with pytest.raises(ValueError, match="must all be of one type"):
        scene = paz.graphics.Scene([phong, build_cook_torrance_mesh()])
        paz.graphics.scene.compile(scene, [], None)
