import jax
import jax.numpy as jp
import pytest
import paz
from paz.graphics.constants import FARAWAY, NO_PATTERN
from paz.graphics.types import PointLight, Material, Pattern
from paz.backend.lie import SE3, SO3
from paz.graphics.mesh.silhouette import blend_fragments
from paz.graphics.mesh.silhouette import build_empty_fragments
from paz.graphics.mesh.silhouette import compute_face_fragments
from paz.graphics.mesh.silhouette import fragment_chunk_or_empty_step
from paz.graphics.mesh.silhouette import merge_fragments
from paz.graphics.mesh.silhouette import Projection
from paz.graphics.mesh import (
    Mesh,
    BinArgs,
    render_coordinates,
    extract_points,
    build_edges,
    compute_canonical_normals,
    compute_position,
    transform_points,
    intersect_canonical_mesh,
    build_cube,
    build_sphere,
    tile_render_binned_soft_mask,
    count_binned_faces,
    assert_exact_tile_side,
    make_tile_coordinates,
    make_ray_origins,
    make_ray_targets,
    transform_tile_rays,
    assemble,
)


def make_triangle():
    vertices = jp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    faces = jp.array([[0, 2, 1]])
    return vertices, faces


def build_scene(*meshes):
    return paz.graphics.Scene(list(meshes)), None


def render(*args):
    return paz.graphics.render(*args)


def compute_max_abs_difference(array_A, array_B):
    return float(jp.max(jp.abs(array_A - array_B)))


def test_extract_points():
    vertices, faces = make_triangle()
    A, B, C = extract_points(vertices, faces)
    assert jp.allclose(A[0], vertices[0])
    assert jp.allclose(B[0], vertices[2])
    assert jp.allclose(C[0], vertices[1])


def test_build_edges_shape():
    vertices, faces = make_triangle()
    edges_AC, edges_AB, points_A = build_edges(vertices, faces)
    assert edges_AC.shape == (1, 1, 3)
    assert edges_AB.shape == (1, 1, 3)
    assert points_A.shape == (1, 1, 3)


def test_intersect_canonical_mesh_hit():
    vertices, faces = make_triangle()
    origins = jp.array([[0.25, 0.25, -1.0]])
    directions = jp.array([[0.0, 0.0, 1.0]])
    hit_mask, depth, _, _ = intersect_canonical_mesh(
        vertices, faces, origins, directions
    )
    assert hit_mask[0, 0] == True
    assert jp.allclose(depth[0, 0], 1.0, atol=1e-5)


def test_intersect_canonical_mesh_miss():
    vertices, faces = make_triangle()
    origins = jp.array([[5.0, 5.0, -1.0]])
    directions = jp.array([[0.0, 0.0, 1.0]])
    hit_mask, depth, _, _ = intersect_canonical_mesh(
        vertices, faces, origins, directions
    )
    assert hit_mask[0, 0] == False
    assert jp.allclose(depth[0, 0], FARAWAY)


def test_intersect_canonical_mesh_miss_returns_faraway():
    vertices, faces = make_triangle()
    origins = jp.array([[-1.0, -1.0, -1.0]])
    directions = jp.array([[0.0, 0.0, 1.0]])
    args = (vertices, faces, origins, directions)
    _, depth, _, _ = intersect_canonical_mesh(*args)
    assert depth[0, 0] >= FARAWAY - 1.0


def test_intersect_canonical_mesh_rejects_negative_depth():
    vertices, faces = make_triangle()
    origins = jp.array([[0.25, 0.25, 1.0]])
    directions = jp.array([[0.0, 0.0, 1.0]])
    hit_mask, depth, _, _ = intersect_canonical_mesh(
        vertices, faces, origins, directions
    )
    assert hit_mask[0, 0] == False
    assert jp.allclose(depth[0, 0], FARAWAY)


def test_intersect_canonical_mesh_rejects_parallel_ray():
    vertices, faces = make_triangle()
    origins = jp.array([[0.25, 0.25, -1.0]])
    directions = jp.array([[1.0, 0.0, 0.0]])
    hit_mask, depth, _, _ = intersect_canonical_mesh(
        vertices, faces, origins, directions
    )
    assert hit_mask[0, 0] == False
    assert jp.allclose(depth[0, 0], FARAWAY)


def test_compute_canonical_normals_direction():
    vertices, faces = make_triangle()
    shape_points = jp.zeros((1, 4, 3))
    normals = compute_canonical_normals(vertices, faces, shape_points)
    assert normals.shape == (1, 4, 3)
    assert jp.abs(normals[0, 0, 2]) > 0.9


def test_compute_canonical_normals_floor_points_up():
    half = 2.0
    vertices = jp.array(
        [
            [-half, 0.0, -half],
            [half, 0.0, -half],
            [half, 0.0, half],
            [-half, 0.0, half],
        ]
    )
    faces = jp.array([[0, 2, 1], [0, 3, 2]])
    shape_points = jp.zeros((2, 1, 3))
    normals = compute_canonical_normals(vertices, faces, shape_points)
    assert jp.all(normals[:, 0, 1] > 0.9)


def test_compute_position_shape():
    origins = jp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    directions = jp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    depths = jp.array([[[2.0], [3.0]], [[4.0], [5.0]]])
    positions = compute_position(origins, directions, depths)
    assert positions.shape == (2, 2, 3)


def test_transform_points_shape():
    points = jp.ones((3, 4, 3))
    affine = jp.eye(4)
    result = transform_points(affine, points)
    assert result.shape == (3, 4, 3)


def test_transform_points_identity():
    points = jp.array([[[1.0, 2.0, 3.0]]])
    affine = jp.eye(4)
    result = transform_points(affine, points)
    assert jp.allclose(result, points, atol=1e-5)


def test_build_cube():
    vertices, faces, edges = build_cube(1.0)
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert edges.shape[1] == 2
    assert len(vertices) == 8
    assert len(faces) == 12


def face_centers_and_normals(vertices, faces):
    A = vertices[faces[:, 0]]
    B = vertices[faces[:, 1]]
    C = vertices[faces[:, 2]]
    centers = (A + B + C) / 3.0
    normals = jp.cross(B - A, C - A)
    return centers, normals


def test_build_cube_faces_point_outward():
    vertices, faces, _ = build_cube(1.0)
    centers, normals = face_centers_and_normals(vertices, faces)
    dots = jp.sum(centers * normals, axis=1)
    assert jp.all(dots > 0.0)


def test_build_sphere_faces_point_outward():
    vertices, faces, _ = build_sphere(1.0, 2)
    centers, normals = face_centers_and_normals(vertices, faces)
    dots = jp.sum(centers * normals, axis=1)
    assert jp.all(dots > 0.0)


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


def snapshot_path(filename):
    return f"paz/graphics/snapshots/{filename}"


def assert_snapshot(array, filename, atol):
    paz.assert_snapshot(array, snapshot_path(filename), atol=atol)


def build_vertex_colors(vertices, color):
    return jp.repeat(jp.array([color]), len(vertices), axis=0)


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


def test_assert_exact_tile_side_valid():
    assert_exact_tile_side(100, 2)
    assert_exact_tile_side(100, 5)
    assert_exact_tile_side(100, 10)


def test_assert_exact_tile_side_invalid():
    with pytest.raises(ValueError):
        assert_exact_tile_side(100, 3)


def test_make_tile_coordinates_shape():
    coords = make_tile_coordinates(3, 4)
    assert coords.shape == (12, 2)


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
    args = (jp.eye(4), origins, targets)
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


def make_soft_square_mesh(shift=0.0):
    half = 0.8
    z = -3.0
    vertices = jp.array([[-half, -half, z], [half, -half, z]])
    vertices = jp.vstack([vertices, jp.array([[half, half, z]])])
    vertices = jp.vstack([vertices, jp.array([[-half, half, z]])])
    vertices = vertices + jp.array([shift, 0.0, 0.0])
    faces = jp.array([[0, 1, 2], [0, 2, 3]])
    edges = jp.array([[0, 1], [1, 2], [2, 3], [0, 3], [0, 2]])
    color = jp.array([[0.7, 0.3, 0.1]])
    colors = jp.repeat(color, len(vertices), axis=0)
    material = Material(jp.zeros(3), 0.1, 0.9, 0.1, 100)
    return Mesh(vertices, colors, jp.eye(4), material, faces, edges)


def test_compute_face_fragments_signs_distances():
    points = jp.array([[[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]]])
    depths = jp.ones((1, 3))
    pixels = jp.array([[0.0, 0.0], [0.8, 0.0]])
    distances, _, valid = compute_face_fragments(points, depths, pixels, 1.0)
    assert distances[0, 0] < 0.0
    assert distances[0, 1] > 0.0
    assert jp.all(valid)


def test_blend_fragments_matches_sigmoid_alpha():
    distances = jp.array([[-0.1, 0.2]])
    valid = jp.array([[True, True]])
    alpha = blend_fragments(distances, valid, 0.1)
    probabilities = jax.nn.sigmoid(-distances / 0.1)
    expected = 1.0 - jp.prod(1.0 - probabilities, axis=1)
    assert jp.allclose(alpha, expected)


def test_merge_fragments_keeps_nearest_faces():
    fragments = build_empty_fragments(1)
    distances = jp.zeros((51, 1))
    depths = jp.arange(1, 52, dtype=jp.float32)[:, None]
    valid = jp.ones((51, 1), dtype=bool)
    fragments = merge_fragments(fragments, distances, depths, valid)
    assert jp.max(fragments.depths[0]) == 50.0
    assert jp.sum(fragments.valid[0]) == 50


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


def test_tile_render_binned_soft_mask_returns_smooth_square():
    mask = render_binned_soft_shift(0.0)
    assert mask.shape == (16, 16)
    assert mask[8, 8] > 0.7
    assert mask[0, 0] < 0.1


def test_tile_render_binned_soft_mask_matches_single_bin():
    expected = render_binned_soft_shift(0.0, 16, 16)
    actual = render_binned_soft_shift(0.0, 16, 8)
    assert jp.allclose(actual, expected, atol=1e-5)


def test_binned_soft_mask_matches_single_bin_with_empty_bins():
    expected = render_binned_soft_shift(0.0, 32, 32)
    actual = render_binned_soft_shift(0.0, 32, 8)
    assert jp.allclose(actual, expected, atol=1e-5)


def test_tile_render_binned_soft_mask_matches_rect_single_bin():
    expected = render_binned_soft_shift(0.0, (16, 24), (16, 24))
    actual = render_binned_soft_shift(0.0, (16, 24), (8, 6))
    assert jp.allclose(actual, expected, atol=1e-5)


def test_binned_soft_mask_matches_rect_bin_with_empty_bins():
    expected = render_binned_soft_shift(0.0, (32, 24), (32, 24))
    actual = render_binned_soft_shift(0.0, (32, 24), (8, 6))
    assert jp.allclose(actual, expected, atol=1e-5)


def test_empty_fragment_chunk_keeps_fragments():
    fragments = build_empty_fragments(1)
    points = jp.zeros((3, 2))
    projection = Projection(points, jp.ones(3))
    data = (jp.array([[0, 1, 2]]), jp.array([False]))
    pixels = jp.zeros((1, 2))
    args = (fragments, data, projection, pixels, 1.0)
    result, _ = fragment_chunk_or_empty_step(*args)
    assert jp.allclose(result.depths, fragments.depths)
    assert jp.allclose(result.distances, fragments.distances)
    assert jp.all(result.valid == fragments.valid)


def test_count_binned_faces_counts_overlaps():
    mesh = make_soft_square_mesh()
    args = ((16, 16), jp.eye(4), mesh, jp.pi / 3.0, 1e-4, BinArgs(8, 2))
    counts = count_binned_faces(*args)
    assert jp.max(counts) == 2


def test_count_binned_faces_counts_rect_overlaps():
    mesh = make_soft_square_mesh()
    args = ((16, 24), jp.eye(4), mesh, jp.pi / 3.0, 1e-4)
    counts = count_binned_faces(*(args + (BinArgs((8, 6), 2),)))
    assert jp.max(counts) == 2


def test_tile_render_binned_soft_mask_is_chunk_invariant():
    mask_A = render_binned_soft_shift(0.0, chunk=1)
    mask_B = render_binned_soft_shift(0.0, chunk=2)
    assert jp.allclose(mask_A, mask_B, atol=1e-5)


def test_tile_render_binned_rect_mask_is_chunk_invariant():
    mask_A = render_binned_soft_shift(0.0, (16, 24), (8, 6), 1)
    mask_B = render_binned_soft_shift(0.0, (16, 24), (8, 6), 2)
    assert jp.allclose(mask_A, mask_B, atol=1e-5)


def test_tile_render_binned_shift_gradient_matches_finite_difference():
    target = render_binned_soft_shift(0.25)

    def loss_fn(shift):
        prediction = render_binned_soft_shift(shift[0])
        return jp.mean((prediction - target) ** 2)

    _, gradient = jax.value_and_grad(loss_fn)(jp.array([0.0]))
    finite = compute_finite_shift_gradient(loss_fn, jp.array([0.0]))
    cosine = gradient[0] * finite / (jp.abs(gradient[0] * finite) + 1e-8)
    assert jp.abs(gradient[0]) > 1e-5
    assert cosine > 0.9


def render_binned_soft_shift(shift, image_shape=16, bin_shape=8, chunk=2):
    H, W = unpack_soft_shape(image_shape)
    mesh = make_soft_square_mesh(shift)
    bins = BinArgs(bin_shape, mesh.faces.shape[0])
    args = (bins, jp.pi / 3.0, H, W, jp.eye(4), mesh)
    args = args + (1e-4, chunk)
    return tile_render_binned_soft_mask(*args)


def unpack_soft_shape(shape):
    try:
        H, W = shape
    except TypeError:
        return shape, shape
    return H, W


def compute_finite_shift_gradient(loss_fn, shift):
    epsilon = 1e-2
    high = loss_fn(shift + jp.array([epsilon]))
    low = loss_fn(shift - jp.array([epsilon]))
    return (high - low) / (2.0 * epsilon)


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


def make_single_mesh():
    vertices, faces, edges = build_cube(1.0)
    color = jp.array([[0.7, 0.3, 0.1]])
    vertex_colors = jp.repeat(color, len(vertices), axis=0)
    transform = SE3.translation(jp.zeros(3))
    material = Material(jp.zeros(3), 0.1, 0.9, 0.1, 100)
    return Mesh(vertices, vertex_colors, transform, material, faces, edges)


def camera_looking_at_origin():
    camera_origin = jp.array([0.0, 1.0, -1.5])
    world_up = jp.array([0.0, 0.0, 1.0])
    return SE3.view_transform(camera_origin, jp.zeros(3), world_up)


def test_render_coordinates_shapes_and_object_frame_bounds():
    mesh = make_single_mesh()
    pose = camera_looking_at_origin()
    coordinates, hit = render_coordinates((20, 20), jp.pi / 4, pose, mesh, 1024)
    assert coordinates.shape == (20, 20, 3)
    assert hit.shape == (20, 20)
    assert jp.any(hit)
    inside = coordinates[hit]
    assert jp.all(inside >= -0.5 - 1e-4) and jp.all(inside <= 0.5 + 1e-4)
    assert jp.allclose(coordinates[~hit], 0.0)


def test_render_coordinates_mask_matches_render_depth():
    mesh = make_single_mesh()
    pose = camera_looking_at_origin()
    scene, mask = build_scene(mesh)
    lights = [PointLight(jp.full((3,), 10.0), jp.array([0.0, 1.0, -1.5]))]
    args = (20, 20), jp.pi / 4, pose, scene, mask, lights
    _, depth = render(*args, (1, 1), 1024)
    _, hit = render_coordinates((20, 20), jp.pi / 4, pose, mesh, 1024)
    assert jp.array_equal(hit, depth > 0)


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


def test_scene_rejects_meshes_with_mixed_pattern_sizes():
    plain = build_cube_mesh()
    textured = build_textured_quad_mesh()
    with pytest.raises(ValueError, match="pattern images"):
        paz.graphics.scene.compile(
            paz.graphics.Scene([plain, textured]), [], None
        )


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
