import jax
import jax.numpy as jp
import pytest
import paz
from paz.graphics.constants import FARAWAY
from paz.graphics.types import PointLight, Material
from paz.backend.lie import SE3
from paz.graphics.camera import build_rays
from paz.graphics.mesh.silhouette import blend_fragments
from paz.graphics.mesh.silhouette import build_empty_fragments
from paz.graphics.mesh.silhouette import compute_face_fragments
from paz.graphics.mesh.silhouette import merge_fragments
from paz.graphics.mesh import (
    Mesh,
    render,
    render_mesh,
    merge_meshes,
    extract_points,
    build_edges,
    compute_canonical_normals,
    compute_position,
    transform_points,
    intersect_canonical_mesh,
    vertex_colors_to_face_colors,
    compute_base_color,
    compute_ambient,
    select_closest_color,
    to_color_image,
    to_depth_image,
    render_depth,
    fill_bottom_with_last,
    fill_mesh,
    build_cube,
    build_sphere,
    tile_render,
    tile_render_depth,
    render_soft_mask,
    tile_render_soft_mask,
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


def build_legacy_rays(image_shape, y_fov, world_to_camera):
    H, W = image_shape[:2]
    aspect_ratio = paz.graphics.camera.compute_aspect_ratio(H, W)
    H_world, W_world = paz.graphics.camera.compute_image_sizes(
        y_fov, aspect_ratio
    )
    args = (H, W, H_world, W_world)
    directions = paz.graphics.camera.build_ray_directions(*args)
    origins = paz.graphics.camera.build_ray_origins(H, W)
    camera_to_world = jp.linalg.inv(world_to_camera)
    return paz.algebra.transform_rays(camera_to_world, origins, directions)


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


def test_vertex_colors_to_face_colors():
    faces = jp.array([[0, 1, 2]])
    colors = jp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    face_colors = vertex_colors_to_face_colors(faces, colors)
    expected = jp.array([[[1.0 / 3, 1.0 / 3, 1.0 / 3]]])
    assert jp.allclose(face_colors, expected, atol=1e-5)


def test_compute_base_color():
    color = jp.array([[[0.5, 0.5, 0.5]]])
    intensity = jp.array([2.0, 2.0, 2.0])
    result = compute_base_color(color, intensity, None)
    assert jp.allclose(result, jp.array([[[1.0, 1.0, 1.0]]]))


def test_compute_ambient():
    color = jp.array([[[1.0, 1.0, 1.0]]])
    light = PointLight(jp.ones(3), jp.zeros(3))
    points = jp.zeros((1, 1, 3))
    result = compute_ambient(0.1, color, light, points)
    assert jp.allclose(result, 0.1, atol=1e-5)


def test_select_closest_color():
    depths = jp.array([[10.0, 20.0], [5.0, 30.0]])
    colors = jp.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]])
    result = select_closest_color(depths, colors)
    assert jp.allclose(result[0], jp.array([0.0, 0.0, 1.0]))
    assert jp.allclose(result[1], jp.array([0.0, 1.0, 0.0]))


def test_to_color_image_shape():
    hit_mask = jp.array([True, True, False, False])
    colors = jp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
    image = to_color_image(hit_mask, colors, 2, 2)
    assert image.shape == (2, 2, 3)


def test_to_color_image_background():
    hit_mask = jp.array([False])
    colors = jp.array([[0.5, 0.5, 0.5]])
    image = to_color_image(hit_mask, colors, 1, 1)
    assert jp.allclose(image[0, 0], jp.array([1.0, 1.0, 1.0]))


def test_to_depth_image_shape():
    hit_mask = jp.array([True, True, False, False])
    depths = jp.array([1.0, 2.0, FARAWAY, FARAWAY])
    rays = (jp.zeros((4, 3)), jp.tile(jp.array([[0, 0, -1.0]]), (4, 1)))
    image = to_depth_image(hit_mask, depths, jp.eye(4), rays, 2, 2)
    assert image.shape == (2, 2)


def test_fill_bottom_with_last():
    x = jp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = fill_bottom_with_last(x, 5)
    assert result.shape == (5, 2)
    assert jp.allclose(result[3], x[1])
    assert jp.allclose(result[4], x[1])


def test_fill_mesh():
    vertices = jp.zeros((3, 3))
    faces = jp.zeros((2, 3), dtype=int)
    edges = jp.zeros((4, 2), dtype=int)
    vertex_colors = jp.ones((3, 3))
    material = Material(jp.zeros(3), 0.1, 0.1, 0.1, 100)
    mesh = Mesh(vertices, vertex_colors, jp.eye(4), material, faces, edges)
    filled = fill_mesh(mesh, 5, 4, 6)
    assert filled.vertices.shape == (5, 3)
    assert filled.faces.shape == (4, 3)
    assert filled.edges.shape == (6, 2)


def test_merge_meshes():
    vertices = jp.zeros((3, 3))
    faces = jp.zeros((2, 3), dtype=int)
    edges = jp.zeros((4, 2), dtype=int)
    colors = jp.ones((3, 3))
    material = Material(jp.zeros(3), 0.1, 0.1, 0.1, 100)
    mesh_a = Mesh(vertices, colors, jp.eye(4), material, faces, edges)
    mesh_b = Mesh(vertices, colors, jp.eye(4), material, faces, edges)
    batched, mask = merge_meshes(mesh_a, mesh_b)
    assert batched.vertices.shape == (2, 3, 3)
    assert mask.shape == (2,)
    assert jp.all(mask)


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
    rays = build_rays(image_shape, y_FOV, camera_pose)
    lights = [PointLight(jp.full((3,), 10.0), camera_origin)]
    vertices, faces, edges = build_cube(1.0)
    color = jp.array([[0.7, 0.3, 0.1]])
    vertex_colors = jp.repeat(color, len(vertices), axis=0)
    transform = SE3.translation(jp.zeros(3))
    material = Material(jp.zeros(3), 0.1, 0.9, 0.1, 100)
    mesh = Mesh(vertices, vertex_colors, transform, material, faces, edges)
    meshes, mask = merge_meshes(mesh)
    return image_shape, camera_pose, rays, meshes, mask, lights


def test_render_returns_correct_shapes():
    image_shape = (20, 20)
    args = make_scene(image_shape)
    image, depth = render(*args)
    assert image.shape == (20, 20, 3)
    assert depth.shape == (20, 20)


def test_render_produces_nonzero_image():
    args = make_scene()
    image, depth = render(*args)
    assert jp.any(image < 1.0)


def test_render_jit_compatible():
    args = make_scene()
    render_fn = jax.jit(render, static_argnums=(0,))
    image, depth = render_fn(*args)
    assert image.shape == (20, 20, 3)


def test_render_depth_matches_render_depth():
    args = make_scene()
    _, expected_depth = render(*args)
    depth = render_depth(*args)
    assert depth.shape == expected_depth.shape
    assert jp.allclose(depth, expected_depth, atol=1e-5)


def test_render_matches_legacy_rays():
    image_shape, camera_pose, rays, meshes, mask, lights = make_scene()
    legacy_rays = build_legacy_rays(image_shape, jp.pi / 4.0, camera_pose)
    image, depth = render(image_shape, camera_pose, rays, meshes, mask, lights)
    legacy_image, legacy_depth = render(
        image_shape, camera_pose, legacy_rays, meshes, mask, lights
    )
    assert compute_max_abs_difference(image, legacy_image) <= 1e-4
    assert compute_max_abs_difference(depth, legacy_depth) <= 1e-4


def test_render_depth_matches_legacy_rays():
    image_shape, camera_pose, _, meshes, mask, lights = make_scene()
    rays = build_rays(image_shape, jp.pi / 4.0, camera_pose)
    legacy_rays = build_legacy_rays(image_shape, jp.pi / 4.0, camera_pose)
    depth = render_depth(image_shape, camera_pose, rays, meshes, mask, lights)
    legacy_depth = render_depth(
        image_shape, camera_pose, legacy_rays, meshes, mask, lights
    )
    assert compute_max_abs_difference(depth, legacy_depth) <= 1e-4


def test_render_depth_respects_mask():
    image_shape, camera_pose, rays, meshes, mask, lights = make_scene()
    mask = jp.zeros_like(mask).astype(bool)
    depth = render_depth(image_shape, camera_pose, rays, meshes, mask, lights)
    assert jp.allclose(depth, 0.0)


def test_render_gradient_through_vertices():
    camera_origin = jp.array([0.0, 1.0, -1.5])
    y_FOV = jp.pi / 4.0
    image_shape = (10, 10)
    camera_pose = SE3.view_transform(
        camera_origin, jp.zeros(3), jp.array([0.0, 0.0, 1.0])
    )
    rays = build_rays(image_shape, y_FOV, camera_pose)
    lights = [PointLight(jp.full((3,), 10.0), camera_origin)]
    vertices, faces, edges = build_cube(1.0)
    color = jp.array([[0.7, 0.3, 0.1]])
    vertex_colors = jp.repeat(color, len(vertices), axis=0)
    transform = SE3.translation(jp.zeros(3))
    material = Material(jp.zeros(3), 0.1, 0.9, 0.1, 100)

    def loss_fn(verts):
        mesh = Mesh(verts, vertex_colors, transform, material, faces, edges)
        meshes, mask = merge_meshes(mesh)
        image, _ = render(image_shape, camera_pose, rays, meshes, mask, lights)
        return jp.sum(image)

    grad = jax.grad(loss_fn)(vertices)
    assert grad.shape == vertices.shape
    assert jp.any(grad != 0.0)


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


def make_tile_scene(image_shape=(20, 20)):
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
    meshes, mask = merge_meshes(mesh)
    H, W = image_shape
    return (2, 2), y_FOV, H, W, camera_pose, meshes, mask, lights


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


def test_tile_render_returns_correct_shapes():
    args = make_tile_scene((20, 20))
    image, depth = tile_render(*args)
    assert image.shape == (20, 20, 3)
    assert depth.shape == (20, 20)


def test_tile_render_produces_nonzero_image():
    args = make_tile_scene()
    image, depth = tile_render(*args)
    assert jp.any(image < 1.0)


def test_tile_render_jit_compatible():
    args = make_tile_scene()
    render_fn = jax.jit(tile_render, static_argnums=(0, 2, 3))
    image, depth = render_fn(*args)
    assert image.shape == (20, 20, 3)


def test_tile_render_depth_matches_render_depth():
    scene = make_tile_scene()
    tile_shape, y_FOV, H, W, camera_pose, meshes, mask, lights = scene
    rays = build_rays((H, W), y_FOV, camera_pose)
    expected_depth = render_depth(
        (H, W), camera_pose, rays, meshes, mask, lights
    )
    depth = tile_render_depth(
        tile_shape, y_FOV, H, W, camera_pose, meshes, mask, lights
    )
    assert depth.shape == expected_depth.shape
    assert jp.allclose(depth, expected_depth, atol=1e-5)


def test_render_soft_mask_returns_smooth_square():
    mesh = make_soft_square_mesh()
    mask = render_soft_mask(*build_soft_shift_args(0.0))
    assert mask.shape == (16, 16)
    assert mask[8, 8] > 0.7
    assert mask[0, 0] < 0.1


def test_tile_render_soft_mask_matches_untiled():
    mesh = make_soft_square_mesh()
    args = ((16, 16), jp.eye(4), mesh, jp.pi / 3.0, 1e-4, 2)
    expected = render_soft_mask(*args)
    args = ((2, 2), jp.pi / 3.0, 16, 16, jp.eye(4), mesh, 1e-4, 2)
    actual = tile_render_soft_mask(*args)
    assert jp.allclose(actual, expected, atol=1e-5)


def test_render_soft_mask_is_chunk_invariant():
    mesh = make_soft_square_mesh()
    args = ((16, 16), jp.eye(4), mesh, jp.pi / 3.0, 1e-4)
    mask_A = render_soft_mask(*(args + (1,)))
    mask_B = render_soft_mask(*(args + (2,)))
    assert jp.allclose(mask_A, mask_B, atol=1e-5)


def test_render_soft_mask_shift_gradient_matches_finite_difference():
    target = render_soft_mask(*build_soft_shift_args(0.25))

    def loss_fn(shift):
        prediction = render_soft_mask(*build_soft_shift_args(shift[0]))
        return jp.mean((prediction - target) ** 2)

    _, gradient = jax.value_and_grad(loss_fn)(jp.array([0.0]))
    finite = compute_finite_shift_gradient(loss_fn, jp.array([0.0]))
    cosine = gradient[0] * finite / (jp.abs(gradient[0] * finite) + 1e-8)
    assert jp.abs(gradient[0]) > 1e-5
    assert cosine > 0.9


def build_soft_shift_args(shift):
    mesh = make_soft_square_mesh(shift)
    return (16, 16), jp.eye(4), mesh, jp.pi / 3.0, 1e-4, 2


def compute_finite_shift_gradient(loss_fn, shift):
    epsilon = 1e-2
    high = loss_fn(shift + jp.array([epsilon]))
    low = loss_fn(shift - jp.array([epsilon]))
    return (high - low) / (2.0 * epsilon)
