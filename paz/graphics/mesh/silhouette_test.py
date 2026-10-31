from pathlib import Path

import jax
import jax.numpy as jp

import paz
from paz.graphics.types import Material, Mesh
from paz.graphics.mesh.silhouette import BinArgs
from paz.graphics.mesh.silhouette import Projection
from paz.graphics.mesh.silhouette import blend_fragments
from paz.graphics.mesh.silhouette import build_empty_fragments
from paz.graphics.mesh.silhouette import compute_face_fragments
from paz.graphics.mesh.silhouette import count_binned_faces
from paz.graphics.mesh.silhouette import fragment_chunk_or_empty_step
from paz.graphics.mesh.silhouette import merge_fragments
from paz.graphics.mesh.silhouette import project_to_pixels
from paz.graphics.mesh.silhouette import tile_render_binned_soft_mask


def build_intrinsics(H, W):
    return paz.graphics.camera.compute_intrinsics(jp.pi / 3.0, H, W)


def snapshot_path(filename):
    return str(Path(__file__).parent.parent / "snapshots" / filename)


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


def render_binned_soft_shift(shift, image_shape=16, bin_shape=8, chunk=2):
    H, W = unpack_soft_shape(image_shape)
    mesh = make_soft_square_mesh(shift)
    bins = BinArgs(bin_shape, mesh.faces.shape[0])
    args = (bins, build_intrinsics(H, W), H, W, jp.eye(4), mesh)
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
    intrinsics = build_intrinsics(16, 16)
    args = ((16, 16), jp.eye(4), mesh, intrinsics, 1e-4, BinArgs(8, 2))
    counts = count_binned_faces(*args)
    assert jp.max(counts) == 2


def test_count_binned_faces_counts_rect_overlaps():
    mesh = make_soft_square_mesh()
    args = ((16, 24), jp.eye(4), mesh, build_intrinsics(16, 24), 1e-4)
    counts = count_binned_faces(*(args + (BinArgs((8, 6), 2),)))
    assert jp.max(counts) == 2


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


def test_project_to_pixels_inverts_pixel_rays():
    rows = [[611.1, 0.0, 315.5], [0.0, 610.0, 237.7], [0.0, 0.0, 1.0]]
    intrinsics = jp.array(rows)
    pixels = jp.array([[0.0, 0.0], [100.0, 37.0], [639.0, 479.0]])
    args = intrinsics, pixels, jp.eye(4)
    _, directions = paz.graphics.camera.build_pixel_rays(*args)
    vertices = directions * (2.5 / -directions[:, 2:])
    recovered, depth = project_to_pixels(vertices, intrinsics)
    assert jp.allclose(recovered, pixels, atol=1e-3)
    assert jp.allclose(depth, 2.5, atol=1e-5)


def soft_square_loss(vertices, target):
    mesh = make_soft_square_mesh()._replace(vertices=vertices)
    bins = BinArgs(8, mesh.faces.shape[0])
    args = (bins, build_intrinsics(16, 16), 16, 16, jp.eye(4), mesh, 1e-4, 2)
    return jp.mean((tile_render_binned_soft_mask(*args) - target) ** 2)


def test_tile_render_binned_soft_mask_matches_snapshot():
    mask = render_binned_soft_shift(0.0)
    path = snapshot_path("silhouette_soft_mask.npy")
    paz.assert_snapshot(mask, path, atol=1e-5)


def test_tile_render_binned_vertex_gradient_matches_snapshot():
    target = render_binned_soft_shift(0.25)
    vertices = make_soft_square_mesh().vertices
    gradient = jax.grad(soft_square_loss)(vertices, target)
    path = snapshot_path("silhouette_soft_mask_gradient.npy")
    paz.assert_snapshot(gradient, path, atol=1e-6)
