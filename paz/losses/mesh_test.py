import jax
import jax.numpy as jp
import paz

from paz.backend.mesh import build_laplacian
from paz.losses.mesh import laplacian_smoothing
from paz.losses.mesh import volume_matching


def legacy_smoothing(vertices, laplacian):
    return jp.trace(vertices.T @ laplacian @ vertices)


def legacy_volume(vertices, faces, target_volumes):
    current = jax.vmap(paz.mesh.compute_volume)(vertices, faces)
    return jp.mean((target_volumes - current) ** 2)


def test_laplacian_smoothing_matches_legacy_formula():
    vertices = jp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = jp.array([[0, 1, 2]], dtype=jp.int32)
    laplacian = build_laplacian(vertices, faces)
    result = laplacian_smoothing(vertices, laplacian)
    expected = legacy_smoothing(vertices, laplacian)
    assert jp.allclose(result, expected)


def test_laplacian_smoothing_batches_with_shared_laplacian():
    base = jp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    vertices = jp.stack((base, base * 2.0))
    faces = jp.array([[0, 1, 2]], dtype=jp.int32)
    laplacian = build_laplacian(base, faces)
    result = laplacian_smoothing(vertices, laplacian, reduction="none")
    expected = jax.vmap(legacy_smoothing, in_axes=(0, None))(vertices, laplacian)
    mean_value = laplacian_smoothing(vertices, laplacian)
    assert jp.allclose(result, expected)
    assert jp.allclose(mean_value, expected.mean())
    total = laplacian_smoothing(vertices, laplacian, reduction="sum")
    assert jp.allclose(total, expected.sum())


def test_volume_matching_matches_legacy_formula():
    vertices, faces, _ = paz.graphics.mesh.build_sphere(1.0, 1)
    batched_vertices = jp.stack((vertices, vertices * 0.9))
    batched_faces = jp.stack((faces, faces))
    targets = jax.vmap(paz.mesh.compute_volume)(batched_vertices, batched_faces)
    result = volume_matching(batched_vertices, batched_faces, targets)
    expected = legacy_volume(batched_vertices, batched_faces, targets)
    assert jp.allclose(result, expected)


def test_volume_matching_shares_faces_and_supports_reductions():
    vertices, faces, _ = paz.graphics.mesh.build_cube(2.0)
    batched_vertices = jp.stack((vertices, vertices * 0.5))
    targets = jp.array([8.0, 1.0])
    losses = volume_matching(batched_vertices, faces, targets, reduction="none")
    current = jax.vmap(
        paz.mesh.compute_volume, in_axes=(0, None))(batched_vertices, faces)
    expected = (targets - current) ** 2
    mean_value = volume_matching(batched_vertices, faces, targets)
    assert jp.allclose(losses, expected)
    assert jp.allclose(mean_value, expected.mean())
    total = volume_matching(batched_vertices, faces, targets, reduction="sum")
    assert jp.allclose(total, expected.sum())


def test_paz_losses_exports_mesh_regularizers():
    vertices, faces, _ = paz.graphics.mesh.build_cube(2.0)
    laplacian = build_laplacian(vertices, faces)
    smooth = paz.losses.laplacian_smoothing(vertices, laplacian)
    volume = paz.losses.volume_matching(vertices, faces, 8.0)
    assert jp.isfinite(smooth)
    assert jp.isfinite(volume)
