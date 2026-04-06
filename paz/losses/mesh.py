import jax
import jax.numpy as jp

from paz.backend.mesh import compute_volume
from paz.losses.standard import _reduce_loss


def _add_batch_axis(values, rank):
    values = jp.array(values)
    if values.ndim == rank:
        values = jp.expand_dims(values, axis=0)
    return values


def _match_batch_size(values, batch_size):
    # Reuse one shared value for every mesh in the batch.
    if len(values) == batch_size:
        return values
    if len(values) != 1:
        raise ValueError("Expected one value or one value per mesh.")
    shape = (batch_size,) + values.shape[1:]
    return jp.broadcast_to(values, shape)


def _match_target_volume_batch(target_volume, batch_size):
    target_volume = jp.array(target_volume)
    if target_volume.ndim == 0:
        target_volume = jp.expand_dims(target_volume, axis=0)
    return _match_batch_size(target_volume, batch_size)


def laplacian_smoothing(vertices, laplacian, reduction="mean"):
    """Shapes: vertices (V, 3)/(B, V, 3), laplacian (V, V)/(B, V, V)."""

    def compute_one(vertices, laplacian):
        return jp.trace(vertices.T @ laplacian @ vertices)

    vertices = _add_batch_axis(vertices, 2)
    batch_size = len(vertices)
    laplacian = _add_batch_axis(laplacian, 2)
    laplacian = _match_batch_size(laplacian, batch_size)
    loss = jax.vmap(compute_one)(vertices, laplacian)
    return _reduce_loss(loss, reduction=reduction)


def volume_matching(vertices, faces, target_volume, reduction="mean"):
    """Shapes: vertices (V, 3)/(B, V, 3), faces (F, 3)/(B, F, 3)."""
    vertices = _add_batch_axis(vertices, 2)
    batch_size = len(vertices)
    faces = _add_batch_axis(faces, 2)
    faces = _match_batch_size(faces, batch_size)
    targets = _match_target_volume_batch(target_volume, batch_size)
    current = jax.vmap(compute_volume)(vertices, faces)
    loss = (targets - current) ** 2
    return _reduce_loss(loss, reduction=reduction)
