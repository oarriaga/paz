"""Multi-scale deformable attention (Deformable DETR).

Each query predicts a few sampling offsets around its reference box, plus the
weights to mix the sampled features with, so attention costs ``num_points``
lookups per head and level instead of one score against every key.
"""
from keras import ops
from keras.layers import Dense

from paz.backend.image import sample_bilinear


def attend(query, value, boxes, grids, num_heads, num_points, name):
    """Attends to ``value`` around normalized ``(cx, cy, w, h)`` ``boxes``.

    ``value`` holds every level flattened and concatenated along the token
    axis, in the order given by ``grids``, a tuple of ``(height, width)``.
    """
    hidden_size = query.shape[-1]
    args = num_heads, len(grids), num_points
    offsets = project_offsets(query, *args, name)
    weights = project_weights(query, *args, name)
    values = project_values(value, hidden_size, num_heads, name)
    positions = compute_positions(boxes, offsets, num_points)
    sampled = sample_levels(values, positions, grids)
    context = mix(sampled, weights, num_heads)
    return Dense(hidden_size, name=f"{name}_output")(context)


def project_offsets(query, num_heads, num_levels, num_points, name):
    units = num_heads * num_levels * num_points * 2
    offsets = Dense(units, name=f"{name}_offsets")(query)
    shape = (-1, query.shape[1], num_heads, num_levels, num_points, 2)
    return ops.reshape(offsets, shape)


def project_weights(query, num_heads, num_levels, num_points, name):
    units = num_heads * num_levels * num_points
    weights = Dense(units, name=f"{name}_weights")(query)
    flat = (-1, query.shape[1], num_heads, num_levels * num_points)
    normalized = ops.softmax(ops.reshape(weights, flat), axis=-1)
    shape = (-1, query.shape[1], num_heads, num_levels, num_points)
    return ops.reshape(normalized, shape)


def project_values(value, hidden_size, num_heads, name):
    projected = Dense(hidden_size, name=f"{name}_values")(value)
    shape = (-1, value.shape[1], num_heads, hidden_size // num_heads)
    return ops.reshape(projected, shape)


def compute_positions(boxes, offsets, num_points):
    centers = boxes[:, :, None, None, None, :2]
    sizes = boxes[:, :, None, None, None, 2:]
    return centers + offsets / num_points * sizes * 0.5


def sample_levels(values, positions, grids):
    sampled = []
    start = 0
    for level, grid in enumerate(grids):
        tokens = grid[0] * grid[1]
        level_values = values[:, start:start + tokens]
        sampled.append(sample_level(level_values, positions, grid, level))
        start = start + tokens
    return ops.stack(sampled, axis=2)


def sample_level(values, positions, grid, level):
    features = ops.transpose(values, (0, 2, 1, 3))
    features = ops.reshape(features, (-1, grid[0], grid[1], values.shape[-1]))
    coordinates = ops.transpose(positions[:, :, :, level], (0, 2, 1, 3, 4))
    coordinates = ops.reshape(coordinates, (-1, *coordinates.shape[2:]))
    return sample_bilinear(features, 2.0 * coordinates - 1.0)


def mix(sampled, weights, num_heads):
    scale = ops.transpose(weights, (0, 2, 1, 3, 4))
    shape = (-1, *scale.shape[2:], 1)
    context = ops.sum(sampled * ops.reshape(scale, shape), axis=(2, 3))
    return merge_heads(context, num_heads)


def merge_heads(context, num_heads):
    num_queries, head_dim = context.shape[1], context.shape[2]
    heads = ops.reshape(context, (-1, num_heads, num_queries, head_dim))
    merged = ops.transpose(heads, (0, 2, 1, 3))
    return ops.reshape(merged, (-1, num_queries, num_heads * head_dim))
