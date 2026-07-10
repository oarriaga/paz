import numpy as np
import jax
import jax.numpy as jp

CUBIC_A = -0.75
CELL = 8


def normalize_positions(positions, height, width):
    scale = jp.array([width - 1, height - 1], positions.dtype)
    return 2.0 * positions / scale - 1.0


def to_source_coordinates(normalized, size):
    return ((normalized + 1.0) * size - 1.0) / 2.0


def sample_features(features, positions, height, width, mode):
    grid = normalize_positions(positions, height, width)
    map_height, map_width = features.shape[0], features.shape[1]
    x = to_source_coordinates(grid[:, 0], map_width)
    y = to_source_coordinates(grid[:, 1], map_height)
    return interpolate(features, x, y, mode)


def interpolate(features, x, y, mode):
    if mode == "nearest":
        return gather_nearest(features, x, y)
    if mode == "bilinear":
        return gather_bilinear(features, x, y)
    return gather_bicubic(features, x, y)


def gather_at(features, columns, rows):
    height, width = features.shape[0], features.shape[1]
    inside = (columns >= 0) & (columns < width)
    inside = inside & (rows >= 0) & (rows < height)
    safe_columns = jp.clip(columns, 0, width - 1)
    safe_rows = jp.clip(rows, 0, height - 1)
    values = features[safe_rows, safe_columns]
    return values * inside[:, None]


def gather_nearest(features, x, y):
    columns = jp.floor(x + 0.5).astype(jp.int32)
    rows = jp.floor(y + 0.5).astype(jp.int32)
    return gather_at(features, columns, rows)


def gather_bilinear(features, x, y):
    left = jp.floor(x).astype(jp.int32)
    top = jp.floor(y).astype(jp.int32)
    tx = (x - left)[:, None]
    ty = (y - top)[:, None]
    top_left = gather_at(features, left, top)
    top_right = gather_at(features, left + 1, top)
    bottom_left = gather_at(features, left, top + 1)
    bottom_right = gather_at(features, left + 1, top + 1)
    top_row = top_left * (1 - tx) + top_right * tx
    bottom_row = bottom_left * (1 - tx) + bottom_right * tx
    return top_row * (1 - ty) + bottom_row * ty


def cubic_near(t):
    return ((CUBIC_A + 2) * t - (CUBIC_A + 3)) * t * t + 1


def cubic_far(t):
    return ((CUBIC_A * t - 5 * CUBIC_A) * t + 8 * CUBIC_A) * t - 4 * CUBIC_A


def cubic_weights(t):
    w0 = cubic_far(t + 1)
    w1 = cubic_near(t)
    w2 = cubic_near(1 - t)
    w3 = cubic_far(2 - t)
    return jp.stack([w0, w1, w2, w3], axis=-1)


def gather_bicubic(features, x, y):
    left = jp.floor(x).astype(jp.int32)
    top = jp.floor(y).astype(jp.int32)
    weights_x = cubic_weights(x - left)
    weights_y = cubic_weights(y - top)
    result = jp.zeros((x.shape[0], features.shape[-1]), features.dtype)
    for row_offset in range(-1, 3):
        row_value = accumulate_row(features, left, top + row_offset, weights_x)
        result += row_value * weights_y[:, row_offset + 1][:, None]
    return result


def accumulate_row(features, left, rows, weights_x):
    row = jp.zeros((left.shape[0], features.shape[-1]), features.dtype)
    for column_offset in range(-1, 3):
        taps = gather_at(features, left + column_offset, rows)
        row += taps * weights_x[:, column_offset + 1][:, None]
    return row


def keypoint_heatmap(logits):
    scores = jax.nn.softmax(logits, axis=-1)[..., :CELL * CELL]
    height, width = scores.shape[0], scores.shape[1]
    scores = jp.reshape(scores, (height, width, CELL, CELL))
    scores = jp.transpose(scores, (0, 2, 1, 3))
    return jp.reshape(scores, (height * CELL, width * CELL))


def local_maxima(heatmap, kernel):
    pad = kernel // 2
    pooled = jax.lax.reduce_window(
        heatmap, -jp.inf, jax.lax.max, (kernel, kernel), (1, 1),
        [(pad, pad), (pad, pad)])
    return heatmap == pooled


def l2_normalize(x, axis=-1):
    norm = jp.sqrt(jp.sum(x * x, axis=axis, keepdims=True))
    return x / jp.clip(norm, 1e-12, None)


def mutual_nearest_neighbors(descriptors1, descriptors2, min_cosine=-1.0):
    similarity = descriptors1 @ descriptors2.T
    match12 = jp.argmax(similarity, axis=1)
    match21 = jp.argmax(similarity, axis=0)
    source = jp.arange(match12.shape[0])
    mutual = match21[match12] == source
    if min_cosine > 0:
        best = jp.max(similarity, axis=1)
        mutual = mutual & (best > min_cosine)
    source = np.asarray(source)[np.asarray(mutual)]
    target = np.asarray(match12)[np.asarray(mutual)]
    return source, target


def full_grid(height, width):
    columns, rows = jp.meshgrid(jp.arange(width), jp.arange(height))
    grid = jp.stack([columns.reshape(-1), rows.reshape(-1)], axis=-1)
    return grid.astype(jp.float32)


def dense_scores(heatmap, heat, threshold, height, width):
    grid = full_grid(heatmap.shape[0], heatmap.shape[1])
    sharp = sample_features(heatmap[..., None], grid, height, width, "nearest")
    soft = sample_features(heat, grid, height, width, "bilinear")
    keep = local_maxima(heatmap, 5) & (heatmap > threshold)
    scores = jp.where(keep.reshape(-1), (sharp * soft)[:, 0], -1.0)
    return grid, scores
