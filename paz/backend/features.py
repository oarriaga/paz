"""Local-feature ops shared by keypoint extractors and matchers.

A feature map is a dense tensor of shape ``(height, width, channels)``, and
positions are an ``(N, 2)`` array of ``(x, y)`` pixel coordinates. These
helpers read a feature map at sparse subpixel positions, normalize descriptor
vectors, match two descriptor sets, and build coordinate grids.
"""
import jax
import jax.numpy as jp


def interpolate(features, positions, height, width, mode):
    grid = normalize_positions(positions, height, width)
    map_height, map_width = features.shape[0], features.shape[1]
    x = to_source_coordinates(grid[:, 0], map_width)
    y = to_source_coordinates(grid[:, 1], map_height)
    return interpolate_at(features, x, y, mode)


def normalize_positions(positions, height, width):
    scale = jp.array([width - 1, height - 1], positions.dtype)
    return 2.0 * positions / scale - 1.0


def to_source_coordinates(normalized, size):
    return ((normalized + 1.0) * size - 1.0) / 2.0


def interpolate_at(features, x, y, mode):
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


def gather_bicubic(features, x, y):
    left = jp.floor(x).astype(jp.int32)
    top = jp.floor(y).astype(jp.int32)
    weights_x = compute_cubic_weights(x - left)
    weights_y = compute_cubic_weights(y - top)
    result = jp.zeros((x.shape[0], features.shape[-1]), features.dtype)
    for row_offset in range(-1, 3):
        row = accumulate_row(features, left, top + row_offset, weights_x)
        result += row * weights_y[:, row_offset + 1][:, None]
    return result


def accumulate_row(features, left, rows, weights_x):
    row = jp.zeros((left.shape[0], features.shape[-1]), features.dtype)
    for column_offset in range(-1, 3):
        taps = gather_at(features, left + column_offset, rows)
        row += taps * weights_x[:, column_offset + 1][:, None]
    return row


def compute_cubic_weights(t):
    weight_0 = compute_cubic_far(t + 1)
    weight_1 = compute_cubic_near(t)
    weight_2 = compute_cubic_near(1 - t)
    weight_3 = compute_cubic_far(2 - t)
    return jp.stack([weight_0, weight_1, weight_2, weight_3], axis=-1)


def compute_cubic_near(t):
    return ((-0.75 + 2) * t - (-0.75 + 3)) * t * t + 1


def compute_cubic_far(t):
    return ((-0.75 * t - 5 * -0.75) * t + 8 * -0.75) * t - 4 * -0.75


def find_local_maxima(heatmap, kernel):
    pad = kernel // 2
    pooled = jax.lax.reduce_window(
        heatmap, -jp.inf, jax.lax.max, (kernel, kernel), (1, 1),
        [(pad, pad), (pad, pad)])
    return heatmap == pooled


def build_pixel_grid(height, width):
    columns, rows = jp.meshgrid(jp.arange(width), jp.arange(height))
    grid = jp.stack([columns.reshape(-1), rows.reshape(-1)], axis=-1)
    return grid.astype(jp.float32)


def l2_normalize(x, axis=-1):
    norm = jp.sqrt(jp.sum(x * x, axis=axis, keepdims=True))
    return x / jp.clip(norm, 1e-12, None)


def find_mutual_nearest_neighbors(descriptors_a, descriptors_b, min_cosine=-1):
    similarity = descriptors_a @ descriptors_b.T
    forward = jp.argmax(similarity, axis=1)
    backward = jp.argmax(similarity, axis=0)
    source = jp.arange(forward.shape[0])
    mutual = backward[forward] == source
    if min_cosine > 0:
        mutual = mutual & (jp.max(similarity, axis=1) > min_cosine)
    return source[mutual], forward[mutual]
