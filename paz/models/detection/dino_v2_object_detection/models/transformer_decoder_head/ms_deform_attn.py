from keras import ops
from keras import layers


def scale_grid_axis(coordinate, extent, align_corners):
    if align_corners:
        scaled = ((coordinate + 1) / 2) * (extent - 1)
    else:
        scaled = ((coordinate + 1) * extent - 1) / 2
    return scaled


def build_bilinear_weights(step_x, step_y):
    low_x, low_y = 1 - step_x, 1 - step_y
    weights = (low_x * low_y, step_x * low_y, low_x * step_y, step_x * step_y)
    return [ops.expand_dims(weight, axis=1) for weight in weights]


def gather_corner_values(input_tensor, shape, height, width, corner_y, corner_x):  # fmt: skip
    num_images, output_height, output_width, channels = shape
    rows = ops.cast(ops.clip(corner_y, 0, height - 1), "int32")
    columns = ops.cast(ops.clip(corner_x, 0, width - 1), "int32")
    batch = ops.reshape(ops.arange(num_images), (num_images, 1, 1))
    batch = ops.broadcast_to(batch, (num_images, output_height, output_width))
    flat = ops.reshape(batch * (height * width) + rows * width + columns, (-1,))
    channels_last = ops.transpose(input_tensor, (0, 2, 3, 1))
    values = ops.take(ops.reshape(channels_last, (-1, channels)), flat, axis=0)
    return ops.transpose(ops.reshape(values, shape), (0, 3, 1, 2))


def build_validity_mask(corner_x, corner_y, height, width, dtype):
    inside_x = ops.logical_and(corner_x >= 0, corner_x <= width - 1)
    inside_y = ops.logical_and(corner_y >= 0, corner_y <= height - 1)
    inside = ops.logical_and(inside_x, inside_y)
    return ops.expand_dims(ops.cast(inside, dtype), axis=1)


def grid_sample(input_tensor, grid, align_corners=False):
    num_images, channels, height, width = input_tensor.shape
    shape = (num_images, grid.shape[1], grid.shape[2], channels)
    x = scale_grid_axis(grid[..., 0], width, align_corners)
    y = scale_grid_axis(grid[..., 1], height, align_corners)
    x0, y0 = ops.floor(x), ops.floor(y)
    corners = ((x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1))
    weights = build_bilinear_weights(x - x0, y - y0)
    output = None
    for (corner_x, corner_y), weight in zip(corners, weights):
        args = (input_tensor, shape, height, width, corner_y, corner_x)
        value = gather_corner_values(*args)
        mask_args = (corner_x, corner_y, height, width, input_tensor.dtype)
        term = weight * (value * build_validity_mask(*mask_args))
        output = term if output is None else output + term
    return output


def split_value_levels(value, spatial_shapes, length_in):
    sizes = [int(height * width) for height, width in spatial_shapes]
    assert sum(sizes) == length_in
    levels = []
    start = 0
    for size in sizes:
        levels.append(value[:, start : start + size, :, :])
        start = start + size
    return levels


def sample_levels(value, spatial_shapes, sampling_locations, num_heads, head_dim, num_queries, num_points):  # fmt: skip
    num_images = value.shape[0]
    levels = split_value_levels(value, spatial_shapes, value.shape[1])
    grids = 2 * sampling_locations - 1
    sampled = []
    for level, (height, width) in enumerate(spatial_shapes):
        stack = ops.transpose(levels[level], (0, 2, 3, 1))
        shape = (num_images * num_heads, head_dim, int(height), int(width))
        grid = ops.transpose(grids[:, :, :, level], (0, 2, 1, 3, 4))
        grid_shape = (num_images * num_heads, num_queries, num_points, 2)
        grid = ops.reshape(grid, grid_shape)
        sampled.append(grid_sample(ops.reshape(stack, shape), grid, False))
    return sampled


def ms_deform_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights):  # fmt: skip
    num_images, _, num_heads, head_dim = value.shape
    num_queries = sampling_locations.shape[1]
    num_levels = sampling_locations.shape[3]
    num_points = sampling_locations.shape[4]
    args = (value, value_spatial_shapes, sampling_locations, num_heads)
    sampled = sample_levels(*args, head_dim, num_queries, num_points)
    groups, span = num_images * num_heads, num_levels * num_points
    attention = ops.transpose(attention_weights, (0, 2, 1, 3, 4))
    attention = ops.reshape(attention, (groups, 1, num_queries, span))
    stacked = ops.stack(sampled, axis=3)
    stacked = ops.reshape(stacked, (groups, head_dim, num_queries, span))
    output = ops.sum(stacked * attention, axis=-1)
    output = ops.reshape(output, (num_images, num_heads, head_dim, num_queries))
    output = ops.transpose(output, (0, 3, 1, 2))
    return ops.reshape(output, (num_images, num_queries, num_heads * head_dim))


def build_ms_deform_dense(d_model, num_levels, num_heads, num_points, name):
    if d_model % num_heads != 0:
        message = f"d_model must be divisible by num_heads, but got {d_model} and {num_heads}"  # fmt: skip
        raise ValueError(message)
    per_head = num_heads * num_levels * num_points
    offsets = layers.Dense(per_head * 2, name=f"{name}_sampling_offsets")
    weights = layers.Dense(per_head, name=f"{name}_attention_weights")
    value_projection = layers.Dense(d_model, name=f"{name}_value_proj")
    output_projection = layers.Dense(d_model, name=f"{name}_output_proj")
    return value_projection, offsets, weights, output_projection


def project_masked_value(input_flatten, input_padding_mask, value_proj):
    value = value_proj(input_flatten)
    if input_padding_mask is not None:
        value = ops.where(ops.expand_dims(input_padding_mask, -1), 0.0, value)
    return value


def normalize_attention_weights(weights, num_images, num_queries, num_heads, num_levels, num_points):  # fmt: skip
    flat = (num_images, num_queries, num_heads, num_levels * num_points)
    weights = ops.softmax(ops.reshape(weights, flat), axis=-1)
    shape = (num_images, num_queries, num_heads, num_levels, num_points)
    return ops.reshape(weights, shape)


def build_point_locations(reference_points, offsets, spatial_shapes, num_levels):  # fmt: skip
    shapes = ops.convert_to_tensor(spatial_shapes, dtype="float32")
    normalizer = ops.stack([shapes[..., 1], shapes[..., 0]], axis=-1)
    normalizer = ops.reshape(normalizer, (1, 1, 1, num_levels, 1, 2))
    centers = ops.expand_dims(ops.expand_dims(reference_points, 2), 4)
    return centers + offsets / normalizer


def build_box_locations(reference_points, offsets, num_points):
    centers = ops.expand_dims(ops.expand_dims(reference_points[..., :2], 2), 4)
    sizes = ops.expand_dims(ops.expand_dims(reference_points[..., 2:], 2), 4)
    return centers + offsets / num_points * sizes * 0.5


def build_sampling_locations(reference_points, offsets, spatial_shapes, num_levels, num_points):  # fmt: skip
    if reference_points.shape[-1] == 2:
        args = (reference_points, offsets, spatial_shapes, num_levels)
        locations = build_point_locations(*args)
    elif reference_points.shape[-1] == 4:
        locations = build_box_locations(reference_points, offsets, num_points)
    else:
        raise ValueError("Last dim of reference_points must be 2 or 4.")
    return locations


def apply_ms_deform_attn(query, reference_points, input_flatten, input_spatial_shapes, input_padding_mask, value_proj, sampling_offsets, attention_weights, output_proj, num_levels, num_heads, num_points):  # fmt: skip
    num_images, num_queries = ops.shape(query)[0], ops.shape(query)[1]
    head_dim = value_proj.units // num_heads
    args = (input_flatten, input_padding_mask, value_proj)
    shape = (num_images, ops.shape(input_flatten)[1], num_heads, head_dim)
    value = ops.reshape(project_masked_value(*args), shape)
    offsets_shape = (num_images, num_queries, num_heads, num_levels, num_points, 2)  # fmt: skip
    offsets = ops.reshape(sampling_offsets(query), offsets_shape)
    args = (attention_weights(query), num_images, num_queries, num_heads)
    weights = normalize_attention_weights(*args, num_levels, num_points)
    args = (reference_points, offsets, input_spatial_shapes)
    locations = build_sampling_locations(*args, num_levels, num_points)
    core = ms_deform_attn_core(value, input_spatial_shapes, locations, weights)
    return output_proj(core)


def materialize_ms_deform_attn(query, memory, d_model, num_levels, num_heads, num_points, name):  # fmt: skip
    args = (d_model, num_levels, num_heads, num_points, name)
    value_proj, sampling_offsets, attention_weights, output_proj = build_ms_deform_dense(*args)  # fmt: skip
    value = value_proj(memory)
    offsets = sampling_offsets(query)
    weights = attention_weights(query)
    return [value, offsets, weights, output_proj(value)]


def run_ms_deform_attn(model, query, reference_points, input_flatten, input_spatial_shapes, input_padding_mask, num_levels, num_heads, num_points, name):  # fmt: skip
    projections = ("value_proj", "sampling_offsets", "attention_weights", "output_proj")  # fmt: skip
    layer_args = [model.get_layer(f"{name}_{part}") for part in projections]
    args = (query, reference_points, input_flatten, input_spatial_shapes)
    return apply_ms_deform_attn(*args, input_padding_mask, *layer_args, num_levels, num_heads, num_points)  # fmt: skip
