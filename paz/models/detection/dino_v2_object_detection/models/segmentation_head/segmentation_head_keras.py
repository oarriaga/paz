import keras
from keras import Input, Model, ops, layers
from keras.layers import EinsumDense

from paz.models.foundation.dinov2.layers.layer_scale import apply_layer_scale


def scale_sample_axis(coordinate, extent, align_corners):
    if align_corners:
        scaled = ((coordinate + 1) / 2) * (extent - 1)
    else:
        scaled = ((coordinate + 1) * extent - 1) / 2
    return scaled


def gather_grid_values(input_tensor, shape, height, width, row, column):
    num_images, _, _, channels = shape
    channels_last = ops.transpose(input_tensor, (0, 2, 3, 1))
    flat = ops.reshape(channels_last, (-1, channels))
    rows = ops.cast(ops.clip(row, 0, height - 1), "int32")
    columns = ops.cast(ops.clip(column, 0, width - 1), "int32")
    strides = ops.arange(num_images) * (height * width)
    offsets = ops.reshape(strides, (num_images, 1, 1))
    indices = ops.reshape(rows * width + columns + offsets, (-1,))
    return ops.reshape(ops.take(flat, indices, axis=0), shape)


def build_corner_weights(x, y, x0, y0, x1, y1):
    left, right = x1 - x, x - x0
    top, bottom = y1 - y, y - y0
    weights = (left * top, left * bottom, right * top, right * bottom)
    return [ops.expand_dims(weight, -1) for weight in weights]


def blend_corners(corners, weights):
    output = None
    for corner, weight in zip(corners, weights):
        term = weight * corner
        output = term if output is None else output + term
    return output


def grid_sample(input_tensor, grid, align_corners=False):
    num_images, channels, height, width = ops.shape(input_tensor)
    shape = (num_images, ops.shape(grid)[1], ops.shape(grid)[2], channels)
    x = scale_sample_axis(grid[..., 0], width, align_corners)
    y = scale_sample_axis(grid[..., 1], height, align_corners)
    x0, y0 = ops.floor(x), ops.floor(y)
    x1, y1 = x0 + 1, y0 + 1
    args = (input_tensor, shape, height, width)
    corners = (gather_grid_values(*args, y0, x0), gather_grid_values(*args, y1, x0), gather_grid_values(*args, y0, x1), gather_grid_values(*args, y1, x1))  # fmt: skip
    weights = build_corner_weights(x, y, x0, y0, x1, y1)
    # Gathering happens in NHWC; convert the blend back to NCHW.
    return ops.transpose(blend_corners(corners, weights), (0, 3, 1, 2))


def point_sample(input_tensor, point_coords, **kwargs):
    # When given (N, P, 2) points, add a dummy spatial dimension so the
    # grid becomes (N, P, 1, 2) which grid_sample can process.
    add_dim = False
    if ops.ndim(point_coords) == 3:
        add_dim = True
        point_coords = ops.expand_dims(point_coords, 2)

    # Rescale from [0, 1] to the [-1, 1] range expected by grid_sample
    grid = 2.0 * point_coords - 1.0

    align_corners = kwargs.get("align_corners", False)
    output = grid_sample(input_tensor, grid, align_corners=align_corners)

    if add_dim:
        output = ops.squeeze(output, 3)  # remove dummy W dim -> (N, C, P)

    return output


def calculate_uncertainty(logits):
    return -ops.abs(logits)


def sample_uncertain_points(candidates, uncertainties, num_images, num_sampled, num_uncertain):  # fmt: skip
    ranked = ops.top_k(uncertainties, k=num_uncertain)[1]
    # Flatten across the batch so one take suffices; the per-image offset
    # keeps each sample indexing its own candidate block.
    shift = ops.expand_dims(ops.arange(num_images) * num_sampled, -1)
    indices = ops.reshape(ranked + ops.cast(shift, ranked.dtype), (-1,))
    selected = ops.take(ops.reshape(candidates, (-1, 2)), indices, axis=0)
    return ops.reshape(selected, (num_images, num_uncertain, 2))


def append_random_points(selected, num_images, num_random):
    if num_random > 0:
        shape = (num_images, num_random, 2)
        extra = keras.random.uniform(shape, minval=0.0, maxval=1.0)
        selected = ops.concatenate([selected, extra], axis=1)
    return selected


def get_uncertain_point_coords_with_randomness(coarse_logits, uncertainty_func, num_points, oversample_ratio=3, importance_sample_ratio=0.75):  # fmt: skip
    num_images = ops.shape(coarse_logits)[0]
    num_sampled = int(num_points * oversample_ratio)
    shape = (num_images, num_sampled, 2)
    candidates = keras.random.uniform(shape, minval=0.0, maxval=1.0)
    logits = point_sample(coarse_logits, candidates, align_corners=False)
    uncertainties = uncertainty_func(logits)[:, 0, :]
    num_uncertain = int(importance_sample_ratio * num_points)
    args = (candidates, uncertainties, num_images, num_sampled, num_uncertain)
    selected = sample_uncertain_points(*args)
    num_random = num_points - num_uncertain
    return append_random_points(selected, num_images, num_random)


def SegmentationHead(in_dim, num_blocks, bottleneck_ratio=1, downsample_ratio=4, name="segmentation_head"):  # fmt: skip
    interaction_dim = in_dim if bottleneck_ratio is None else in_dim // bottleneck_ratio  # fmt: skip
    spatial = Input(shape=(in_dim, None, None), name="spatial_features")
    query = Input(shape=(None, in_dim), name="query_features")
    args = (spatial, query, in_dim, num_blocks, interaction_dim, bottleneck_ratio)  # fmt: skip
    model = Model([spatial, query], build_segmentation_head(*args), name=name)
    model.downsample_ratio = downsample_ratio
    return model


def build_segmentation_head(spatial, query, in_dim, num_blocks, interaction_dim, bottleneck_ratio):  # fmt: skip
    refined = spatial
    for index in range(num_blocks):
        refined = depthwise_conv_block(refined, in_dim, 0, f"block_{index}")
    projected_spatial = project_spatial_features(refined, interaction_dim, bottleneck_ratio, "spatial_features_proj")  # fmt: skip
    projected_query = project_query_features(query, in_dim, interaction_dim, bottleneck_ratio)  # fmt: skip
    logit = ops.einsum("bchw,bnc->bnhw", projected_spatial, projected_query)
    return apply_mask_bias(logit, "bias")


def depthwise_conv_block(x, dim, layer_scale_init_value, name):
    input_tensor = x
    x = build_depthwise_conv(x, f"{name}_dwconv")
    x = ops.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC for norm/dense
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm")(x)
    x = layers.Dense(dim, name=f"{name}_pwconv1")(x)
    x = layers.Activation("gelu", name=f"{name}_act")(x)
    x = apply_layer_scale(x, dim, layer_scale_init_value, f"{name}_gamma")
    x = ops.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x + input_tensor


def mlp_block(x, dim, layer_scale_init_value, name):
    input_tensor = x
    x = layers.LayerNormalization(epsilon=1e-5, name=f"{name}_norm_in")(x)
    x = layers.Dense(dim * 4, name=f"{name}_linear1")(x)
    x = layers.Activation("gelu", name=f"{name}_act")(x)
    x = layers.Dense(dim, name=f"{name}_linear2")(x)
    x = apply_layer_scale(x, dim, layer_scale_init_value, f"{name}_gamma")
    return x + input_tensor


def build_depthwise_conv(x, name):
    keys = ("kernel_size", "padding", "data_format", "depth_multiplier", "use_bias", "name")  # fmt: skip
    values = (3, "same", "channels_first", 1, True, name)
    return layers.DepthwiseConv2D(**dict(zip(keys, values)))(x)


def project_spatial_features(x, interaction_dim, bottleneck_ratio, name):
    if bottleneck_ratio is None:
        result = layers.Identity(name=name)(x)
    else:
        keys = ("kernel_size", "data_format", "use_bias", "name")
        values = (1, "channels_first", True, name)
        result = layers.Conv2D(interaction_dim, **dict(zip(keys, values)))(x)
    return result


def project_query_features(query, in_dim, interaction_dim, bottleneck_ratio):
    refined = mlp_block(query, in_dim, 0, "query_features_block")
    if bottleneck_ratio is None:
        result = layers.Identity(name="query_features_proj")(refined)
    else:
        result = layers.Dense(interaction_dim, name="query_features_proj")(refined)  # fmt: skip
    return result


def apply_mask_bias(logit, name):
    # EinsumDense holds the learnable scalar bias as its (1,) kernel; applying
    # it to a ones tensor broadcasts that scalar additively over the logit map.
    ones = ops.expand_dims(ops.ones_like(logit), -1)
    keys = ("output_shape", "bias_axes", "kernel_initializer", "name")
    values = ((1,), None, "zeros", name)
    bias = EinsumDense("...d,d->...d", **dict(zip(keys, values)))(ones)
    return logit + ops.squeeze(bias, -1)


def apply_segmentation_head(model, spatial_features, query_features, image_size=None, skip_blocks=False):  # fmt: skip
    if image_size is not None:
        spatial_features = resize_spatial_features(spatial_features, image_size, model.downsample_ratio)  # fmt: skip
    if skip_blocks:
        result = skip_blocks_mask_logits(model, spatial_features, query_features)  # fmt: skip
    else:
        result = blocks_mask_logits(model, spatial_features, query_features)
    return result


def blocks_mask_logits(model, spatial_features, query_features):
    mask_logits = []
    for index, query in enumerate(query_features):
        if not has_layer(model, f"block_{index}_dwconv"):
            break
        spatial_features = run_depthwise_conv_block(model, spatial_features, f"block_{index}")  # fmt: skip
        projected_spatial = run_spatial_projection(model, spatial_features)
        projected_query = run_query_projection(model, query)
        logit = ops.einsum("bchw,bnc->bnhw", projected_spatial, projected_query)  # fmt: skip
        mask_logits.append(apply_bias(model, logit))
    return mask_logits


def skip_blocks_mask_logits(model, spatial_features, query_features):
    if len(query_features) != 1:
        raise ValueError("skip_blocks is only supported for length 1 query features")  # fmt: skip
    projected_query = run_query_projection(model, query_features[0])
    logit = ops.einsum("bchw,bnc->bnhw", spatial_features, projected_query)
    return [apply_bias(model, logit)]


def apply_export_segmentation_head(model, spatial_features, query_features, image_size=None, skip_blocks=False):  # fmt: skip
    if len(query_features) != 1:
        raise ValueError("at export time, segmentation head expects exactly one query feature")  # fmt: skip
    if image_size is not None:
        spatial_features = resize_spatial_features(spatial_features, image_size, model.downsample_ratio)  # fmt: skip
    if not skip_blocks:
        spatial_features = run_all_blocks(model, spatial_features)
    projected_spatial = run_spatial_projection(model, spatial_features)
    projected_query = run_query_projection(model, query_features[0])
    logit = ops.einsum("bchw,bnc->bnhw", projected_spatial, projected_query)
    return [apply_bias(model, logit)]


def sparse_segmentation_head(model, spatial_features, query_features, image_size=None, skip_blocks=False):  # fmt: skip
    if image_size is not None:
        spatial_features = resize_spatial_features(spatial_features, image_size, model.downsample_ratio)  # fmt: skip
    if skip_blocks:
        result = sparse_skip_blocks(model, spatial_features, query_features)
    else:
        result = sparse_blocks(model, spatial_features, query_features)
    return result


def sparse_blocks(model, spatial_features, query_features):
    outputs = []
    for index, query in enumerate(query_features):
        if not has_layer(model, f"block_{index}_dwconv"):
            break
        spatial_features = run_depthwise_conv_block(model, spatial_features, f"block_{index}")  # fmt: skip
        projected_spatial = run_spatial_projection(model, spatial_features)
        projected_query = run_query_projection(model, query)
        outputs.append(sparse_output(model, projected_spatial, projected_query))  # fmt: skip
    return outputs


def sparse_skip_blocks(model, spatial_features, query_features):
    if len(query_features) != 1:
        raise ValueError("skip_blocks is only supported for length 1 query features")  # fmt: skip
    projected_query = run_query_projection(model, query_features[0])
    return [sparse_output(model, spatial_features, projected_query)]


def sparse_output(model, spatial_features, query_features):
    bias = model.get_layer("bias").kernel
    return {"spatial_features": spatial_features, "query_features": query_features, "bias": bias}  # fmt: skip


def run_depthwise_conv_block(model, x, name):
    input_tensor = x
    x = model.get_layer(f"{name}_dwconv")(x)
    x = ops.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC for norm/dense
    x = model.get_layer(f"{name}_norm")(x)
    x = model.get_layer(f"{name}_pwconv1")(x)
    x = model.get_layer(f"{name}_act")(x)
    x = model.get_layer(f"{name}_gamma")(x)
    x = ops.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x + input_tensor


def run_mlp_block(model, x, name):
    input_tensor = x
    x = model.get_layer(f"{name}_norm_in")(x)
    x = model.get_layer(f"{name}_linear1")(x)
    x = model.get_layer(f"{name}_act")(x)
    x = model.get_layer(f"{name}_linear2")(x)
    x = model.get_layer(f"{name}_gamma")(x)
    return x + input_tensor


def run_query_projection(model, query):
    refined = run_mlp_block(model, query, "query_features_block")
    return model.get_layer("query_features_proj")(refined)


def run_spatial_projection(model, spatial_features):
    return model.get_layer("spatial_features_proj")(spatial_features)


def run_all_blocks(model, spatial_features):
    index = 0
    while has_layer(model, f"block_{index}_dwconv"):
        spatial_features = run_depthwise_conv_block(model, spatial_features, f"block_{index}")  # fmt: skip
        index = index + 1
    return spatial_features


def apply_bias(model, logit):
    ones = ops.expand_dims(ops.ones_like(logit), -1)
    return logit + ops.squeeze(model.get_layer("bias")(ones), -1)


def resize_spatial_features(spatial_features, image_size, downsample_ratio):
    target_height = image_size[0] // downsample_ratio
    target_width = image_size[1] // downsample_ratio
    spatial_features = ops.transpose(spatial_features, (0, 2, 3, 1))
    spatial_features = ops.image.resize(spatial_features, (target_height, target_width), interpolation="bilinear")  # fmt: skip
    return ops.transpose(spatial_features, (0, 3, 1, 2))


def has_layer(model, name):
    result = True
    try:
        model.get_layer(name)
    except ValueError:
        result = False
    return result
