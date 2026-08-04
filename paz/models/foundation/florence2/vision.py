"""DaViT vision tower and the Florence-2 image feature path.

Mirrors microsoft/Florence-2-large ``modeling_florence2.py`` (revision
21a599d4) restricted to inference: dual-attention stages (window and
channel attention), learned 2D position embeddings, the temporal cosine
embedding for a single frame, average-pool token, projection and norm.
Default parameter values are the DaViT-large architecture.
"""
from keras import ops
from keras.layers import Conv2D, Dense, DepthwiseConv2D, Embedding
from keras.layers import Input, Lambda, LayerNormalization, ZeroPadding2D
from keras.models import Model

from paz.models.transformers import feedforward
from paz.models.transformers.attention import masked_attend


def ImageEncoder(image_size=112, projection_dim=1024,
                 stage_dims=(256, 512, 1024, 2048),
                 stage_depths=(1, 1, 9, 1), stage_heads=(8, 16, 32, 64),
                 stage_groups=(8, 16, 32, 64), window_size=12,
                 name="image_encoder"):
    image = Input((image_size, image_size, 3), name="image")
    stage_args = (stage_dims, stage_depths, stage_heads, stage_groups)
    tokens, side = build_stages(image, image_size, *stage_args, window_size)
    tokens = add_2d_positions(tokens, side)
    tokens = add_temporal_embedding(tokens)
    tokens = concatenate_pooled(tokens)
    project = Dense(projection_dim, use_bias=False, name="image_projection")
    tokens = build_norm("image_proj_norm")(project(tokens))
    return Model(image, tokens, name=name)


def build_stages(image, image_size, stage_dims, stage_depths, stage_heads,
                 stage_groups, window_size):
    x, grid = image, (image_size, image_size)
    for stage, dim in enumerate(stage_dims):
        x, grid = embed_patches(x, grid, stage, dim)
        for block in range(stage_depths[stage]):
            name = f"blocks_{stage}_{block}"
            heads, groups = stage_heads[stage], stage_groups[stage]
            spatial_args = (x, grid, dim, heads, window_size)
            x = spatial_block(*spatial_args, f"{name}_spatial")
            x = channel_block(x, grid, dim, groups, f"{name}_channel")
    return x, grid[0]


def embed_patches(x, grid, stage, dim):
    patch_sizes, patch_strides = (7, 3, 3, 3), (4, 2, 2, 2)
    patch_paddings = (3, 1, 1, 1)
    prenorms = (False, True, True, True)
    if len(x.shape) == 3:
        if prenorms[stage]:
            x = build_norm(f"convs_{stage}_norm")(x)
        x = to_grid(x, grid)
    x = ZeroPadding2D(patch_paddings[stage], name=f"convs_{stage}_pad")(x)
    conv_args = (dim, patch_sizes[stage], patch_strides[stage])
    x = Conv2D(*conv_args, "valid", name=f"convs_{stage}_proj")(x)
    grid = (x.shape[1], x.shape[2])
    x = to_tokens(x)
    if not prenorms[stage]:
        x = build_norm(f"convs_{stage}_norm")(x)
    return x, grid


def spatial_block(x, grid, dim, num_heads, window, name):
    x = add_depthwise(x, grid, f"{name}_conv1")
    y = build_norm(f"{name}_window_attention_norm")(x)
    window_args = (y, grid, dim, num_heads, window)
    x = x + attend_windows(*window_args, f"{name}_window_attention")
    x = add_depthwise(x, grid, f"{name}_conv2")
    y = build_norm(f"{name}_ffn_norm")(x)
    ffn_args = (y, 4 * dim, dim, f"{name}_ffn_fc1", f"{name}_ffn_fc2")
    return x + feedforward.gelu(*ffn_args)


def channel_block(x, grid, dim, groups, name):
    num_tokens = grid[0] * grid[1]
    x = add_depthwise(x, grid, f"{name}_conv1")
    y = build_norm(f"{name}_attention_norm")(x)
    channel_args = (y, num_tokens, groups, dim)
    x = x + attend_channels(*channel_args, f"{name}_attention")
    x = add_depthwise(x, grid, f"{name}_conv2")
    y = build_norm(f"{name}_ffn_norm")(x)
    ffn_args = (y, 4 * dim, dim, f"{name}_ffn_fc1", f"{name}_ffn_fc2")
    return x + feedforward.gelu(*ffn_args)


def attend_windows(x, grid, dim, num_heads, window, name):
    H, W = grid
    padded = (H + (-H) % window, W + (-W) % window)
    x = to_grid(x, grid)
    pads = ((0, 0), (0, padded[0] - H), (0, padded[1] - W), (0, 0))
    x = ops.pad(x, pads)
    x = partition_windows(x, padded, window, dim)
    x = masked_attend(x, x, None, num_heads, dim // num_heads, 0.0, name)
    x = merge_windows(x, padded, window, dim)
    return to_tokens(x[:, :H, :W, :])


def partition_windows(x, padded, window, dim):
    H, W = padded
    x = ops.reshape(x, (-1, H // window, window, W // window, window, dim))
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return ops.reshape(x, (-1, window * window, dim))


def merge_windows(x, padded, window, dim):
    H, W = padded
    x = ops.reshape(x, (-1, H // window, W // window, window, window, dim))
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return ops.reshape(x, (-1, H, W, dim))


def attend_channels(x, num_tokens, groups, dim, name):
    group_dim = dim // groups
    qkv = Dense(3 * dim, name=f"{name}_qkv")(x)
    qkv = ops.reshape(qkv, (-1, num_tokens, 3, groups, group_dim))
    qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * float(num_tokens) ** -0.5
    scores = ops.matmul(ops.transpose(q, (0, 1, 3, 2)), k)
    weights = ops.softmax(scores, axis=-1)
    out = ops.matmul(weights, ops.transpose(v, (0, 1, 3, 2)))
    out = ops.transpose(out, (0, 3, 1, 2))
    out = ops.reshape(out, (-1, num_tokens, dim))
    return Dense(dim, name=f"{name}_proj")(out)


def add_depthwise(x, grid, name):
    y = to_grid(x, grid)
    y = DepthwiseConv2D(3, 1, "same", name=name)(y)
    return x + to_tokens(y)


def add_2d_positions(x, side, num_position_rows=50):
    dim = x.shape[-1]
    half = dim // 2
    rows = Embedding(num_position_rows, half, name="image_row_embedding")
    column_name = "image_column_embedding"
    columns = Embedding(num_position_rows, dim - half, name=column_name)
    column_fn = lambda t: ops.tile(ops.arange(side, dtype="int32"), [side])
    row_fn = lambda t: ops.repeat(ops.arange(side, dtype="int32"), side)
    column_indices = build_token_indices(x, side, column_fn, "image_columns")
    row_indices = build_token_indices(x, side, row_fn, "image_rows")
    embedded = (columns(column_indices), rows(row_indices))
    return x + ops.concatenate(embedded, axis=-1)


def build_token_indices(x, side, fn, name):
    return Lambda(fn, output_shape=(side * side,), name=name)(x)


def add_temporal_embedding(x):
    row_zero = ops.tile(ops.convert_to_tensor([0.0, 1.0]), [x.shape[-1] // 2])
    return x + row_zero


def concatenate_pooled(x):
    pooled = ops.mean(x, axis=1, keepdims=True)
    return ops.concatenate([pooled, x], axis=1)


def to_grid(x, grid):
    return ops.reshape(x, (-1, grid[0], grid[1], x.shape[-1]))


def to_tokens(x):
    return ops.reshape(x, (-1, x.shape[1] * x.shape[2], x.shape[-1]))


def build_norm(name):
    return LayerNormalization(epsilon=1e-5, name=name)
