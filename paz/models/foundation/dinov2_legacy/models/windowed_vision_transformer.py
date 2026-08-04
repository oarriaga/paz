import math
from collections import namedtuple

import keras
from keras import Input, Model, layers, ops, initializers

from paz.models.foundation.dinov2_legacy.layers import (
    attend,
    mlp,
    apply_layer_scale,
    swiglu_ffn_fused,
)

CLS_TOKEN = "cls_token"
POS_EMBED = "position_embeddings"
REGISTER_TOKENS = "register_tokens"
PATCH_PROJECTION = "patch_embeddings_projection"
EMBEDDINGS = "embeddings"
ENCODER = "encoder"
ENCODER_LAYER = "layer_{}"
FINAL_NORM = "layernorm"
NORM1 = "norm1"
NORM2 = "norm2"
ATTENTION = "attention"
LAYER_SCALE1 = "ls1"
LAYER_SCALE2 = "ls2"
DROP_PATH1 = "drop_path1"
DROP_PATH2 = "drop_path2"
FFN = "mlp"

EmbedArgs = namedtuple(
    "EmbedArgs", "patch_size hidden_size num_register_tokens num_windows"
)
LayerArgs = namedtuple(
    "LayerArgs", "hidden_size num_heads mlp_ratio use_swiglu init_values drop_path_rate"
)
ModelArgs = namedtuple(
    "ModelArgs", "image_size embed block depth num_windows window_block_indexes"
)


def Kernel(stddev=0.02):
    return initializers.TruncatedNormal(stddev=stddev)


# ─────────────────────────────────────────────────────────────
#  Patch embeddings + windowing
# ─────────────────────────────────────────────────────────────


def build_patch_embeddings(pixels, embed, image_size, name):
    grid = compute_patch_grid(image_size, embed.patch_size)
    patches = project_patches(pixels, embed.hidden_size, embed.patch_size, name)
    tokens = prepend_cls_token(patches, embed.hidden_size, name)
    tokens = add_position_embeddings(tokens, embed, image_size, name)
    tokens = apply_windowing(tokens, embed, grid)
    tokens = insert_register_tokens(tokens, embed, name)
    return tokens


def project_patches(pixels, hidden_size, patch_size, name):
    args = dict(kernel_size=patch_size, strides=patch_size, padding="valid")
    conv = layers.Conv2D(hidden_size, name=f"{name}_{PATCH_PROJECTION}", **args)
    projected = conv(pixels)
    return layers.Reshape((-1, hidden_size))(projected)


def prepend_cls_token(patches, hidden_size, name):
    table = lookup_constant(f"{name}_{CLS_TOKEN}", 1, hidden_size, Kernel(), patches)
    broadcasted = broadcast_token(table, patches)
    return ops.concatenate([broadcasted, patches], axis=1)


def add_position_embeddings(tokens, embed, image_size, name):
    num = compute_patch_grid(image_size, embed.patch_size)
    pos_num = num[0] * num[1] + 1
    table = lookup_constant(
        f"{name}_{POS_EMBED}", pos_num, embed.hidden_size, Kernel(), tokens
    )
    height, width = image_size[0], image_size[1]
    pos = interpolate_pos_encoding(table, tokens, embed, height, width)
    return tokens + pos


def apply_windowing(tokens, embed, grid):
    if embed.num_windows <= 1:
        return tokens
    args = (tokens, embed.hidden_size, embed.num_windows, grid)
    return window_partition(*args)


def window_partition(tokens, hidden_size, num_windows, grid):
    num_h, num_w = grid
    cls = tokens[:, :1]
    pixels = tokens[:, 1:]
    pixels = ops.reshape(pixels, (-1, num_h, num_w, hidden_size))
    pixels, num_h, num_w = pad_to_windows(pixels, num_h, num_w, num_windows)
    pixels = group_windows(pixels, num_h, num_w, num_windows)
    cls = group_window_cls(cls, pixels, num_windows)
    return ops.concatenate([cls, pixels], axis=1)


def pad_to_windows(pixels, num_h, num_w, num_windows):
    pad_h = (num_windows - num_h % num_windows) % num_windows
    pad_w = (num_windows - num_w % num_windows) % num_windows
    if pad_h > 0 or pad_w > 0:
        pixels = ops.pad(pixels, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
    return pixels, num_h + pad_h, num_w + pad_w


def group_windows(pixels, num_h, num_w, num_windows):
    h_win = num_h // num_windows
    w_win = num_w // num_windows
    shape = (-1, num_windows, h_win, num_windows, w_win, pixels.shape[-1])
    grouped = ops.reshape(pixels, shape)
    grouped = ops.transpose(grouped, (0, 1, 3, 2, 4, 5))
    return ops.reshape(grouped, (-1, h_win * w_win, pixels.shape[-1]))


def group_window_cls(cls, windowed_pixels, num_windows):
    squared = num_windows**2
    repeated = ops.repeat(cls, squared, axis=0)
    anchor = windowed_pixels[:, :1, :] * 0.0
    return anchor + repeated


def insert_register_tokens(tokens, embed, name):
    if embed.num_register_tokens == 0:
        return tokens
    num = embed.num_register_tokens
    table = lookup_constant(
        f"{name}_{REGISTER_TOKENS}", num, embed.hidden_size, Kernel(), tokens
    )
    broadcasted = broadcast_tokens(table, tokens)
    return ops.concatenate([tokens[:, :1], broadcasted, tokens[:, 1:]], axis=1)


def lookup_constant(name, num, hidden_size, initializer, anchor):
    args = dict(embeddings_initializer=initializer, name=name)
    embedding = layers.Embedding(num, hidden_size, **args)
    indices = build_indices(num, anchor, name)
    return embedding(indices)


def build_indices(num, anchor, name):
    layer = layers.Lambda(
        arange_indices,
        arguments={"num": num},
        output_shape=(num,),
        name=f"{name}_indices",
    )
    return layer(anchor)


@keras.saving.register_keras_serializable(package="backbone")
def arange_indices(reference, num=0):
    return ops.arange(num, dtype="int32")


def broadcast_token(token, reference):
    zeros = reference[:, :1, :] * 0.0
    return zeros + token


def broadcast_tokens(tokens, reference):
    num = tokens.shape[1]
    zeros = reference[:, :num, :] * 0.0
    return zeros + tokens


def interpolate_pos_encoding(table, tokens, embed, height, width):
    num_patches = tokens.shape[1] - 1
    num_positions = table.shape[1] - 1
    same = num_patches == num_positions and height == width
    if same and height == embed.patch_size * int(math.sqrt(num_positions)):
        return table
    class_pos = table[:, :1]
    patch_pos = table[:, 1:]
    dim = table.shape[-1]
    side = int(math.sqrt(num_positions))
    h0 = height // embed.patch_size
    w0 = width // embed.patch_size
    patch_pos = ops.reshape(patch_pos, (1, side, side, dim))
    patch_pos = ops.image.resize(patch_pos, (h0, w0), interpolation="bicubic")
    patch_pos = ops.reshape(patch_pos, (1, -1, dim))
    return ops.concatenate([class_pos, patch_pos], axis=1)


def compute_patch_grid(image_size, patch_size):
    return image_size[0] // patch_size, image_size[1] // patch_size


# ─────────────────────────────────────────────────────────────
#  Transformer block (windowed / full attention)
# ─────────────────────────────────────────────────────────────


def windowed_block(x, block, num_windows, run_full_attention, drop_path, name):
    normalization = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_{NORM1}")
    normed = normalization(x)
    attended = run_attention(normed, block, num_windows, run_full_attention, name)
    x = merge_residual(x, attended, block, drop_path, name, 1)
    normalization = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_{NORM2}")
    normed = normalization(x)
    forwarded = apply_FFN(normed, block, name)
    x = merge_residual(x, forwarded, block, drop_path, name, 2)
    return x


def run_attention(x, block, num_windows, run_full_attention, name):
    full = run_full_attention and num_windows > 1
    if full:
        x = merge_windows(x, num_windows)
    attended = attention_projection(x, block, name)
    if full:
        attended = split_windows(attended, num_windows)
    return attended


def attention_projection(x, block, name):
    head_dim = block.hidden_size // block.num_heads
    args = (block.num_heads, head_dim, 0.0, 0.0, True, True, f"{name}_{ATTENTION}")
    return attend(x, *args)


def merge_windows(x, num_windows):
    squared = num_windows**2
    tokens = x.shape[1]
    channels = x.shape[-1]
    return ops.reshape(x, (-1, squared * tokens, channels))


def split_windows(x, num_windows):
    squared = num_windows**2
    tokens = x.shape[1] // squared
    channels = x.shape[-1]
    return ops.reshape(x, (-1, tokens, channels))


def apply_FFN(x, block, name):
    hidden = int(block.hidden_size * block.mlp_ratio)
    if block.use_swiglu:
        args = (x, hidden, block.hidden_size, True, 0.0, f"{name}_{FFN}", "silu")
        return swiglu_ffn_fused(*args)
    args = (x, hidden, block.hidden_size, True, 0.0, f"{name}_{FFN}", "gelu")
    return mlp(*args)


def merge_residual(x, branch, block, drop_path, name, index):
    scale_name = f"{name}_{LAYER_SCALE1 if index == 1 else LAYER_SCALE2}"
    scaled = apply_layer_scale(branch, block.hidden_size, block.init_values, scale_name)
    drop_name = f"{name}_{DROP_PATH1 if index == 1 else DROP_PATH2}"
    dropped = build_drop_path(scaled, drop_path, drop_name)
    return layers.Add(name=f"{name}_add{index}")([x, dropped])


def build_drop_path(x, rate, name):
    if rate > 0.0:
        return layers.Dropout(rate, noise_shape=(None, 1, 1), name=name)(x)
    return layers.Identity(name=name)(x)


# ─────────────────────────────────────────────────────────────
#  Encoder + collected hidden states
# ─────────────────────────────────────────────────────────────


def build_encoder_blocks(tokens, block, depth, num_windows, window_block_indexes, name):  # fmt: skip
    drop_rates = compute_drop_path_rates(block.drop_path_rate, depth)
    hidden_states = []
    for i in range(depth):
        run_full_attention = i not in window_block_indexes
        layer_name = f"{name}_{ENCODER_LAYER.format(i)}"
        tokens = windowed_block(
            tokens, block, num_windows, run_full_attention, drop_rates[i], layer_name
        )
        hidden_states.append(tokens)
    return hidden_states


def compute_drop_path_rates(drop_path_rate, depth):
    if depth <= 1:
        return [0.0] * depth
    return [x * drop_path_rate / (depth - 1) for x in range(depth)]


# ─────────────────────────────────────────────────────────────
#  Full feature-map model (used by the DinoV2 backbone wrapper)
# ─────────────────────────────────────────────────────────────


def build_feature_model(model, out_feature_indexes, name):
    pixels = Input((model.image_size[0], model.image_size[1], 3), name="pixels")
    blocks = run_backbone(pixels, model)
    grid = compute_patch_grid(model.image_size, model.embed.patch_size)
    features = collect_features(
        blocks, model.embed, model.num_windows, grid, out_feature_indexes
    )
    return Model(pixels, features, name=name)


def run_backbone(pixels, model):
    tokens = build_patch_embeddings(pixels, model.embed, model.image_size, EMBEDDINGS)
    return build_encoder_blocks(
        tokens, model.block, model.depth, model.num_windows,
        model.window_block_indexes, ENCODER,
    )


def collect_features(blocks, embed, num_windows, grid, out_feature_indexes):
    normalization = layers.LayerNormalization(epsilon=1e-6, name=FINAL_NORM)
    features = []
    for i in out_feature_indexes:
        normalized = normalization(blocks[i])
        feature = drop_special_tokens(normalized, embed.num_register_tokens)
        feature = window_unmerge(feature, embed.hidden_size, num_windows, grid)
        features.append(feature)
    return features


def apply_final_norm(tokens):
    layer = layers.LayerNormalization(epsilon=1e-6, name=FINAL_NORM)
    return layer(tokens)


def drop_special_tokens(tokens, num_register_tokens):
    start = 1 + num_register_tokens
    return tokens[:, start:]


def window_unmerge(feature, hidden_size, num_windows, grid):
    num_h, num_w = grid
    if num_windows <= 1:
        return ops.reshape(feature, (-1, num_h, num_w, hidden_size))
    return unmerge_windows(feature, hidden_size, num_windows, grid)


def unmerge_windows(feature, hidden_size, num_windows, grid):
    num_h, num_w = grid
    pad_h = (num_windows - num_h % num_windows) % num_windows
    pad_w = (num_windows - num_w % num_windows) % num_windows
    padded_h, padded_w = num_h + pad_h, num_w + pad_w
    h_win, w_win = padded_h // num_windows, padded_w // num_windows
    shape = (-1, num_windows, num_windows, h_win, w_win, hidden_size)
    feature = ops.reshape(feature, shape)
    feature = ops.transpose(feature, (0, 1, 3, 2, 4, 5))
    feature = ops.reshape(feature, (-1, padded_h, padded_w, hidden_size))
    if pad_h > 0 or pad_w > 0:
        feature = feature[:, :num_h, :num_w, :]
    return feature


# ─────────────────────────────────────────────────────────────
#  Sequence model (seq_out + all hidden states)
# ─────────────────────────────────────────────────────────────


def build_sequence_model(model, name):
    pixels = Input((model.image_size[0], model.image_size[1], 3), name="pixels")
    blocks = run_backbone(pixels, model)
    sequence = apply_final_norm(blocks[-1])
    return Model(pixels, [sequence, *blocks], name=name)


# ─────────────────────────────────────────────────────────────
#  Public symbols: builders returning sequence models / tensors
# ─────────────────────────────────────────────────────────────


def WindowedDinov2PatchEmbeddings(pixels, image_size, patch_size, hidden_size, num_register_tokens=0, num_windows=1, name=EMBEDDINGS):  # fmt: skip
    embed = EmbedArgs(patch_size, hidden_size, num_register_tokens, num_windows)
    return build_patch_embeddings(pixels, embed, to_pair(image_size), name)


def WindowedDinov2Layer(x, hidden_size, num_attention_heads, mlp_ratio=4.0, use_swiglu_ffn=False, num_windows=1, init_values=1e-5, drop_path_rate=0.0, run_full_attention=False, name="layer_0"):  # fmt: skip
    block = make_block_args(
        hidden_size, num_attention_heads, mlp_ratio, use_swiglu_ffn,
        init_values, drop_path_rate,
    )
    return windowed_block(x, block, num_windows, run_full_attention, drop_path_rate, name)  # fmt: skip


def WindowedDinov2Encoder(tokens, hidden_size, num_hidden_layers, num_attention_heads, mlp_ratio=4.0, use_swiglu_ffn=False, num_windows=1, window_block_indexes=None, init_values=1e-5, drop_path_rate=0.0, name=ENCODER):  # fmt: skip
    if window_block_indexes is None:
        window_block_indexes = list(range(num_hidden_layers))
    block = make_block_args(
        hidden_size, num_attention_heads, mlp_ratio, use_swiglu_ffn,
        init_values, drop_path_rate,
    )
    return build_encoder_blocks(
        tokens, block, num_hidden_layers, num_windows, window_block_indexes, name
    )


def WindowedDinov2Model(image_size=518, patch_size=14, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, mlp_ratio=4.0, use_swiglu_ffn=False, num_register_tokens=0, num_windows=1, window_block_indexes=None, init_values=1e-5, drop_path_rate=0.0, out_feature_indexes=None, name="dinov2_model", **kwargs):  # fmt: skip
    model = make_model_args(
        image_size, patch_size, hidden_size, num_hidden_layers,
        num_attention_heads, mlp_ratio, use_swiglu_ffn, num_register_tokens,
        num_windows, window_block_indexes, init_values, drop_path_rate,
    )
    if out_feature_indexes is not None:
        return build_feature_model(model, out_feature_indexes, name)
    return build_sequence_model(model, name)


def make_model_args(image_size, patch_size, hidden_size, num_hidden_layers, num_attention_heads, mlp_ratio, use_swiglu_ffn, num_register_tokens, num_windows, window_block_indexes, init_values, drop_path_rate):  # fmt: skip
    image_size = to_pair(image_size)
    if window_block_indexes is None:
        window_block_indexes = list(range(num_hidden_layers))
    embed = EmbedArgs(patch_size, hidden_size, num_register_tokens, num_windows)
    block = make_block_args(
        hidden_size, num_attention_heads, mlp_ratio, use_swiglu_ffn,
        init_values, drop_path_rate,
    )
    return ModelArgs(
        image_size, embed, block, num_hidden_layers, num_windows, window_block_indexes
    )


def make_block_args(hidden_size, num_heads, mlp_ratio, use_swiglu, init_values, drop_path_rate):  # fmt: skip
    return LayerArgs(
        hidden_size, num_heads, mlp_ratio, use_swiglu, init_values, drop_path_rate
    )


def to_pair(size):
    if isinstance(size, (list, tuple)):
        return tuple(size)
    return (size, size)


# ─────────────────────────────────────────────────────────────
#  Variant builders
# ─────────────────────────────────────────────────────────────


def dinov2_windowed_small(img_size=518, patch_size=14, number_of_register_tokens=0, num_windows=37, window_block_indexes=None, **kwargs):  # fmt: skip
    if window_block_indexes is None:
        window_block_indexes = default_window_indexes(12)
    return WindowedDinov2Model(
        image_size=img_size, patch_size=patch_size,
        num_register_tokens=number_of_register_tokens, hidden_size=384,
        num_hidden_layers=12, num_attention_heads=6, mlp_ratio=4.0,
        num_windows=num_windows, window_block_indexes=window_block_indexes,
        **kwargs,
    )


def dinov2_windowed_base(img_size=518, patch_size=14, number_of_register_tokens=0, num_windows=37, window_block_indexes=None, **kwargs):  # fmt: skip
    if window_block_indexes is None:
        window_block_indexes = default_window_indexes(12)
    return WindowedDinov2Model(
        image_size=img_size, patch_size=patch_size,
        num_register_tokens=number_of_register_tokens, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, mlp_ratio=4.0,
        num_windows=num_windows, window_block_indexes=window_block_indexes,
        **kwargs,
    )


def dinov2_windowed_large(img_size=518, patch_size=14, number_of_register_tokens=0, num_windows=37, window_block_indexes=None, **kwargs):  # fmt: skip
    if window_block_indexes is None:
        window_block_indexes = default_window_indexes(24)
    return WindowedDinov2Model(
        image_size=img_size, patch_size=patch_size,
        num_register_tokens=number_of_register_tokens, hidden_size=1024,
        num_hidden_layers=24, num_attention_heads=16, mlp_ratio=4.0,
        num_windows=num_windows, window_block_indexes=window_block_indexes,
        **kwargs,
    )


def dinov2_windowed_giant(img_size=518, patch_size=14, number_of_register_tokens=0, num_windows=37, window_block_indexes=None, **kwargs):  # fmt: skip
    if window_block_indexes is None:
        window_block_indexes = list(range(0, 39))
    return WindowedDinov2Model(
        image_size=img_size, patch_size=patch_size,
        num_register_tokens=number_of_register_tokens, hidden_size=1536,
        num_hidden_layers=40, num_attention_heads=24, mlp_ratio=4.0,
        use_swiglu_ffn=True, num_windows=num_windows,
        window_block_indexes=window_block_indexes, **kwargs,
    )


def default_window_indexes(depth):
    indexes = []
    for start in range(0, depth, 3):
        indexes.extend(range(start, min(start + 2, depth)))
    return indexes
