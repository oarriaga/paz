"""Grounding DINO encoder: image cells and text tokens refine each other.

Every layer first fuses both streams with attention in both directions, then
updates the text tokens with self-attention and the image cells with
multi-scale deformable attention around their own centers.
"""
import numpy as np
from keras import ops
from keras.layers import Dense

from paz.models.detection.rf_detr.decoder import apply_self_attention
from paz.models.detection.rf_detr.decoder import build_cell_indices
from paz.models.detection.rf_detr.decoder import build_table, normalize
from paz.models.foundation.sam2.memory_encoder import interleave_sine_cosine
from paz.models.foundation.sam2.memory_encoder import normalized_axis
from paz.models.transformers import deformable, feedforward
from paz.models.transformers.attention import merge_attention_heads
from paz.models.transformers.attention import transpose_to_heads


def build(memory, texts, grids, num_layers):
    positions = embed_positions(memory, grids)
    centers = build_centers(grids)
    for layer in range(num_layers):
        name = f"encoder_fusion_layers_{layer}"
        memory, texts = fuse(memory, texts, 4, 1024, name)
        texts = apply_text_layer(texts, f"encoder_text_layers_{layer}")
        args = positions, centers, grids, f"encoder_layers_{layer}"
        memory = apply_layer(memory, *args)
    return memory, texts


def embed_positions(memory, grids):
    """Sine position of every cell plus a learned embedding of its level."""
    levels = build_table(memory, len(grids), memory.shape[-1], "level_embed")
    sines, level_args = [], []
    for level, (height, width) in enumerate(grids):
        sines.append(embed_cells(height, width, memory.shape[-1] // 2))
        level_args.extend([level] * height * width)
    return ops.take(levels, level_args, axis=1) + np.concatenate(sines)


def embed_cells(height, width, half):
    rows = normalized_axis(height) * 2 * np.pi
    columns = normalized_axis(width) * 2 * np.pi
    grid_y, grid_x = np.meshgrid(rows, columns, indexing="ij")
    # Grounding DINO lowers the usual temperature of 10000 to 20.
    encoded_y = interleave_sine_cosine(grid_y, half, 20)
    encoded_x = interleave_sine_cosine(grid_x, half, 20)
    encoded = np.concatenate([encoded_y, encoded_x], axis=-1)
    return encoded.reshape(-1, 2 * half).astype("float32")


def build_centers(grids):
    """Normalized ``(x, y)`` center of every cell of every level."""
    levels = []
    for height, width in grids:
        rows, columns = build_cell_indices(height, width)
        centers = (np.stack([columns, rows], -1) + 0.5) / [width, height]
        levels.append(centers.reshape(-1, 2))
    return np.concatenate(levels)[np.newaxis].astype("float32")


def fuse(memory, texts, num_heads, embed_dim, name):
    """Cells read from tokens and tokens from cells, sharing their scores."""
    memory = normalize(memory, f"{name}_layer_norm_v")
    texts = normalize(texts, f"{name}_layer_norm_l")
    args = embed_dim, num_heads
    query = project(memory, *args, f"{name}_cells_query")
    key = project(texts, *args, f"{name}_texts_key")
    scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
    scores = scores * (embed_dim // num_heads) ** -0.5
    to_cells = transfer(scores, texts, memory, *args, f"{name}_cells")
    scores = ops.transpose(scores, (0, 1, 3, 2))
    to_texts = transfer(scores, memory, texts, *args, f"{name}_texts")
    return memory + to_cells, texts + to_texts


def transfer(scores, source, target, embed_dim, num_heads, name):
    """Update of ``target`` read from ``source``, with a learned scale."""
    hidden_size = target.shape[-1]
    values = project(source, embed_dim, num_heads, f"{name}_values")
    context = ops.matmul(ops.softmax(scores, axis=-1), values)
    context = merge_attention_heads(context)
    update = Dense(hidden_size, name=f"{name}_output")(context)
    return update * build_table(target, 1, hidden_size, f"{name}_scale")


def project(tokens, embed_dim, num_heads, name):
    projected = Dense(embed_dim, name=name)(tokens)
    shape = (-1, tokens.shape[1], num_heads, embed_dim // num_heads)
    return transpose_to_heads(ops.reshape(projected, shape))


def apply_text_layer(texts, name):
    # Every pseudo token sits at text position zero, whose sine embedding is
    # this constant.
    position = np.tile([0.0, 1.0], texts.shape[-1] // 2).astype("float32")
    attended = apply_self_attention(texts, position, 4, f"{name}_self_attn")
    texts = normalize(texts + attended, f"{name}_norms_0")
    names = f"{name}_ffn_layers_0_0", f"{name}_ffn_layers_1"
    forwarded = feedforward.relu(texts, 1024, texts.shape[-1], *names)
    return normalize(texts + forwarded, f"{name}_norms_1")


def apply_layer(memory, positions, centers, grids, name):
    args = memory, centers, grids, 8, 4, f"{name}_self_attn"
    attended = deformable.attend_points(memory + positions, *args)
    memory = normalize(memory + attended, f"{name}_norms_0")
    names = f"{name}_ffn_layers_0_0", f"{name}_ffn_layers_1"
    forwarded = feedforward.relu(memory, 2048, memory.shape[-1], *names)
    return normalize(memory + forwarded, f"{name}_norms_1")
