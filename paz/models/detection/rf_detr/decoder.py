"""Two-stage deformable DETR decoder, as used by RF-DETR and LW-DETR.

A first stage scores one anchor box per feature-map cell and keeps the best
``num_queries`` of them. Those boxes become the reference points that the
decoder layers refine with deformable cross-attention. Boxes stay in
normalized ``(cx, cy, w, h)`` and are never squashed through a sigmoid:
each stage predicts a delta relative to its reference box.

This is the inference graph. Training-time group-DETR duplicates the query
tables and the first-stage heads; only the first group is used at inference,
so only the first group is built here.
"""
import numpy as np
from keras import ops
from keras.layers import Dense, LayerNormalization

from paz.models.transformers import deformable, feedforward
from paz.models.transformers.attention import attend_key
from paz.models.transformers.attention import project_biased_key
from paz.models.transformers.embeddings.sine import embed_boxes
from paz.models.transformers.embeddings.token import LearnableTokens


def build(features, num_layers, num_self_heads, num_cross_heads, num_points,
          feedforward_size, num_classes, num_queries):
    grids = read_grids(features)
    memory = flatten_levels(features)
    hidden_size = memory.shape[-1]
    selected = select_proposals(memory, grids, num_classes, num_queries)
    deltas = build_table(memory, num_queries, 4, "refpoint_embed")
    boxes = refine(deltas, selected)
    target = build_table(memory, num_queries, hidden_size, "query_feat")
    query_pos = embed_reference(boxes, hidden_size)
    sizes = num_self_heads, num_cross_heads, num_points, feedforward_size
    for layer in range(num_layers):
        args = memory, boxes, query_pos, grids
        target = apply_layer(target, *args, *sizes, f"decoder_{layer}")
    tokens = normalize(target, "decoder_norm")
    logits = Dense(num_classes, name="class_embed")(tokens)
    return logits, refine(apply_box_head(tokens, "bbox_embed"), boxes)


def select_proposals(memory, grids, num_classes, num_queries):
    """Scores one anchor per cell and keeps the best ``num_queries``.

    The feature maps must therefore hold at least ``num_queries`` cells, which
    is checked here so an undersized detector fails while it is being built
    rather than on its first forward pass.
    """
    num_cells = sum(height * width for height, width in grids)
    if num_cells < num_queries:
        message = f"Need at least {num_queries} feature cells, got {num_cells}"
        raise ValueError(message)
    anchors, valid = build_anchor_boxes(grids)
    tokens = Dense(memory.shape[-1], name="enc_output")(memory * valid)
    tokens = normalize(tokens, "enc_output_norm")
    logits = Dense(num_classes, name="enc_class_embed")(tokens)
    deltas = apply_box_head(tokens, "enc_bbox_embed")
    boxes = refine(deltas, anchors * valid)
    ranked = ops.top_k(ops.max(logits, axis=-1), num_queries)[1]
    return ops.take_along_axis(boxes, ranked[..., None], axis=1)


def build_anchor_boxes(grids):
    """One anchor per cell: cell center, side ``0.05`` doubling per level."""
    levels = []
    for level, (height, width) in enumerate(grids):
        rows, columns = build_cell_indices(height, width)
        centers = (np.stack([columns, rows], -1) + 0.5) / [width, height]
        sizes = np.full_like(centers, 0.05 * 2.0**level)
        levels.append(np.concatenate([centers, sizes], -1).reshape(-1, 4))
    anchors = np.concatenate(levels, 0)[np.newaxis].astype("float32")
    inside = (anchors > 0.01) & (anchors < 0.99)
    valid = np.all(inside, axis=-1, keepdims=True).astype("float32")
    return anchors * valid, valid


def build_cell_indices(height, width):
    rows = np.arange(height, dtype="float32")
    columns = np.arange(width, dtype="float32")
    return np.meshgrid(rows, columns, indexing="ij")


def apply_layer(target, memory, boxes, query_pos, grids, num_self_heads,
                num_cross_heads, num_points, feedforward_size, name):
    attended = apply_self_attention(target, query_pos, num_self_heads, name)
    target = normalize(target + attended, f"{name}_norm1")
    args = memory, boxes, grids, num_cross_heads, num_points
    attended = deformable.attend(target + query_pos, *args, f"{name}_cross")
    target = normalize(target + attended, f"{name}_norm2")
    names = f"{name}_linear1", f"{name}_linear2"
    hidden_size = target.shape[-1]
    forwarded = feedforward.relu(target, feedforward_size, hidden_size, *names)
    return normalize(target + forwarded, f"{name}_norm3")


def apply_self_attention(target, query_pos, num_heads, name):
    head_dim = target.shape[-1] // num_heads
    query = target + query_pos
    key = project_biased_key(query, num_heads, head_dim, name)
    return attend_key(query, target, key, None, head_dim, 0.0, name)


def embed_reference(boxes, hidden_size):
    embedded = embed_boxes(boxes, hidden_size // 2)
    names = "decoder_ref_point_head_0", "decoder_ref_point_head_1"
    return feedforward.relu(embedded, hidden_size, hidden_size, *names)


def apply_box_head(tokens, name):
    hidden_size = tokens.shape[-1]
    kwargs = dict(activation="relu")
    inner = Dense(hidden_size, name=f"{name}_0", **kwargs)(tokens)
    inner = Dense(hidden_size, name=f"{name}_1", **kwargs)(inner)
    return Dense(4, name=f"{name}_2")(inner)


def refine(deltas, reference):
    """Places ``deltas`` relative to a reference ``(cx, cy, w, h)`` box."""
    centers = deltas[..., :2] * reference[..., 2:] + reference[..., :2]
    sizes = ops.exp(deltas[..., 2:]) * reference[..., 2:]
    return ops.concatenate([centers, sizes], axis=-1)


def build_table(reference, count, hidden_size, name):
    """Broadcasts a learnable ``(count, hidden_size)`` table to the batch."""
    seed = ops.zeros_like(reference[:, :1, :1])
    return LearnableTokens(count, hidden_size, name=name)(seed)


def normalize(tokens, name):
    return LayerNormalization(epsilon=1e-5, name=name)(tokens)


def read_grids(features):
    return tuple((feature.shape[1], feature.shape[2]) for feature in features)


def flatten_levels(features):
    tokens = []
    for feature in features:
        length = feature.shape[1] * feature.shape[2]
        tokens.append(ops.reshape(feature, (-1, length, feature.shape[3])))
    return ops.concatenate(tokens, axis=1)
