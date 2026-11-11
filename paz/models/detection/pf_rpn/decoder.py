"""Grounding DINO decoder with confidence-guided query selection.

A first stage scores one anchor box per image cell and keeps ``num_queries``
of them: half by predicted confidence, half by similarity to the pseudo-text
tokens. Each decoder layer then refines those boxes in logit space.
"""
import numpy as np
from keras import Sequential, ops
from keras.layers import Dense

from paz.models.detection.rf_detr.decoder import apply_self_attention
from paz.models.detection.rf_detr.decoder import build_anchor_boxes
from paz.models.detection.rf_detr.decoder import build_table, normalize
from paz.models.transformers import deformable, feedforward
from paz.models.transformers.attention import masked_attend
from paz.models.transformers.embeddings.sine import embed_boxes


def build(memory, texts, grids, num_layers, num_queries):
    """Returns per-query scores and normalized ``(cx, cy, w, h)`` boxes."""
    hidden_size = memory.shape[-1]
    boxes = select_boxes(memory, texts, grids, num_layers, num_queries)
    args = memory, num_queries, hidden_size
    query = build_table(*args, "query_embedding_weight")
    args = build_reference_head(hidden_size), memory, texts, grids
    last = num_layers - 1
    for layer in range(last):
        query = apply_layer(query, boxes, *args, f"decoder_layers_{layer}")
        deltas = apply_box_head(query, f"bbox_head_reg_branches_{layer}")
        boxes = refine(deltas, boxes, 1e-3)
    query = apply_layer(query, boxes, *args, f"decoder_layers_{last}")
    tokens = normalize(query, "decoder_norm")
    deltas = apply_box_head(tokens, f"bbox_head_reg_branches_{last}")
    return score(tokens, texts, last), refine(deltas, boxes, 1e-5)


def select_boxes(memory, texts, grids, stage, num_queries):
    if memory.shape[1] < num_queries:
        message = f"Need at least {num_queries} cells, got {memory.shape[1]}"
        raise ValueError(message)
    anchors, valid = build_anchor_boxes(grids)
    tokens = Dense(memory.shape[-1], name="memory_trans_fc")(memory * valid)
    tokens = normalize(tokens, "memory_trans_norm")
    similarity = compute_similarity(tokens, texts, stage)
    confidence = compute_confidence(tokens, stage)
    selected = select_cells(confidence, similarity, num_queries)
    deltas = apply_box_head(tokens, f"bbox_head_reg_branches_{stage}")
    # Invalid anchors get an infinite logit, which the sigmoid turns into 1.
    with np.errstate(divide="ignore"):
        logits = np.where(valid > 0, np.log(anchors / (1 - anchors)), np.inf)
    boxes = ops.sigmoid(deltas + logits)
    return ops.take_along_axis(boxes, selected[..., None], axis=1)


def select_cells(confidence, similarity, num_queries):
    """Half the cells by confidence, the rest by similarity, no repeats."""
    half = num_queries // 2
    confident = ops.top_k(confidence, half)[1]
    taken = ops.max(ops.one_hot(confident, confidence.shape[1]), axis=1)
    remaining = ops.where(taken > 0, -1.0, similarity)
    similar = ops.top_k(remaining, num_queries - half)[1]
    return ops.concatenate([confident, similar], axis=1)


def score(tokens, texts, stage):
    """Text similarity plus confidence, as the published model ranks boxes."""
    # Upstream squashes the confidence twice; the weights were tuned with it.
    confidence = ops.sigmoid(compute_confidence(tokens, stage))
    return compute_similarity(tokens, texts, stage) + confidence


def compute_similarity(tokens, texts, stage):
    """Sigmoid of the best match of every token against the text tokens."""
    name = f"bbox_head_cls_branches_{stage}_log_scale"
    scale = ops.exp(build_table(tokens, 1, 1, name))
    logits = ops.matmul(tokens, ops.transpose(texts, (0, 2, 1))) * scale
    return ops.max(ops.sigmoid(logits), axis=-1)


def compute_confidence(tokens, stage):
    names = [f"bbox_head_conf_branches_{stage}_{index}" for index in (0, 2)]
    logits = feedforward.relu(tokens, tokens.shape[-1], 1, *names)
    return ops.sigmoid(logits[..., 0])


def apply_box_head(tokens, name):
    hidden_size = tokens.shape[-1]
    inner = Dense(hidden_size, "relu", name=f"{name}_0")(tokens)
    names = f"{name}_2", f"{name}_4"
    return feedforward.relu(inner, hidden_size, 4, *names)


def build_reference_head(hidden_size):
    """Maps sine-embedded boxes to query positions; shared by all layers."""
    name = "decoder_ref_point_head_layers"
    inner = Dense(hidden_size, "relu", name=f"{name}_0")
    return Sequential([inner, Dense(hidden_size, name=f"{name}_1")])


def apply_layer(query, boxes, embed, memory, texts, grids, name):
    hidden_size = query.shape[-1]
    query_pos = embed(embed_boxes(boxes, hidden_size // 2))
    attended = apply_self_attention(query, query_pos, 8, f"{name}_self_attn")
    query = normalize(query + attended, f"{name}_norms_0")
    args = texts, None, 8, hidden_size // 8, 0.0, f"{name}_cross_attn_text"
    attended = masked_attend(query + query_pos, *args)
    query = normalize(query + attended, f"{name}_norms_1")
    args = memory, boxes, grids, 8, 4, f"{name}_cross_attn"
    attended = deformable.attend(query + query_pos, *args)
    query = normalize(query + attended, f"{name}_norms_2")
    names = f"{name}_ffn_layers_0_0", f"{name}_ffn_layers_1"
    forwarded = feedforward.relu(query, 2048, hidden_size, *names)
    return normalize(query + forwarded, f"{name}_norms_3")


def refine(deltas, boxes, epsilon):
    """Moves ``boxes`` by ``deltas`` in logit space."""
    boxes = ops.clip(boxes, 0.0, 1.0)
    odds = ops.maximum(boxes, epsilon) / ops.maximum(1 - boxes, epsilon)
    return ops.sigmoid(deltas + ops.log(odds))
