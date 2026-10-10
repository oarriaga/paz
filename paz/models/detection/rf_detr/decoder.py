"""Two-stage deformable DETR decoder, as used by RF-DETR and LW-DETR.

A first stage scores one anchor box per feature-map cell and keeps the best
``num_queries`` of them. Those boxes become the reference points that the
decoder layers refine with deformable cross-attention. Boxes stay in
normalized ``(cx, cy, w, h)`` and are never squashed through a sigmoid:
each stage predicts a delta relative to its reference box.

``build_stages`` returns one ``(logits, boxes)`` pair per supervised stage:
the first stage, then every decoder layer. Inference reads the last pair;
training scores all of them, which is what the reference calls its auxiliary
and encoder losses.

Group-DETR duplicates the query tables and the first-stage heads
``num_groups`` times and keeps self-attention inside a group. Group zero
keeps the ungrouped layer names, so a single-group detector's weights can be
copied into it by name and only the added groups start fresh.
"""
import numpy as np
from keras import ops
from keras.layers import Dense, LayerNormalization

from paz.models.transformers import deformable, feedforward
from paz.models.transformers.attention import attend_key
from paz.models.transformers.attention import project_biased_key
from paz.models.transformers.embeddings.sine import embed_boxes
from paz.models.transformers.embeddings.token import LearnableTokens


def build_stages(features, num_layers, num_self_heads, num_cross_heads,
                 num_points, feedforward_size, num_classes, num_queries,
                 num_groups):
    grids = read_grids(features)
    memory = flatten_levels(features)
    hidden_size = memory.shape[-1]
    args = memory, grids, num_classes, num_queries, num_groups
    proposals = select_proposals(*args)
    args = memory, num_queries, 4, num_groups, "refpoint_embed"
    # The reference is detached: the first stage learns from its own loss.
    boxes = refine(build_tables(*args), ops.stop_gradient(proposals[1]))
    args = memory, num_queries, hidden_size, num_groups, "query_feat"
    target = build_tables(*args)
    query_pos = embed_reference(boxes, hidden_size)
    heads = build_prediction_heads(hidden_size, num_classes)
    sizes = num_self_heads, num_cross_heads, num_points, feedforward_size
    stages = [proposals]
    for layer in range(num_layers):
        args = memory, boxes, query_pos, grids
        name = f"decoder_{layer}"
        target = apply_layer(target, *args, *sizes, num_groups, name)
        stages.append(predict_stage(heads, target, boxes))
    return stages


def select_proposals(memory, grids, num_classes, num_queries, num_groups):
    """First-stage logits and boxes of every group's best anchors.

    The feature maps must hold at least ``num_queries`` cells, which is
    checked here so an undersized detector fails while it is being built
    rather than on its first forward pass.
    """
    num_cells = sum(height * width for height, width in grids)
    if num_cells < num_queries:
        message = f"Need at least {num_queries} feature cells, got {num_cells}"
        raise ValueError(message)
    anchors, valid = build_anchor_boxes(grids)
    groups = []
    for group in range(num_groups):
        args = memory, anchors, valid, num_classes, num_queries, group
        groups.append(select_group_proposals(*args))
    return tuple(ops.concatenate(field, axis=1) for field in zip(*groups))


def select_group_proposals(memory, anchors, valid, num_classes, num_queries,
                           group):
    hidden_size = memory.shape[-1]
    project = Dense(hidden_size, name=name_group("enc_output", group))
    norm_name = name_group("enc_output_norm", group)
    tokens = normalize(project(memory * valid), norm_name)
    classify = Dense(num_classes, name=name_group("enc_class_embed", group))
    logits = classify(tokens)
    head = build_box_head(hidden_size, name_group("enc_bbox_embed", group))
    boxes = refine(apply_box_head(head, tokens), anchors * valid)
    ranked = ops.top_k(ops.max(logits, axis=-1), num_queries)[1]
    return take_queries(logits, ranked), take_queries(boxes, ranked)


def name_group(name, group):
    """Group zero keeps the ungrouped name, matching a single-group model.

    The suffix spells out ``group`` because the box head names its own three
    layers ``_0`` to ``_2``, which a bare index would collide with.
    """
    return name if group == 0 else f"{name}_group_{group}"


def take_queries(values, ranked):
    return ops.take_along_axis(values, ranked[..., None], axis=1)


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
                num_cross_heads, num_points, feedforward_size, num_groups,
                name):
    args = target, query_pos, num_self_heads, num_groups, name
    attended = apply_self_attention(*args)
    target = normalize(target + attended, f"{name}_norm1")
    args = memory, boxes, grids, num_cross_heads, num_points
    attended = deformable.attend(target + query_pos, *args, f"{name}_cross")
    target = normalize(target + attended, f"{name}_norm2")
    names = f"{name}_linear1", f"{name}_linear2"
    hidden_size = target.shape[-1]
    forwarded = feedforward.relu(target, feedforward_size, hidden_size, *names)
    return normalize(target + forwarded, f"{name}_norm3")


def apply_self_attention(target, query_pos, num_heads, num_groups, name):
    """Attends inside one query group at a time, as group-DETR requires."""
    head_dim = target.shape[-1] // num_heads
    num_queries = target.shape[1]
    query = split_groups(target + query_pos, num_groups)
    value = split_groups(target, num_groups)
    key = project_biased_key(query, num_heads, head_dim, name)
    attended = attend_key(query, value, key, None, head_dim, 0.0, name)
    return merge_groups(attended, num_queries)


def split_groups(tokens, num_groups):
    """Folds the query groups into the batch, leaving one group per row."""
    num_queries = tokens.shape[1] // num_groups
    return ops.reshape(tokens, (-1, num_queries, tokens.shape[2]))


def merge_groups(tokens, num_queries):
    return ops.reshape(tokens, (-1, num_queries, tokens.shape[2]))


def build_prediction_heads(hidden_size, num_classes):
    """Norm, class head and box head, shared by every decoder layer."""
    normalizer = LayerNormalization(epsilon=1e-5, name="decoder_norm")
    classifier = Dense(num_classes, name="class_embed")
    return normalizer, classifier, build_box_head(hidden_size, "bbox_embed")


def predict_stage(heads, target, reference):
    normalizer, classifier, box_head = heads
    tokens = normalizer(target)
    boxes = refine(apply_box_head(box_head, tokens), reference)
    return classifier(tokens), boxes


def build_box_head(hidden_size, name):
    """The three dense layers that turn tokens into a box delta."""
    kwargs = dict(activation="relu")
    inner_0 = Dense(hidden_size, name=f"{name}_0", **kwargs)
    inner_1 = Dense(hidden_size, name=f"{name}_1", **kwargs)
    return inner_0, inner_1, Dense(4, name=f"{name}_2")


def apply_box_head(head, tokens):
    for layer in head:
        tokens = layer(tokens)
    return tokens


def embed_reference(boxes, hidden_size):
    embedded = embed_boxes(boxes, hidden_size // 2)
    names = "decoder_ref_point_head_0", "decoder_ref_point_head_1"
    return feedforward.relu(embedded, hidden_size, hidden_size, *names)


def refine(deltas, reference):
    """Places ``deltas`` relative to a reference ``(cx, cy, w, h)`` box."""
    centers = deltas[..., :2] * reference[..., 2:] + reference[..., :2]
    sizes = ops.exp(deltas[..., 2:]) * reference[..., 2:]
    return ops.concatenate([centers, sizes], axis=-1)


def build_tables(reference, count, hidden_size, num_groups, name):
    """One learnable table per query group, joined along the queries."""
    tables = []
    for group in range(num_groups):
        args = reference, count, hidden_size, name_group(name, group)
        tables.append(build_table(*args))
    return ops.concatenate(tables, axis=1)


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
