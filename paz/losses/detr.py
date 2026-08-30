"""Set-prediction loss for DETR-style detectors.

Queries and ground-truth boxes are matched one-to-one with the Hungarian
algorithm, then scored with an IOU-aware binary cross entropy on the classes
and an L1 plus generalized-IOU term on the boxes. Targets are padded to a
fixed count so every tensor keeps a static shape; only the assignment runs on
the host, through ``jax.pure_callback``. The assignment is treated as fixed,
so no gradient flows into it.

``y_true`` is ``(batch, max_boxes, 5)``: normalized ``(cx, cy, w, h)`` and a
class index, padded with -1. ``y_pred`` is
``(batch, num_queries, 4 + num_classes)``: normalized ``(cx, cy, w, h)``
followed by class logits, which is what ``rf_detr.models.join_outputs``
produces.
"""
import numpy as np
import jax
import jax.numpy as jp
from keras.saving import register_keras_serializable
from scipy.optimize import linear_sum_assignment

import paz

CLASS_WEIGHT, BOX_WEIGHT, IOU_WEIGHT = 1.0, 5.0, 2.0
MATCH_CLASS_COST, MATCH_BOX_COST, MATCH_IOU_COST = 2.0, 5.0, 2.0
FOCAL_ALPHA, FOCAL_GAMMA = 0.25, 2.0
UNMATCHABLE_COST = 1e6


@register_keras_serializable("detr_loss", "call")
def call(y_true, y_pred):
    """Sums the classification, regression and generalized-IOU terms."""
    boxes, logits, targets, labels, valid = unpack(y_true, y_pred)
    queries = match(boxes, logits, targets, labels, valid)
    args = boxes, targets, valid, queries
    weighted = BOX_WEIGHT * compute_regression(*args)
    weighted = weighted + IOU_WEIGHT * compute_IOU_error(*args)
    args = boxes, logits, targets, labels, valid, queries
    return weighted + CLASS_WEIGHT * compute_classification(*args)


@register_keras_serializable("detr_loss", "classification")
def classification(y_true, y_pred):
    """IOU-aware binary cross entropy over every query and class."""
    boxes, logits, targets, labels, valid = unpack(y_true, y_pred)
    queries = match(boxes, logits, targets, labels, valid)
    args = boxes, logits, targets, labels, valid, queries
    return compute_classification(*args)


@register_keras_serializable("detr_loss", "regression")
def regression(y_true, y_pred):
    """Mean absolute error between matched boxes, in center form."""
    boxes, logits, targets, labels, valid = unpack(y_true, y_pred)
    queries = match(boxes, logits, targets, labels, valid)
    return compute_regression(boxes, targets, valid, queries)


@register_keras_serializable("detr_loss", "generalized_IOU")
def generalized_IOU(y_true, y_pred):
    """One minus the generalized IOU of every matched pair."""
    boxes, logits, targets, labels, valid = unpack(y_true, y_pred)
    queries = match(boxes, logits, targets, labels, valid)
    return compute_IOU_error(boxes, targets, valid, queries)


def compute_classification(boxes, logits, targets, labels, valid, queries):
    """Pulls matched queries towards a confidence and IOU blend.

    Coupling the two keeps a confident but badly localized query from
    scoring well. Every other query and class is a negative, down-weighted
    by its own confidence.
    """
    quality = read_matched_IOUs(boxes, targets, queries)
    scores = jax.nn.sigmoid(logits)
    matched = scatter_matches(queries, labels, valid, scores.shape)
    soft = build_soft_targets(scores, queries, labels, quality, valid)
    positive = matched * soft
    negative = jp.where(matched > 0.0, matched - positive, scores**FOCAL_GAMMA)
    entropy = -jax.nn.log_sigmoid(logits) * (positive + negative)
    return jp.sum(negative * logits + entropy) / count_boxes(valid)


def compute_regression(boxes, targets, valid, queries):
    matched = gather_queries(boxes, queries)
    errors = jp.sum(jp.abs(matched - targets), axis=-1)
    return jp.sum(errors * valid) / count_boxes(valid)


def compute_IOU_error(boxes, targets, valid, queries):
    matched = gather_queries(boxes, queries)
    overlaps = compute_paired_IOUs(matched, targets)
    return jp.sum((1.0 - overlaps) * valid) / count_boxes(valid)


def match(boxes, logits, targets, labels, valid):
    """Assigns one query per target slot, padding included but masked later."""
    costs = compute_costs(boxes, logits, targets, labels, valid)
    shape = jax.ShapeDtypeStruct(labels.shape, "int32")
    return jax.pure_callback(solve_assignments, shape, costs)


def solve_assignments(costs):
    assignments = []
    for image_costs in np.asarray(costs):
        rows, columns = linear_sum_assignment(image_costs)
        queries = np.zeros(image_costs.shape[1], "int32")
        queries[columns] = rows
        assignments.append(queries)
    return np.stack(assignments)


def compute_costs(boxes, logits, targets, labels, valid):
    class_costs = compute_class_costs(logits, labels)
    box_costs = compute_box_costs(boxes, targets)
    iou_costs = -compute_IOU_matrices(boxes, targets)
    costs = MATCH_CLASS_COST * class_costs + MATCH_BOX_COST * box_costs
    costs = costs + MATCH_IOU_COST * iou_costs
    costs = jp.where(valid[:, None, :], costs, UNMATCHABLE_COST)
    return jax.lax.stop_gradient(costs)


def compute_class_costs(logits, labels):
    """Focal cost of naming each query after each target's class."""
    scores = jax.nn.sigmoid(logits)
    negative = (1 - FOCAL_ALPHA) * scores**FOCAL_GAMMA
    negative = negative * -jp.log1p(-scores + 1e-8)
    positive = FOCAL_ALPHA * (1 - scores) ** FOCAL_GAMMA
    positive = positive * -jp.log(scores + 1e-8)
    return take_label_columns(positive - negative, labels)


def take_label_columns(costs, labels):
    return jp.take_along_axis(costs, labels[:, None, :], axis=2)


def compute_box_costs(boxes, targets):
    differences = boxes[:, :, None, :] - targets[:, None, :, :]
    return jp.sum(jp.abs(differences), axis=-1)


def compute_IOU_matrices(boxes, targets):
    overlaps = jax.vmap(paz.boxes.compute_generalized_IOUs)
    return overlaps(to_corners(boxes), to_corners(targets))


def compute_paired_IOUs(boxes, targets):
    overlaps = compute_IOU_matrices(boxes, targets)
    return jp.diagonal(overlaps, axis1=1, axis2=2)


def to_corners(boxes):
    return jax.vmap(paz.boxes.to_corner_form)(boxes)


def read_matched_IOUs(boxes, targets, queries):
    matched = gather_queries(boxes, queries)
    overlaps = compute_paired_IOUs(matched, targets)
    return jax.lax.stop_gradient(jp.clip(overlaps, 0.0, 1.0))


def gather_queries(boxes, queries):
    return jp.take_along_axis(boxes, queries[..., None], axis=1)


def scatter_matches(queries, labels, valid, shape):
    """Marks the matched (query, class) cell of every valid target."""
    selections = build_selections(queries, labels, shape)
    return jp.sum(selections * valid[:, :, None, None], axis=1)


def build_soft_targets(scores, queries, labels, quality, valid):
    matched = gather_queries(scores, queries)
    confidence = jp.take_along_axis(matched, labels[..., None], axis=2)
    confidence = jp.squeeze(confidence, axis=-1)
    blended = confidence**FOCAL_ALPHA * quality ** (1.0 - FOCAL_ALPHA)
    soft = jax.lax.stop_gradient(jp.maximum(blended, 0.01))
    selections = build_selections(queries, labels, scores.shape)
    return jp.sum(selections * (soft * valid)[:, :, None, None], axis=1)


def build_selections(queries, labels, shape):
    query_hot = jax.nn.one_hot(queries, shape[1])
    class_hot = jax.nn.one_hot(labels, shape[2])
    return query_hot[..., None] * class_hot[:, :, None, :]


def unpack(y_true, y_pred):
    boxes, logits = y_pred[:, :, :4], y_pred[:, :, 4:]
    targets, labels, valid = split_targets(y_true)
    return boxes, logits, targets, labels, valid


def split_targets(y_true):
    valid = jp.asarray(y_true[:, :, 4] >= 0.0, "float32")
    labels = jp.asarray(jp.maximum(y_true[:, :, 4], 0.0), "int32")
    return y_true[:, :, :4], labels, valid


def count_boxes(valid):
    return jp.maximum(jp.sum(valid), 1.0)
