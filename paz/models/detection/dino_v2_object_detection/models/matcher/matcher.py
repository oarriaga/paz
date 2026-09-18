import numpy as np
import keras
from keras import ops
from scipy.optimize import linear_sum_assignment

from paz.models.detection.dino_v2_object_detection.utils.box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
    batch_sigmoid_ce_loss,
    batch_dice_loss,
)
from paz.models.detection.dino_v2_object_detection.models.segmentation_head.segmentation_head_keras import (  # fmt: skip
    point_sample,
)

MASK_TYPES = (keras.KerasTensor, np.ndarray)
FOCAL_GAMMA = 2.0
NON_FINITE_COST = 1e6


def hungarian_matcher(outputs, targets, cost_class=1, cost_bbox=1, cost_giou=1, focal_alpha=0.25, mask_point_sample_ratio=16, cost_mask_ce=1, cost_mask_dice=1, group_detr=1):  # fmt: skip
    assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"  # fmt: skip
    batch_size = ops.shape(outputs["pred_logits"])[0]
    num_queries = ops.shape(outputs["pred_logits"])[1]
    args = (outputs, targets, cost_class, cost_bbox, cost_giou, focal_alpha)
    weights = (cost_mask_ce, cost_mask_dice, mask_point_sample_ratio)
    cost_matrix = ops.cast(compute_cost_matrix(*args, *weights), "float32")
    batched_cost = ops.reshape(cost_matrix, (batch_size, num_queries, -1))
    target_sizes = [len(target["boxes"]) for target in targets]
    queries_per_group = num_queries // group_detr
    indices = []
    for group, group_cost in enumerate(split_costs_by_group(batched_cost, group_detr)):  # fmt: skip
        solved = solve_group_assignment(group_cost, target_sizes, batch_size)
        if group == 0:
            indices = solved
        else:
            indices = merge_group_indices(indices, solved, queries_per_group * group)  # fmt: skip
    return indices


def split_costs_by_group(batched_cost, group_detr):
    if group_detr > 1:
        groups = ops.split(batched_cost, group_detr, axis=1)
    else:
        groups = [batched_cost]
    return groups


def solve_group_assignment(group_cost, target_sizes, batch_size):
    group_indices = []
    start = 0
    for image_index in range(batch_size):
        end = start + target_sizes[image_index]
        group_indices.append(solve_assignment(group_cost[image_index][:, start:end]))  # fmt: skip
        start = end
    return group_indices


def solve_assignment(image_cost):
    if keras.backend.backend() == "tensorflow":
        rows, columns = solve_assignment_in_graph(image_cost)
    else:
        rows, columns = optimize_linear_assignment(ops.convert_to_numpy(image_cost))  # fmt: skip
        rows = ops.convert_to_tensor(rows, dtype="int64")
        columns = ops.convert_to_tensor(columns, dtype="int64")
    return rows, columns


def solve_assignment_in_graph(image_cost):
    # tf.numpy_function keeps the scipy solver callable under a TF graph.
    import tensorflow as tf
    signature = [tf.int64, tf.int64]
    return tf.numpy_function(optimize_linear_assignment, [image_cost], signature)  # fmt: skip


def merge_group_indices(indices, group_indices, offset):
    merged = []
    for (rows, columns), (new_rows, new_columns) in zip(indices, group_indices):
        shifted = ops.concatenate([rows, new_rows + offset], axis=0)
        joined = ops.concatenate([columns, new_columns], axis=0)
        merged.append((shifted, joined))
    return merged


def compute_cost_matrix(outputs, targets, cost_class, cost_bbox, cost_giou, focal_alpha, cost_mask_ce=1, cost_mask_dice=1, mask_point_sample_ratio=16):  # fmt: skip
    logits = outputs["pred_logits"]
    flat_logits = ops.reshape(logits, (-1, ops.shape(logits)[-1]))
    predicted_boxes = ops.reshape(outputs["pred_boxes"], (-1, 4))
    target_ids = ops.concatenate([t["labels"] for t in targets], axis=0)
    target_boxes = ops.concatenate([t["boxes"] for t in targets], axis=0)
    box_term = cost_bbox * compute_box_cost(predicted_boxes, target_boxes)
    class_cost = compute_class_cost(flat_logits, target_ids, focal_alpha)
    giou_cost = compute_giou_cost(predicted_boxes, target_boxes)
    cost_matrix = box_term + cost_class * class_cost + cost_giou * giou_cost
    if "masks" in targets[0]:
        args = (outputs, targets, mask_point_sample_ratio)
        mask_ce, mask_dice = compute_mask_costs(*args)
        cost_matrix = cost_matrix + cost_mask_ce * mask_ce
        cost_matrix = cost_matrix + cost_mask_dice * mask_dice
    return cost_matrix


def compute_class_cost(flat_logits, target_ids, focal_alpha):
    probabilities = ops.sigmoid(flat_logits)
    negative_weight = (1 - focal_alpha) * (probabilities**FOCAL_GAMMA)
    positive_weight = focal_alpha * ((1 - probabilities) ** FOCAL_GAMMA)
    negative_cost = negative_weight * (-log_sigmoid(-flat_logits))
    positive_cost = positive_weight * (-log_sigmoid(flat_logits))
    target_ids = ops.cast(target_ids, "int32")
    positive = ops.take(positive_cost, target_ids, axis=1)
    negative = ops.take(negative_cost, target_ids, axis=1)
    return positive - negative


def compute_box_cost(predicted_boxes, target_boxes):
    predicted = ops.expand_dims(predicted_boxes, 1)
    difference = ops.abs(predicted - ops.expand_dims(target_boxes, 0))
    return ops.sum(difference, axis=-1)


def compute_giou_cost(predicted_boxes, target_boxes):
    predicted_xyxy = box_cxcywh_to_xyxy(predicted_boxes)
    target_xyxy = box_cxcywh_to_xyxy(target_boxes)
    return -generalized_box_iou(predicted_xyxy, target_xyxy)


def has_dense_masks(outputs):
    masks = outputs.get("pred_masks", None)
    if masks is None:
        dense = False
    else:
        dense = ops.is_tensor(masks) or isinstance(masks, MASK_TYPES)
    return dense


def compute_mask_costs(outputs, targets, mask_point_sample_ratio):
    target_masks = ops.concatenate([t["masks"] for t in targets], axis=0)
    if has_dense_masks(outputs):
        sample = sample_dense_mask_logits
    else:
        sample = sample_lazy_mask_logits
    mask_logits, point_coordinates = sample(outputs, mask_point_sample_ratio)
    args = (target_masks, point_coordinates, mask_logits)
    sampled_targets = sample_target_masks(*args)
    mask_ce = batch_sigmoid_ce_loss(mask_logits, sampled_targets)
    return mask_ce, batch_dice_loss(mask_logits, sampled_targets)


def sample_dense_mask_logits(outputs, mask_point_sample_ratio):
    masks = outputs["pred_masks"]
    height, width = ops.shape(masks)[-2], ops.shape(masks)[-1]
    masks = ops.reshape(masks, (-1, height, width))
    num_points = (height * width) // mask_point_sample_ratio
    point_coordinates = sample_point_coordinates(num_points)
    shape = (ops.shape(masks)[0], num_points, 2)
    coordinates = ops.broadcast_to(point_coordinates, shape)
    sampled = sample_at_points(ops.expand_dims(masks, 1), coordinates)
    return ops.squeeze(sampled, 1), point_coordinates


def sample_lazy_mask_logits(outputs, mask_point_sample_ratio):
    spatial = outputs["pred_masks"]["spatial_features"]
    queries = outputs["pred_masks"]["query_features"]
    bias = outputs["pred_masks"]["bias"]
    area = ops.shape(spatial)[-2] * ops.shape(spatial)[-1]
    num_points = area // mask_point_sample_ratio
    point_coordinates = sample_point_coordinates(num_points)
    shape = (ops.shape(spatial)[0], num_points, 2)
    coordinates = ops.broadcast_to(point_coordinates, shape)
    sampled = sample_at_points(spatial, coordinates)
    logits = ops.einsum("bcp,bnc->bnp", sampled, queries) + bias
    return ops.reshape(logits, (-1, ops.shape(logits)[-1])), point_coordinates


def sample_target_masks(target_masks, point_coordinates, mask_logits):
    target_masks = ops.cast(target_masks, mask_logits.dtype)
    num_points = ops.shape(point_coordinates)[1]
    shape = (ops.shape(target_masks)[0], num_points, 2)
    coordinates = ops.broadcast_to(point_coordinates, shape)
    expanded = ops.expand_dims(target_masks, 1)
    return ops.squeeze(sample_at_points(expanded, coordinates), 1)


def sample_point_coordinates(num_points):
    return keras.random.uniform((1, num_points, 2), minval=0.0, maxval=1.0)


def sample_at_points(masks, coordinates):
    return point_sample(masks, coordinates, align_corners=False)


def log_sigmoid(x):
    # log(sigmoid(x)) = -softplus(-x), the numerically stable form
    return -ops.softplus(-x)


def optimize_linear_assignment(cost_matrix):
    # Host-by-design: the Hungarian algorithm is not jittable, so this stays
    # on numpy/scipy and must not be converted to keras.ops.
    cost_matrix = np.array(cost_matrix)
    # Replace non-finite values to ensure the solver converges
    cost_matrix[np.isinf(cost_matrix) | np.isnan(cost_matrix)] = NON_FINITE_COST
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    return row_indices.astype(np.int64), col_indices.astype(np.int64)
