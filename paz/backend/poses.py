import cv2
import numpy as np


def match_poses(boxes, poses, prior_boxes, iou_threshold=0.5):
    """Assigns ground-truth poses to prior boxes by IoU, appending a positive
    flag column. Returns an array of shape `(num_priors, poses_dim + 1)`."""
    matched_poses = np.zeros((prior_boxes.shape[0], poses.shape[1] + 1))
    ious = compute_ious(boxes, to_corner_form(np.float32(prior_boxes)))
    per_prior_iou = np.max(ious, axis=0)
    per_prior_arg = np.argmax(ious, axis=0)
    per_box_arg = np.argmax(ious, axis=1)
    per_prior_iou[per_box_arg] = 2.0
    for box_arg in range(len(per_box_arg)):
        per_prior_arg[per_box_arg[box_arg]] = box_arg
    matched_poses[:, :-1] = poses[per_prior_arg]
    matched_poses[per_prior_iou >= iou_threshold, -1] = 1.0
    return matched_poses


def rotation_matrix_to_axis_angle(rotations, num_pose_dims=3):
    """Converts flattened rotation matrices `(n, 9)` to normalized axis-angle
    targets `(n, num_pose_dims + 2)` with the angle divided by pi."""
    axis_angles = []
    for rotation in rotations:
        target = np.zeros(num_pose_dims + 2)
        matrix = np.reshape(rotation, (num_pose_dims, num_pose_dims))
        vector, _ = cv2.Rodrigues(matrix)
        target[:num_pose_dims] = np.squeeze(vector) / np.pi
        axis_angles.append(target[np.newaxis])
    return np.concatenate(axis_angles, axis=0)


def concatenate_poses(rotations, translations):
    return np.concatenate((rotations, translations), axis=-1)


def concatenate_scale(poses, scale):
    scale = np.repeat(scale, poses.shape[0])[np.newaxis, :]
    return np.concatenate((poses, scale.T), axis=1)


def to_corner_form(boxes):
    center, size = boxes[:, :2], boxes[:, 2:4]
    return np.concatenate([center - size / 2.0, center + size / 2.0], axis=1)


def compute_ious(boxes_A, boxes_B):
    xy_min = np.maximum(boxes_A[:, None, 0:2], boxes_B[:, 0:2])
    xy_max = np.minimum(boxes_A[:, None, 2:4], boxes_B[:, 2:4])
    intersection = np.maximum(0.0, xy_max - xy_min)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    areas_A = (boxes_A[:, 2] - boxes_A[:, 0]) * (boxes_A[:, 3] - boxes_A[:, 1])
    areas_B = (boxes_B[:, 2] - boxes_B[:, 0]) * (boxes_B[:, 3] - boxes_B[:, 1])
    union_area = (areas_A[:, None] + areas_B) - intersection_area
    union_area = np.maximum(union_area, 1e-8)
    return np.clip(intersection_area / union_area, 0.0, 1.0)
