import cv2
import numpy as np

from paz.datasets import human36m


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


def project_to_image(rotation, translation, points3D, camera_intrinsics):
    points3D = np.matmul(rotation, points3D.T).T + translation
    x, y, z = np.split(points3D, 3, axis=1)
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    return np.concatenate([fx * (x / z) + cx, fy * (y / z) + cy], axis=1)


def filter_keypoints3D(keypoints3D, args_to_joints3D):
    keypoints3D = np.reshape(keypoints3D, [len(keypoints3D), 32, 3])
    return keypoints3D[:, args_to_joints3D, :]


def filter_keypoints2D_to_h36m(keypoints2D):
    keypoints2D = np.array(keypoints2D, dtype="float32")
    for joint, (first, second) in human36m.args_to_mean.items():
        keypoints2D[:, joint] = (keypoints2D[:, first] + keypoints2D[:, second]) / 2  # fmt: skip
    selected = keypoints2D[:, human36m.h36m_to_coco_joints2D, :]
    return selected.reshape(selected.shape[0], -1)


def get_bones_length(poses2D, poses3D, start_joints,
                     end_joints=np.arange(1, 16)):
    poses3D = np.reshape(poses3D, (poses3D.shape[0], 16, -1))
    poses2D = np.reshape(poses2D, (poses2D.shape[0], 16, -1))
    length2D = sum_bone_lengths(poses2D, start_joints, end_joints)
    length3D = sum_bone_lengths(poses3D, start_joints, end_joints)
    return length2D, length3D


def sum_bone_lengths(poses, start_joints, end_joints):
    lengths = []
    for person in poses:
        bones = person[start_joints] - person[end_joints]
        lengths.append(np.linalg.norm(bones, axis=-1).sum())
    return np.array(lengths)


def initialize_translation(root2D, camera_intrinsics, ratio):
    focal_length = camera_intrinsics[0, 0]
    center_x, center_y = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    z = focal_length * ratio
    x = (root2D[:, 0] - center_x) * ratio
    y = (root2D[:, 1] - center_y) * ratio
    return np.array((x, y, z)).flatten()


def compute_reprojection_error(translation, keypoints3D, keypoints2D,
                               camera_intrinsics):
    translation = np.reshape(translation, (-1, 3))
    poses3D = keypoints3D + translation[:, np.newaxis, :]
    poses3D = poses3D.reshape((-1, 3))
    projection = project_to_image(np.identity(3), np.zeros(3), poses3D,
                                  camera_intrinsics)
    return np.sum(np.linalg.norm(np.ravel(keypoints2D) - np.ravel(projection)))


def solve_least_squares(solver, error, initial_translation, joints3D,
                        joints2D, camera_intrinsics):
    solution = solver(error, initial_translation, verbose=0,
                      args=(joints3D, joints2D, camera_intrinsics))
    return np.reshape(solution.x, (-1, 3))


def compute_optimized_pose3D(keypoints3D, translation, camera_intrinsics):
    optimized_pose3D, projection2D = [], []
    for person in range(keypoints3D.shape[0]):
        pose = (keypoints3D[person] + translation[person]).reshape((-1, 3))
        points = project_to_image(np.identity(3), np.zeros(3), pose,
                                  camera_intrinsics)
        optimized_pose3D.append(pose)
        projection2D.append(np.reshape(points, [1, -1]))
    return np.array(optimized_pose3D), np.array(projection2D)


def optimize_human_pose3D(keypoints3D, keypoints2D, solver, camera_intrinsics):
    joints3D = filter_keypoints3D(keypoints3D, human36m.args_to_joints3D)
    joints2D = filter_keypoints2D_to_h36m(keypoints2D)
    length2D, length3D = get_bones_length(joints2D, keypoints3D,
                                          human36m.human_start_joints)
    ratio = length3D / length2D
    initial = initialize_translation(joints2D[:, :2], camera_intrinsics, ratio)
    translation = solve_least_squares(solver, compute_reprojection_error,
                                      initial, joints3D, joints2D,
                                      camera_intrinsics)
    return compute_optimized_pose3D(keypoints3D, translation, camera_intrinsics)


def human_pose3D_to_pose6D(poses3D):
    right_hip, left_hip, thorax = poses3D[1], poses3D[6], poses3D[13]
    x_vector = right_hip - left_hip
    projection = thorax - left_hip
    scalar = np.dot(x_vector, projection) / np.linalg.norm(x_vector) ** 2
    z_vector = thorax - (left_hip + scalar * x_vector)
    x_unit = x_vector / np.linalg.norm(x_vector)
    z_unit = z_vector / np.linalg.norm(z_vector)
    y_unit = np.cross(z_unit, x_unit)
    rotation = np.column_stack((x_unit, y_unit, z_unit))
    return rotation, (poses3D[0] / 1e3).tolist()
