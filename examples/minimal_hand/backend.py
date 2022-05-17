import numpy as np
from paz.datasets import MANOHandJoints
from paz.datasets import MANO_REF_JOINTS
from paz.backend.groups import quaternion_to_rotation_matrix
from paz.backend.groups import to_affine_matrix


def get_scaling_factor(image, size, scaling_factor):
    if isinstance(size, int):
        size = (size, size)
    H, W = image.shape[:2]
    H_scale = H / size[0]
    W_scale = W / size[1]
    return np.array([W_scale * scaling_factor, H_scale * scaling_factor])


def map_joint_config(joints, joint_config1, joint_config2):
    """
    Map data from joint_config1 to joint_config2.

    Parameters
    # mano : np.ndarray, [21, ...]
        Data in joint_config1. Note that the joints are along axis 0.

    Returns
    # np.ndarray
        Data in joint_config2.
    """
    mapped_joints = []

    for joint_arg in range(joint_config1.num_joints):
        joint_label = joint_config1.labels[joint_arg]
        joint_index_in_joint_config2 = joint_config2.labels.index(joint_label)
        joint_in_joint_config2 = joints[joint_index_in_joint_config2]
        mapped_joints.append(joint_in_joint_config2)
    mapped_joints = np.stack(mapped_joints, 0)
    return mapped_joints


def keypoints3D_to_delta(keypoints3D, joints_config):
    """
    Compute bone orientations from joint coordinates
    (child joint - parent joint).
    The returned vectors are normalized.
    For the root joint, it will be a zero vector.

    # Parameters
    keypoints3D : np.ndarray, shape [J, 3]
        Joint coordinates.
    joints_config : object
        An object that defines the kinematic skeleton, e.g. MPIIHandJoints.

    # Returns
    np.ndarray, shape [J, 3]
        The **unit** vectors from each child joint to its parent joint.
        For the root joint, it's are zero vector.
    """
    delta = []
    for joint_arg in range(joints_config.num_joints):
        parent = joints_config.parents[joint_arg]
        if parent is None:
            delta.append(np.zeros(3))
        else:
            delta.append(keypoints3D[joint_arg] - keypoints3D[parent])
    delta = np.stack(delta, 0)
    return delta


def calculate_relative_angle(absolute_angles, num_joints=21):
    rotated_ref_joints = rotate_ref_joints(absolute_angles)

    # combine each joint with absolute rotation to transformation:
    rotated_ref_joint_transform = np.zeros(shape=(num_joints, 4, 4))
    for joint_arg in range(num_joints):
        rotation_matrix = matrix_from_quaternion(absolute_angles[joint_arg])
        rotated_ref_joint_transform[joint_arg] = to_affine_matrix(
            rotation_matrix, rotated_ref_joints[joint_arg])

    relative_angles = get_relative_angle(absolute_angles,
                                         rotated_ref_joint_transform)

    # Generate final array with 16 joint angles
    joint_angles = np.zeros(shape=(num_joints, 3))

    # Root joint gets same orientation like absolute root quaternion
    root_joint_angle = matrix_from_quaternion(absolute_angles[0])
    joint_angles[0] = rotation_matrix_to_compact_axis_angle(root_joint_angle)

    # Joint 1-15 gets calculated orientation of child's join
    for joint_arg in range(1, 16):
        joint_angles[joint_arg] = rotation_matrix_to_compact_axis_angle(
            relative_angles[MANOHandJoints.childs[joint_arg]])
    return joint_angles


def get_relative_angle(absolute_angles, ref_joint_transform, num_joints=21):
    relative_angles = np.zeros(shape=(num_joints, 3, 3))
    for absolute_arg in range(len(absolute_angles)):
        rotation = matrix_from_quaternion(absolute_angles[absolute_arg])
        transform = to_affine_matrix(rotation, np.array([0, 0, 0]))
        inverted_transform = np.linalg.inv(transform)
        parent_arg = MANOHandJoints.parents[absolute_arg]

        if parent_arg is not None:
            child_to_parent_arg_transform = np.dot(
                inverted_transform, ref_joint_transform[parent_arg])

            joint_arg_relative_quaternion = rotation_matrix_to_quaternion(
                child_to_parent_arg_transform)
            relative_angles[absolute_arg] = matrix_from_quaternion(
                quaternion_conjugate(joint_arg_relative_quaternion))
    return relative_angles


def rotate_ref_joints(quat):
    ref_pose = hand_mesh()
    rotation_matrices = np.zeros(shape=(21, 3, 3))
    for j in range(len(quat)):
        rotation_matrices[j] = matrix_from_quaternion(quat[j])
    rotation_matrices = np.stack(rotation_matrices, 0)
    joint_xyz = np.matmul(rotation_matrices, ref_pose)[..., 0]
    return joint_xyz


def hand_mesh(left=True):
    if left:
        joints = MANO_REF_JOINTS
    else:
        joints = transform_column_to_negative(joints)
    ref_pose = []
    for j in range(MANOHandJoints.num_joints):
        parent = MANOHandJoints.parents[j]
        if parent is None:
            ref_pose.append(joints[j])
        else:
            ref_pose.append(joints[j] - joints[parent])
    ref_pose = np.expand_dims(np.stack(ref_pose, 0), -1)
    return ref_pose


def normalize_quaternion(quaternion):
    norm = np.linalg.norm(quaternion)
    normalized_quaternion = quaternion / norm
    return normalized_quaternion


def matrix_from_quaternion(quaternion):
    quaternion = normalize_quaternion(quaternion)
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    return rotation_matrix


def rotation_matrix_to_axis_angle(rotation_matrix):
    """Transforms rotation matrix to axis angle.

    # Arguments
        Rotation matrix [3, 3].

    # Returns
        axis_angle: Array containing axis angle represent [wx, wy, wz, theta].
    """
    cos_theta = (np.trace(rotation_matrix) - 1.0) / 2.0
    angle = np.arccos(cos_theta)
    axis = np.array([rotation_matrix[2, 1] - rotation_matrix[1, 2],
                     rotation_matrix[0, 2] - rotation_matrix[2, 0],
                     rotation_matrix[1, 0] - rotation_matrix[0, 1]])
    axis = axis / np.linalg.norm(axis)
    axis_angle = np.hstack([axis, angle])
    return axis_angle


def rotation_matrix_to_compact_axis_angle(matrix):
    """Transforms rotation matrix to compact axis angle.

    # Arguments
        Rotation matrix [3, 3].

    # Returns
        compact axis_angle
    """
    axis_angle = rotation_matrix_to_axis_angle(matrix)
    axis = axis_angle[:3]
    angle = axis_angle[3]
    compact_axis_angle = axis * angle
    return compact_axis_angle


def quaternion_conjugate(quaternion):
    """Estimate conjugate of a quaternion.

    # Arguments
        quaternion: Array containing quaternion value [q1, q2, q3, w0].

    # Returns
        quaternion: Array containing quaternion value [-q1, -q2, -q3, w0].
    """
    q1, q2, q3, w0 = quaternion
    return np.array([-q1, -q2, -q3, w0])


def rotation_matrix_to_quaternion(rotation_matrix):
    """Transforms rotation matrix to quaternion.

    # Arguments
        Rotation matrix [3, 3].

    # Returns
        quaternion: Array containing quaternion value [q1, q2, q3, w0].
    """
    rotation_matrix = rotation_matrix[:3, :3]
    trace = np.trace(rotation_matrix)
    w0 = np.sqrt(1.0 + trace) / 2
    q1 = 0.25 * (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / w0
    q2 = 0.25 * (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / w0
    q3 = 0.25 * (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / w0
    quaternion = np.array([q1, q2, q3, w0])
    return quaternion


def transform_column_to_negative(self, array, column=0):
    array[:, column] = -array[:, column]
    return
