import numpy as np
from paz.datasets import MANOHandJoints
from paz.datasets import MANO_REF_JOINTS
from paz.datasets import MPIIHandJoints
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


def calculate_relative_angle(absolute_angles):
    ref_joints = hand_mesh(MANO_REF_JOINTS)
    absolute_rotation = joints_quaternions_to_rotations(absolute_angles)
    rotated_ref_joints = rotate_keypoints(absolute_rotation, ref_joints)
    rotated_ref_joints_transform = construct_joints_transform(
        absolute_rotation, rotated_ref_joints)
    relative_angles = get_relative_angle(
        absolute_rotation, rotated_ref_joints_transform)

    joint_angles = np.zeros(shape=(len(absolute_rotation), 3))
    joint_angles[0] = rotation_matrix_to_compact_axis_angle(absolute_rotation[0])
    childs = MANOHandJoints.childs
    joint_angles[1:len(childs), :] = relative_angles[childs[1:], :]
    return joint_angles


def get_relative_angle(absolute_rotation, ref_joint_transform, num_joints=21):
    relative_angles = np.zeros(shape=(num_joints, 3))
    for absolute_arg in range(len(absolute_rotation)):
        transform = to_affine_matrix(
            absolute_rotation[absolute_arg], np.array([0, 0, 0]))
        inverted_transform = np.linalg.inv(transform)
        parent_arg = MANOHandJoints.parents[absolute_arg]
        if parent_arg is not None:
            child_to_parent_transform = np.dot(
                inverted_transform, ref_joint_transform[parent_arg])[:3, :3]
            parent_to_child_rotation = calculate_matrix_inverse(
                child_to_parent_transform)
            parent_to_child_rotation = rotation_matrix_to_compact_axis_angle(
                parent_to_child_rotation)
            relative_angles[absolute_arg] = parent_to_child_rotation
    return relative_angles


def calculate_matrix_inverse(matrix):
    quaternion = rotation_matrix_to_quaternion(matrix)
    quaternion_conjugate = get_quaternion_conjugate(quaternion)
    inverse_matrix = quaternion_to_rotation_matrix(quaternion_conjugate)
    return inverse_matrix


def construct_joints_transform(rotations, translations):
    joints_transform = np.zeros(shape=(len(rotations), 4, 4))
    for joint_arg in range(len(rotations)):
        joints_transform[joint_arg] = to_affine_matrix(
            rotations[joint_arg], translations[joint_arg])
    return joints_transform


def joints_quaternions_to_rotations(quaternions):
    joints_rotations = np.zeros(shape=(len(quaternions), 3, 3))
    for joint_arg in range(len(quaternions)):
        rotation_matrix = quaternion_to_rotation_matrix(quaternions[joint_arg])
        joints_rotations[joint_arg] = rotation_matrix
    return joints_rotations


def rotate_keypoints(rotation_matrix, keypoints):
    joint_xyz = np.matmul(rotation_matrix, keypoints)[..., 0]
    return joint_xyz
# ***********************************************************************


def hand_mesh(joint_config=MANO_REF_JOINTS, left=True):
    if left:
        joints = joint_config
    else:
        joints = transform_column_to_negative(joints)

    ref_pose = []
    for j in range(MANOHandJoints.num_joints):
        parent = MANOHandJoints.parents[j]
        if parent is None:
            ref_pose.append(joints[j])
        else:
            ref_pose.append(joints[j] - joints[parent])

    # make a config file just for that
    ref_pose = np.expand_dims(np.stack(ref_pose, 0), -1)
    return ref_pose


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


def get_quaternion_conjugate(quaternion):
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
