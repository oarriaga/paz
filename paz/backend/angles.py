import numpy as np
from paz.datasets import MANOHandJoints
from paz.datasets import MPIIHandJoints
from paz.backend.groups import keypoints_quaternions_to_rotations
from paz.backend.groups import construct_keypoints_transform
from paz.backend.groups import to_affine_matrix
from paz.backend.groups import rotation_matrix_to_compact_axis_angle
from paz.backend.groups import calculate_rotation_matrix_inverse
from paz.backend.keypoints import get_reference_keypoints
from paz.backend.keypoints import rotate_keypoints_with_rotation_matrix
from paz.backend.standard import map_joint_config


def compute_relative_angle(absolute_angles, right_hand=False):
    """Compute the realtive joint rotation for the minimal hand joints and map
       it in the MPII kinematic chain form.

    # Arguments
        absolute_angles : Numpy array [num_joints, 4].
        Absolute joint angle rotation for the minimal hand joints in
        quaternion representation [q1, q2, q3, w0].

    # Returns
        relative_angles: Numpy array [num_joints, 3].
        Relative joint rotation of the minimal hand joints in compact
        axis angle representation.
    """
    absolute_angles = map_joint_config(
        absolute_angles, MPIIHandJoints, MANOHandJoints)
    ref_keypoints = get_reference_keypoints(MANOHandJoints, right_hand)
    absolute_rotation = keypoints_quaternions_to_rotations(absolute_angles)
    rotated_ref_keypoints = rotate_keypoints_with_rotation_matrix(
        absolute_rotation, ref_keypoints)
    rotated_ref_keypoints_transform = construct_keypoints_transform(
        absolute_rotation, rotated_ref_keypoints)
    relative_angles = calculate_relative_angle(
        absolute_rotation, rotated_ref_keypoints_transform)

    relative_angles = map_joint_config(
        relative_angles, MANOHandJoints, MPIIHandJoints)
    joint_angles = np.zeros(shape=(len(absolute_rotation), 3))
    joint_angles[0] = rotation_matrix_to_compact_axis_angle(
        absolute_rotation[0])
    childs = MPIIHandJoints.childs
    joint_angles[1:len(childs), :] = relative_angles[childs[1:], :]
    # joint_angles[childs[1:], :] = relative_angles[childs[1:], :]
    return joint_angles


def calculate_relative_angle(absolute_rotation, ref_keypoint_transform):
    """Calculate the realtive joint rotation for the minimal hand joints.

    # Arguments
        absolute_angles : Numpy array [num_joints, 4].
        Absolute joint angle rotation for the minimal hand joints in
        Euler representation.

    # Returns
        relative_angles: Numpy array [num_joints, 3].
        Relative joint rotation of the minimal hand joints in compact
        axis angle representation.
    """
    relative_angles = np.zeros(shape=(len(absolute_rotation), 3))
    for absolute_arg in range(len(absolute_rotation)):
        transform = to_affine_matrix(
            absolute_rotation[absolute_arg], np.array([0, 0, 0]))
        inverted_transform = np.linalg.inv(transform)
        parent_arg = MANOHandJoints.parents[absolute_arg]
        if parent_arg is not None:
            child_to_parent_transform = np.dot(
                inverted_transform, ref_keypoint_transform[parent_arg])[:3, :3]
            parent_to_child_rotation = calculate_rotation_matrix_inverse(
                child_to_parent_transform)
            parent_to_child_rotation = rotation_matrix_to_compact_axis_angle(
                parent_to_child_rotation)
            relative_angles[absolute_arg] = parent_to_child_rotation
    return relative_angles
