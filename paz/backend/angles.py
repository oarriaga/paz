import numpy as np
from paz.datasets import MANOHandJoints
from paz.datasets import MPIIHandJoints
from paz.backend.groups import keypoints_quaternions_to_rotations
from paz.backend.groups import construct_keypoints_transform
from paz.backend.groups import to_affine_matrix
from paz.backend.groups import rotation_matrix_to_compact_axis_angle
from paz.backend.groups import invert_rotation_matrix
from paz.backend.keypoints import rotate_points3D
from paz.backend.standard import map_joint_config


def compute_relative_angles(absolute_angles, links_origin, right_hand=False,
                            angles_config=MANOHandJoints,
                            output_config=MPIIHandJoints):
    """Compute the realtive joint rotation for the minimal hand joints and map
       it to the output_config kinematic chain form.

    # Arguments
        absolute_angles : Array [num_joints, 4].
        Absolute joint angle rotation for the minimal hand joints in
        quaternion representation [q1, q2, q3, w0].

    # Returns
        relative_angles: Array [num_joints, 3].
        Relative joint rotation of the minimal hand joints in compact
        axis angle representation.
    """
    absolute_rotation = keypoints_quaternions_to_rotations(absolute_angles)
    rotated_links_origin = rotate_points3D(
        absolute_rotation, links_origin)
    rotated_links_origin_transform = construct_keypoints_transform(
        absolute_rotation, rotated_links_origin)
    relative_angles = calculate_relative_angle(
        absolute_rotation, rotated_links_origin_transform, angles_config)
    relative_angles = map_relative_angles(relative_angles)
    relative_angles[0] = rotation_matrix_to_compact_axis_angle(
        absolute_rotation[0])
    return relative_angles


def calculate_relative_angle(absolute_rotation, links_origin_transform,
                             config=MANOHandJoints):
    """Calculate the realtive joint rotation for the minimal hand joints.

    # Arguments
        absolute_angles : Array [num_joints, 4].
        Absolute joint angle rotation for the minimal hand joints in
        Euler representation.

    # Returns
        relative_angles: Array [num_joints, 3].
        Relative joint rotation of the minimal hand joints in compact
        axis angle representation.
    """
    relative_angles = np.zeros((len(absolute_rotation), 3))
    for angle_arg in range(len(absolute_rotation)):
        rotation = absolute_rotation[angle_arg]
        transform = to_affine_matrix(rotation, np.array([0, 0, 0]))
        inverted_transform = np.linalg.inv(transform)
        parent_arg = config.parents[angle_arg]
        if parent_arg is not None:
            link_transform = links_origin_transform[parent_arg]
            child_to_parent_transform = np.dot(inverted_transform,
                                               link_transform)
            chils_to_parent_rotation = child_to_parent_transform[:3, :3]
            parent_to_child_rotation = invert_rotation_matrix(
                chils_to_parent_rotation)
            parent_to_child_rotation = rotation_matrix_to_compact_axis_angle(
                parent_to_child_rotation)
            relative_angles[angle_arg] = parent_to_child_rotation
    return relative_angles


def map_relative_angles(relative_angles, angles_config=MANOHandJoints,
                        output_config=MPIIHandJoints):
    """Map data from joint_config1 to joint_config2.

    # Arguments
        relative_angles: Array
        angles_config: joint configuration of the links origin
        output_config: Output joint configuration

    # Returns
        Array: Mapped angles
    """
    angles = np.zeros(shape=(len(relative_angles), 3))
    children = angles_config.children
    if output_config is not MANOHandJoints:
        relative_angles = map_joint_config(
            relative_angles, angles_config, output_config)
        children = output_config.children
    angles[1:len(children), :] = relative_angles[children[1:], :]
    # angles[children[1:], :] = relative_angles[children[1:], :]
    return angles
