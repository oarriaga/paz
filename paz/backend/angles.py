import numpy as np

from paz.backend.lie import quaternion
from paz.datasets.hands import MANOHandJoints
from paz.datasets.hands import MPIIHandJoints


def compute_orientation_vector(keypoints3D, parents):
    deltas = []
    for joint_arg, parent in enumerate(parents):
        if parent is None:
            deltas.append(np.zeros(3))
        else:
            deltas.append(keypoints3D[joint_arg] - keypoints3D[parent])
    return np.stack(deltas, axis=0)


def quaternion_to_rotation_matrix(quaternion_xyzw):
    x, y, z, w = quaternion_xyzw
    return np.asarray(quaternion.to_matrix([w, x, y, z]))


def quaternions_to_rotation_matrices(quaternions):
    matrices = [quaternion_to_rotation_matrix(q) for q in quaternions]
    return np.array(matrices)


def rotate_vectors(rotations, vectors):
    return np.einsum("ijk,ik->ij", rotations, vectors)


def to_affine_matrix(rotation, translation):
    translation = np.reshape(translation, (3, 1))
    affine_top = np.concatenate([rotation, translation], axis=1)
    affine_row = np.array([[0.0, 0.0, 0.0, 1.0]])
    return np.concatenate([affine_top, affine_row], axis=0)


def to_affine_matrices(rotations, translations):
    pairs = zip(rotations, translations)
    matrices = [to_affine_matrix(rotation, t) for rotation, t in pairs]
    return np.array(matrices)


def rotation_matrix_to_compact_axis_angle(rotation):
    angle = np.arccos((np.trace(rotation) - 1.0) / 2.0)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    axis = axis / np.linalg.norm(axis)
    return axis * angle


def change_link_order(joints, source_labels, target_labels):
    mapped = [joints[source_labels.index(label)] for label in target_labels]
    return np.stack(mapped, axis=0)


def calculate_relative_angle(absolute_rotations, links_transform, parents):
    relative_angles = np.zeros((len(absolute_rotations), 3))
    for joint_arg, parent in enumerate(parents):
        if parent is None:
            continue
        transform = to_affine_matrix(absolute_rotations[joint_arg], np.zeros(3))
        child_to_parent = np.dot(np.linalg.inv(transform),
                                 links_transform[parent])
        parent_to_child = np.linalg.inv(child_to_parent[:3, :3])
        relative_angles[joint_arg] = rotation_matrix_to_compact_axis_angle(
            parent_to_child)
    return relative_angles


def reorder_relative_angles(relative_angles, root_rotation, children,
                            root_joints=(1, 4, 7, 10, 13)):
    root_angle = rotation_matrix_to_compact_axis_angle(root_rotation)
    children_angles = relative_angles[children[1:], :]
    children_angles = np.concatenate(
        [np.expand_dims(root_angle, 0), children_angles])
    return np.insert(children_angles, root_joints, np.zeros(3), axis=0)


def flip_along_x_axis(keypoints):
    x, y, z = np.split(keypoints, 3, axis=1)
    return np.concatenate([-x, y, z], axis=1)


def compute_relative_angles(absolute_quaternions, right_hand=False):
    mano_links_origin = MANOHandJoints.links_origin
    if right_hand:
        mano_links_origin = flip_along_x_axis(mano_links_origin)
    quaternions = change_link_order(
        absolute_quaternions, MPIIHandJoints.labels, MANOHandJoints.labels)
    rotations = quaternions_to_rotation_matrices(quaternions)
    links_orientation = compute_orientation_vector(
        mano_links_origin, MANOHandJoints.parents)
    rotated_links = rotate_vectors(rotations, links_orientation)
    links_transform = to_affine_matrices(rotations, rotated_links)
    relative_angles = calculate_relative_angle(
        rotations, links_transform, MANOHandJoints.parents)
    relative_angles = change_link_order(
        relative_angles, MANOHandJoints.labels, MPIIHandJoints.labels)
    return reorder_relative_angles(
        relative_angles, rotations[0], MPIIHandJoints.children)
