import numpy as np
from paz.datasets import MANOHandJoints
from paz.datasets import MANO_REF_JOINTS
from paz.backend.groups import quaternion_to_rotation_matrix
# from paz.backend.groups import matrix_from_quaternion


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


def calculate_relative_angle(quats):

    # rotate reference joints and get posed hand sceleton J
    J = rotated_ref_joints_from_quats(quats)

    # combine each joint with absolute rotation to transformation: t_posed_super_rotated
    t_posed_super_rotated = np.zeros(shape=(21, 4, 4))
    for i in range(21):
        t_posed_super_rotated[i] = to_affine_matrix(
            matrix_from_quaternion(quats[i]),
            J[i]
        )

    t_relative = np.zeros(shape=(21, 3, 3))

    # For each quaternion Q:
    for i in range(len(quats)):

        # Calc transformation with inverted rotation of Qi
        T_abs_rotations_i_inverted = invert_affine_matrix(
            to_affine_matrix(
                matrix_from_quaternion(quats[i]),
                np.array([0, 0, 0])  # translation does not matter
            )
        )


        # Update Q_orientation if joint i has a parent (substract parents orientation)
        parent_index = MANOHandJoints.parents[i]
        if parent_index is not None:
            # Concatenate transformation get rotation difference (child to parent):
            # posed and super rotated joint i
            # inverted rotation of Qi
            t_posed_rotation_child_to_parent_i = concatenate_transformations(
                t_posed_super_rotated[parent_index],
                T_abs_rotations_i_inverted
            )

            # clear out translationand keep only rotation
            t_rotation_child_to_parent_i = rotation_matrix_to_quaternion(t_posed_rotation_child_to_parent_i)
            t_relative[i] = matrix_from_quaternion(
                quaternion_conjugate(t_rotation_child_to_parent_i)
            )

    # Generate final array with 16 joint angles
    joint_angles = np.zeros(shape=(21, 3))

    # Root joint gets same orientation like absolute root quaternion
    joint_angles[0] = compact_axis_angle_from_matrix(
        matrix_from_quaternion(quats[0])
    )

    # Map of childs array_index = joint_index => parent_joint_index
    childs = [
        [1, 4, 7, 10, 13],  # root_joint has multiple childs
        2, 3, 16, 5, 6, 17, 8, 9, 18, 11, 12, 19, 14, 15, 20  # other joints have exactly one parent
    ]
    # Joint 1-15 gets calculated orientation of child's join
    for i in range(1, 16):
        joint_angles[i] = compact_axis_angle_from_matrix(
            t_relative[childs[i]]
        )
    return joint_angles


def rotated_ref_joints_from_quats(quat):
    ref_pose = hand_mesh()
    rotation_matrices = np.zeros(shape=(21, 3, 3))
    for j in range(len(quat)):
        rotation_matrices[j] = matrix_from_quaternion(quat[j])
    mats = np.stack(rotation_matrices, 0)
    joint_xyz = np.matmul(mats, ref_pose)[..., 0]
    return joint_xyz


def hand_mesh(left=True):
    if left:
        joints = MANO_REF_JOINTS
    else:
        # include case for left == False
        pass
    ref_pose = []
    for j in range(MANOHandJoints.num_joints):
        parent = MANOHandJoints.parents[j]
        if parent is None:
            ref_pose.append(joints[j])
        else:
            ref_pose.append(joints[j] - joints[parent])
    ref_pose = np.expand_dims(np.stack(ref_pose, 0), -1)
    return ref_pose


def matrix_from_quaternion(quaternion):
    norm = np.linalg.norm(quaternion)
    quaternion = quaternion / norm
    w0, q1, q2, q3 = quaternion
    rm = quaternion_to_rotation_matrix(np.array([q1, q2, q3, w0]))
    return rm


def rotation_matrix_to_axis_angle(rotation_matrix):
    cos_theta = (np.trace(rotation_matrix) - 1.0) / 2.0
    angle = np.arccos(cos_theta)
    axis = np.array([rotation_matrix[2, 1] - rotation_matrix[1, 2],
                     rotation_matrix[0, 2] - rotation_matrix[2, 0],
                     rotation_matrix[1, 0] - rotation_matrix[0, 1]])
    axis_angle = np.hstack([axis, angle])
    return axis_angle


def compact_axis_angle_from_matrix(matrix):
    axis_angle = rotation_matrix_to_axis_angle(matrix)
    axis = axis_angle[:3]
    angle = axis_angle[3]
    axis = axis / np.linalg.norm(axis)
    compact_axis_angle = axis * angle
    return compact_axis_angle


def quaternion_conjugate(quaternion):
    w0, q1, q2, q3 = quaternion
    return np.array([w0, -q1, -q2, -q3])


def to_affine_matrix(rotation_matrix, translation):
    """Builds affine matrix from rotation matrix and translation vector.

    # Arguments
        rotation_matrix: Array (3, 3). Representing a rotation matrix.
        translation: Array (3). Translation vector.

    # Returns
        Array (4, 4) representing an affine matrix.
    """
    translation = translation.reshape(3, 1)
    affine_top = np.concatenate([rotation_matrix, translation], axis=1)
    affine_row = np.array([[0.0, 0.0, 0.0, 1.0]])
    affine_matrix = np.concatenate([affine_top, affine_row], axis=0)
    return affine_matrix


def invert_affine_matrix(affine_matrix):
    return np.linalg.inv(affine_matrix)


def concatenate_transformations(affine_matrix_a, affine_matrix_b):
    return np.dot(affine_matrix_b, affine_matrix_a)


def rotation_matrix_to_quaternion(rotation_matrix):
    rotation_matrix = rotation_matrix[:3, :3]
    trace = np.trace(rotation_matrix)
    w0 = np.sqrt(1.0 + trace) / 2
    q1 = 0.25 * (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / w0
    q2 = 0.25 * (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / w0
    q3 = 0.25 * (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / w0
    quaternion = np.array([w0, q1, q2, q3])
    return quaternion
