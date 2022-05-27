import numpy as np
from paz.datasets import MANOHandJoints
from paz.datasets import MANO_REF_JOINTS
from paz.datasets import MPIIHandJoints
from paz.backend.groups import quaternion_to_rotation_matrix
from paz.backend.groups import to_affine_matrix


def get_scaling_factor(image, scale=1, shape=(128, 128)):
    '''
    Return scaling factor for the image.

    # Arguments
        image: Numpy array.
        scale: Int.
        shape: Tuple of integers. eg. (128, 128)

    # Returns
        scaling factor: Numpy array of size 2
    '''
    H, W = image.shape[:2]
    H_scale = H / shape[0]
    W_scale = W / shape[1]
    return np.array([W_scale * scale, H_scale * scale])


def map_joint_config(joints, joint_config1, joint_config2):
    """Map data from joint_config1 to joint_config2.

    # Arguments
        joints: Numpy array
        joint_config1: joint configuration of the joints
        joint_config2: joint configuration the joints to be converted

    # Returns
        Numpy array: joints maped to the joint_config2
    """
    mapped_joints = []

    for joint_arg in range(joint_config2.num_joints):
        joint_label = joint_config2.labels[joint_arg]
        joint_index_in_joint_config1 = joint_config1.labels.index(joint_label)
        joint_in_joint_config1 = joints[joint_index_in_joint_config1]
        mapped_joints.append(joint_in_joint_config1)
    mapped_joints = np.stack(mapped_joints, 0)
    return mapped_joints


def keypoints3D_to_delta(keypoints3D, joints_config):
    """Compute bone orientations from joint coordinates
       (child joint - parent joint). The returned vectors are normalized.
       For the root joint, it will be a zero vector.

    # Arguments
        keypoints3D : Numpy array [num_joints, 3]. Joint coordinates.
        joints_config : joint configuration of the joints. e.g. MPIIHandJoints.

    # Returns
        Numpy array [num_joints, 3]. The unit vectors from each child joint to
        its parent joint. For the root joint, it's are zero vector.
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
    ref_keypoints = get_reference_keypoints(MANO_REF_JOINTS, right_hand)
    absolute_rotation = keypoints_quaternions_to_rotations(absolute_angles)
    rotated_ref_keypoints = rotate_keypoints(absolute_rotation, ref_keypoints)
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


def calculate_rotation_matrix_inverse(matrix):
    """Calculate the inverse of ratation matrix using quaternions.

    # Arguments
        Rotation matrix [3, 3]

    # Returns
        Rotation matrix inverse [3, 3]
    """
    quaternion = rotation_matrix_to_quaternion(matrix)
    quaternion_conjugate = get_quaternion_conjugate(quaternion)
    inverse_matrix = quaternion_to_rotation_matrix(quaternion_conjugate)
    return inverse_matrix


def construct_keypoints_transform(rotations, translations):
    """Construct vectorised transformation matrix from ratation matrix vector
    and translation vector.

    # Arguments
        ratations: Rotation matrix vector [N, 3, 3].
        translations: Translation vector [N, 3, 1].

    # Returns
        Transformation matrix [N, 4, 4]
    """
    keypoints_transform = np.zeros(shape=(len(rotations), 4, 4))
    for keypoint_arg in range(len(rotations)):
        keypoints_transform[keypoint_arg] = to_affine_matrix(
            rotations[keypoint_arg], translations[keypoint_arg])
    return keypoints_transform


def keypoints_quaternions_to_rotations(quaternions):
    """Transform quaternion vectors to rotation matrix vector.

    # Arguments
        quaternions [N, 4].

    # Returns
        Rotated matrices [N, 3, 3]
    """
    keypoints_rotations = np.zeros(shape=(len(quaternions), 3, 3))
    for keypoint_arg in range(len(quaternions)):
        rotation_matrix = quaternion_to_rotation_matrix(
            quaternions[keypoint_arg])
        keypoints_rotations[keypoint_arg] = rotation_matrix
    return keypoints_rotations


def rotate_keypoints(rotation_matrix, keypoints):
    """Rotatate the keypoints

    # Arguments
        Rotation matrix [N, 3, 3].
        keypoints [N, 3, 1]

    # Returns
        Rotated keypoints [N, 3, 1]
    """
    keypoint_xyz = np.einsum('ijk, ikl -> ij', rotation_matrix, keypoints)
    return keypoint_xyz


def get_reference_keypoints(keypoints=MANO_REF_JOINTS, right_hand=False):
    if right_hand:
        keypoints = transform_column_to_negative(keypoints)
    ref_pose = keypoints3D_to_delta(keypoints, MANOHandJoints)
    ref_pose = np.expand_dims(ref_pose, -1)
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


def transform_column_to_negative(array, column=0):
    """Transforms a column of an array to negative value.

    # Arguments
        array: Numpy array
        column: int/list

    # Returns
        array: Numpy array
    """
    array[:, column] = -array[:, column]
    return array


def flip_keypoints_wrt_image(keypoints, image_size=(32, 32), axis=1):
    """Flio the detected keypoints with respect to image

    # Arguments
        keypoints: Numpy array 
        image_size: list/tuple
        axis: int

    # Returns
        flipped_keypoints: Numpy array
    """
    keypoints[:, axis] = image_size[axis] - keypoints[:, axis]
    return keypoints
