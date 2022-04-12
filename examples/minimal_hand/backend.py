
import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import json
from hand_mesh import HandMesh





def get_scaling_factor(image, size, scaling_factor):
    if isinstance(size, int):
        size = (size, size)
    H, W = image.shape[:2]
    H_scale = H / size[0]
    W_scale = W / size[1]
    return np.array([W_scale * scaling_factor, H_scale * scaling_factor])


def load_json(path):
    """
    Load pickle data.
    Parameter
    ---------
    path: Path to pickle file.
    Return
    ------
    Data in pickle file.
    """
    with open(path) as f:
        data = json.load(f)
        x = data
    return data


class MANOHandJoints:
    n_joints = 21

    labels = [
        'W', #0
        'I0', 'I1', 'I2', #3
        'M0', 'M1', 'M2', #6
        'L0', 'L1', 'L2', #9
        'R0', 'R1', 'R2', #12
        'T0', 'T1', 'T2', #15
        'I3', 'M3', 'L3', 'R3', 'T3' #20, tips are manually added (not in MANO)
    ]

    # finger tips are not joints in MANO, we label them on the mesh manually
    mesh_mapping = {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}

    parents = [
        None,
        0, 1, 2,
        0, 4, 5,
        0, 7, 8,
        0, 10, 11,
        0, 13, 14,
        3, 6, 9, 12, 15
    ]


class MPIIHandJoints:
    n_joints = 21

    labels = [
        'W', #0
        'T0', 'T1', 'T2', 'T3', #4
        'I0', 'I1', 'I2', 'I3', #8
        'M0', 'M1', 'M2', 'M3', #12
        'R0', 'R1', 'R2', 'R3', #16
        'L0', 'L1', 'L2', 'L3', #20
    ]

    parents = [
        None,
        0, 1, 2, 3,
        0, 5, 6, 7,
        0, 9, 10, 11,
        0, 13, 14, 15,
        0, 17, 18, 19
    ]



def mano_to_mpii(mano):
    """
    Map data from MANOHandJoints order to MPIIHandJoints order.

    Parameters
    ----------
    mano : np.ndarray, [21, ...]
        Data in MANOHandJoints order. Note that the joints are along axis 0.

    Returns
    -------
    np.ndarray
        Data in MPIIHandJoints order.
    """
    mpii = []
    for j in range(MPIIHandJoints.n_joints):
        mpii.append(
        mano[MANOHandJoints.labels.index(MPIIHandJoints.labels[j])]
        )
    mpii = np.stack(mpii, 0)
    return mpii


def mpii_to_mano(mpii):
  """
  Map data from MPIIHandJoints order to MANOHandJoints order.

  Parameters
  ----------
  mpii : np.ndarray, [21, ...]
    Data in MPIIHandJoints order. Note that the joints are along axis 0.

  Returns
  -------
  np.ndarray
    Data in MANOHandJoints order.
  """
  mano = []
  for j in range(MANOHandJoints.n_joints):
    mano.append(
      mpii[MPIIHandJoints.labels.index(MANOHandJoints.labels[j])]
    )
  mano = np.stack(mano, 0)
  return mano


def xyz_to_delta(xyz, joints_def):
    """
    Compute bone orientations from joint coordinates (child joint - parent joint).
    The returned vectors are normalized.
    For the root joint, it will be a zero vector.

    Parameters
    ----------
    xyz : np.ndarray, shape [J, 3]
        Joint coordinates.
    joints_def : object
        An object that defines the kinematic skeleton, e.g. MPIIHandJoints.

    Returns
    -------
    np.ndarray, shape [J, 3]
        The **unit** vectors from each child joint to its parent joint.
        For the root joint, it's are zero vector.
    np.ndarray, shape [J, 1]
        The length of each bone (from child joint to parent joint).
        For the root joint, it's zero.
    """
    delta = []
    for j in range(joints_def.n_joints):
        p = joints_def.parents[j]
        if p is None:
            delta.append(np.zeros(3))
        else:
            delta.append(xyz[j] - xyz[p])
    delta = np.stack(delta, 0)
    lengths = np.linalg.norm(delta, axis=-1, keepdims=True)
    delta /= np.maximum(lengths, np.finfo(xyz.dtype).eps)
    return delta, lengths


def calculate_handstate_joint_angles_from_min_hand_absolute_angles(quats):

    # rotate reference joints and get posed hand sceleton J
    J = rotated_ref_joints_from_quats(quats)

    # combine each joint with absolute rotation to transformation: t_posed_super_rotated
    t_posed_super_rotated = np.zeros(shape=(21, 4, 4))
    for i in range(21):
        t_posed_super_rotated[i] = pt.transform_from(
            pr.matrix_from_quaternion(quats[i]),
            J[i]
        )

    t_relative = np.zeros(shape=(21, 3, 3))

    # For each quaternion Q:
    for i in range(len(quats)):

        # Calc transformation with inverted rotation of Qi
        T_abs_rotations_i_inverted = pt.invert_transform(
            pt.transform_from(
                pr.matrix_from_quaternion(quats[i]),
                [0, 0, 0]  # translation does not matter
            )
        )

        # Update Q_orientation if joint i has a parent (substract parents orientation)
        parent_index = MANOHandJoints.parents[i]
        if parent_index is not None:
            # Concatenate transformation get rotation difference (child to parent):
            # posed and super rotated joint i
            # inverted rotation of Qi
            t_posed_rotation_child_to_parent_i = pt.concat(
                t_posed_super_rotated[parent_index],
                T_abs_rotations_i_inverted
            )

            # clear out translationand keep only rotation
            t = pt.pq_from_transform(t_posed_rotation_child_to_parent_i)
            t_rotation_child_to_parent_i = np.array([t[3], t[4], t[5], t[6]])

            t_relative[i] = pr.matrix_from_quaternion(
                pr.q_conj(t_rotation_child_to_parent_i)
            )

    # Generate final array with 16 joint angles
    joint_angles = np.zeros(shape=(21, 3))

    # Root joint gets same orientation like absolute root quaternion
    joint_angles[0] = pr.compact_axis_angle_from_matrix(
        pr.matrix_from_quaternion(quats[0])
    )

    # Map of childs array_index = joint_index => parent_joint_index
    childs = [
        [1, 4, 7, 10, 13],  # root_joint has multiple childs
        2, 3, 16, 5, 6, 17, 8, 9, 18, 11, 12, 19, 14, 15, 20  # other joints have exactly one parent
    ]
    # Joint 1-15 gets calculated orientation of child's join
    for i in range(1, 16):
        joint_angles[i] = pr.compact_axis_angle_from_matrix(
            t_relative[childs[i]]
        )

    return joint_angles


def rotated_ref_joints_from_quats(quat):
    hand_mesh = HandMesh(left=True)
    rotation_matrices = np.zeros(shape=(21, 3, 3))
    for j in range(len(quat)):
        rotation_matrices[j] = pr.matrix_from_quaternion(quat[j])
    mats = np.stack(rotation_matrices, 0)
    joint_xyz = np.matmul(mats, hand_mesh.ref_pose)[..., 0]
    return joint_xyz