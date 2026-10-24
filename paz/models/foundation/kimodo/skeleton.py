"""Unitree G1 skeleton as Kimodo models it: 34 joints, y up, z forward.

Joint order, parents and the neutral (T-pose) joint positions follow the
upstream ``g1skel34`` asset. Rotations are 3x3 matrices; functions act on
one frame and are batched with ``jax.vmap``.
"""
import jax.numpy as jp

JOINT_NAMES = (
    "pelvis",
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll", "left_toe_base",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll", "right_toe_base",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "left_hand_roll",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch",
    "right_wrist_yaw", "right_hand_roll",
)
PARENTS = (
    -1, 0, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 0, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 17, 26, 27, 28, 29, 30, 31, 32,
)
NUM_JOINTS = len(JOINT_NAMES)
RIGHT_HIP, LEFT_HIP = 8, 1
NEUTRAL_JOINTS = jp.array([
    [0.0, 0.0, 0.0],
    [0.064452, -0.1027, 0.0],
    [0.116452, -0.133165, 0.0],
    [0.116452, -0.257285, 0.025001],
    [0.1186009, -0.434625, -0.053272],
    [0.1185065, -0.734635, -0.053272],
    [0.1185065, -0.752193, -0.053272],
    [0.1185065, -0.787193, 0.086728],
    [-0.064452, -0.1027, 0.0],
    [-0.116452, -0.133165, 0.0],
    [-0.116452, -0.257285, 0.025001],
    [-0.1186009, -0.434625, -0.053272],
    [-0.1185065, -0.734635, -0.053272],
    [-0.1185065, -0.752193, -0.053272],
    [-0.1185065, -0.787193, 0.086728],
    [0.0, 0.0, 0.0],
    [0.0, 0.044, -0.0039635],
    [0.0, 0.044, -0.0039635],
    [0.10022, 0.29178, -7.2e-06],
    [0.13822, 0.277949, -7.2e-06],
    [0.14446, 0.174749, -7.2e-06],
    [0.14446, 0.094231, 0.0157758],
    [0.1463479, 0.084231, 0.1157758],
    [0.1463479, 0.084231, 0.1537758],
    [0.1463479, 0.084231, 0.1997758],
    [0.1463479, 0.084231, 0.2997758],
    [-0.10021, 0.29178, -7.2e-06],
    [-0.13821, 0.277949, -7.2e-06],
    [-0.14445, 0.174749, -7.2e-06],
    [-0.14445, 0.094231, 0.0157758],
    [-0.1463379, 0.084231, 0.1157758],
    [-0.1463379, 0.084231, 0.1537758],
    [-0.1463379, 0.084231, 0.1997758],
    [-0.1463379, 0.084231, 0.2997758],
])


def compute_joint_positions(global_rotations, root_position):
    """Places every joint from its parent's global rotation and bone offset."""
    positions = [root_position]
    for joint in range(1, NUM_JOINTS):
        parent = PARENTS[joint]
        offset = NEUTRAL_JOINTS[joint] - NEUTRAL_JOINTS[parent]
        positions.append(positions[parent] + global_rotations[parent] @ offset)
    return jp.stack(positions)


def to_local_rotations(global_rotations):
    parents = global_rotations[jp.array(PARENTS)].at[0].set(jp.eye(3))
    return jp.einsum("jnm,jno->jmo", parents, global_rotations)


def to_global_rotations(local_rotations):
    rotations = [local_rotations[0]]
    for joint in range(1, NUM_JOINTS):
        parent_rotation = rotations[PARENTS[joint]]
        rotations.append(parent_rotation @ local_rotations[joint])
    return jp.stack(rotations)


def compute_heading_angle(joint_positions):
    """Yaw of the hip line; zero when the body faces +z."""
    hips = joint_positions[RIGHT_HIP] - joint_positions[LEFT_HIP]
    return jp.arctan2(hips[2], -hips[0])
