"""Kimodo G1 motion to MuJoCo ``qpos`` rows: z up, x forward, 29 hinges.

Mirrors the reference ``MujocoQposConverter.to_qpos`` with its default
raw joint angles. Body rotations baked into the Unitree XML (hip roll,
knee, shoulder pitch and roll) become per-joint rotation offsets, and
each hinge angle is the Euler projection of the offset local rotation
onto the hinge axis.
"""
import jax.numpy as jp

from paz import quaternion
from paz.models.foundation.kimodo import skeleton

HINGE_JOINTS = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch",
    "right_wrist_yaw",
)
HINGE_AXES = ("y", "x", "z", "y", "y", "x") * 2 + ("z", "x", "y") + (
    "y", "x", "z", "y", "x", "y", "z") * 2
BODY_QUATERNIONS = {
    "left_hip_roll": (0.996179, 0.0, -0.0873386, 0.0),
    "left_knee": (0.996179, 0.0, 0.0873386, 0.0),
    "right_hip_roll": (0.996179, 0.0, -0.0873386, 0.0),
    "right_knee": (0.996179, 0.0, 0.0873386, 0.0),
    "left_shoulder_pitch": (0.990264, 0.139201, 1.38722e-05, -9.86868e-05),
    "left_shoulder_roll": (0.990268, -0.139172, 0.0, 0.0),
    "right_shoulder_pitch": (0.990264, -0.139201, 1.38722e-05, 9.86868e-05),
    "right_shoulder_roll": (0.990268, 0.139172, 0.0, 0.0),
}
# Kimodo (x, y, z) = MuJoCo (y, z, x).
MUJOCO_FROM_KIMODO = jp.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
KIMODO_FROM_MUJOCO = MUJOCO_FROM_KIMODO.T
HINGE_INDICES = tuple(
    skeleton.JOINT_NAMES.index(name) for name in HINGE_JOINTS)


def build_rotation_offsets():
    """Per-joint offsets taking Kimodo local rotations into the XML frames."""
    offsets = jp.tile(jp.eye(3), (skeleton.NUM_JOINTS, 1, 1))
    for name, body_quaternion in BODY_QUATERNIONS.items():
        rotation = quaternion.to_matrix(jp.array(body_quaternion))
        offset = KIMODO_FROM_MUJOCO @ rotation.T @ MUJOCO_FROM_KIMODO
        offsets = offsets.at[skeleton.JOINT_NAMES.index(name)].set(offset)
    return offsets


def build_hinge_axes(offsets):
    axes = []
    for hinge, axis_name in enumerate(HINGE_AXES):
        axis = KIMODO_FROM_MUJOCO @ jp.eye(3)["xyz".index(axis_name)]
        axis = offsets[HINGE_INDICES[hinge]] @ axis
        axes.append(axis / jp.linalg.norm(axis))
    return jp.stack(axes)


ROTATION_OFFSETS = build_rotation_offsets()
HINGE_AXIS_VECTORS = build_hinge_axes(ROTATION_OFFSETS)


def to_qpos(local_rotations, root_position):
    """Converts one frame of ``(34, 3, 3)`` local rotations to a qpos row."""
    rotations = jp.einsum("jmn,jno->jmo", ROTATION_OFFSETS, local_rotations)
    position = MUJOCO_FROM_KIMODO @ root_position
    root = MUJOCO_FROM_KIMODO @ rotations[0] @ KIMODO_FROM_MUJOCO
    angles = extract_euler_angles(rotations[jp.array(HINGE_INDICES)])
    hinge_angles = jp.sum(angles * HINGE_AXIS_VECTORS, axis=-1)
    root_quaternion = quaternion.from_matrix(root)
    return jp.concatenate([position, root_quaternion, hinge_angles])


def extract_euler_angles(rotations):
    x = jp.arctan2(rotations[..., 2, 1], rotations[..., 2, 2])
    y = jp.arctan2(rotations[..., 0, 2], rotations[..., 0, 0])
    z = jp.arctan2(rotations[..., 1, 0], rotations[..., 1, 1])
    return jp.stack([x, y, z], axis=-1)
