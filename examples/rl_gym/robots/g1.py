from collections import namedtuple
from pathlib import Path

import jax.numpy as jp
import mujoco
import numpy as np

from robots import build as build_robot
from robots import read_sensor_addresses, reject_keyword_args

ASSET_PATH = Path(__file__).parent.parent / "assets/g1_29dof.xml"
# the knees-bent default pose, the derived actuator gains, and the
# per-group action scale follow mjlab, which trains this robot on this
# simulator; the pose keeps the base at 0.76 m
DEFAULT_ANGLES = jp.array([-0.312, 0.0, 0.0, 0.669, -0.363, 0.0, -0.312, 0.0, 0.0, 0.669, -0.363, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0, 0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0])  # fmt: skip
ACTION_SCALE = jp.array([0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386, 0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386, 0.5475, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745])  # fmt: skip
FOOT_SUFFIX = "ankle_roll_link"


def G1DoF29():
    mjspec = mujoco.MjSpec.from_file(str(ASSET_PATH))
    return build_robot(mjspec, configure, FOOT_SUFFIX)


def configure(model, simulation_step=0.005):
    model.opt.timestep = simulation_step
    model = configure_actuators(model)
    return configure_joints(model)


def configure_actuators(model):
    gains = np.array([40.1792, 99.0984, 40.1792, 99.0984, 28.5012, 28.5012, 40.1792, 99.0984, 40.1792, 99.0984, 28.5012, 28.5012, 40.1792, 28.5012, 28.5012, 14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783, 14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783], "float32")  # fmt: skip
    dampings = np.array([2.5579, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144, 2.5579, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144, 2.5579, 1.8144, 1.8144, 0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681, 0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681], "float32")  # fmt: skip
    limits = np.array([88, 139, 88, 139, 50, 50, 88, 139, 88, 139, 50, 50, 88, 50, 50, 25, 25, 25, 25, 25, 5, 5, 25, 25, 25, 25, 25, 5, 5], "float32")  # fmt: skip
    model.actuator_gainprm[:, 0] = gains
    model.actuator_biasprm[:, 1] = -gains
    model.actuator_biasprm[:, 2] = -dampings
    model.actuator_forcelimited[:] = 1
    model.actuator_forcerange[:, 0] = -limits
    model.actuator_forcerange[:, 1] = limits
    return model


def configure_joints(model):
    armature = np.array([0.010178, 0.025102, 0.010178, 0.025102, 0.007219, 0.007219, 0.010178, 0.025102, 0.010178, 0.025102, 0.007219, 0.007219, 0.010178, 0.007219, 0.007219, 0.003610, 0.003610, 0.003610, 0.003610, 0.003610, 0.004250, 0.004250, 0.003610, 0.003610, 0.003610, 0.003610, 0.003610, 0.004250, 0.004250], "float32")  # fmt: skip
    model.dof_damping[6:] = 0.0
    model.dof_frictionloss[6:] = 0.0
    model.dof_armature[6:] = armature
    return model


ARM_KEYWORDS = ("shoulder", "elbow", "wrist")
WAIST_KEYWORDS = ("waist",)
HIP_KEYWORDS = ("hip_roll", "hip_yaw")
FOOT_FORCES = ("left_foot_collision_0_force", "left_foot_collision_1_force", "left_foot_collision_2_force", "left_foot_collision_3_force", "right_foot_collision_0_force", "right_foot_collision_1_force", "right_foot_collision_2_force", "right_foot_collision_3_force")  # fmt: skip
FOOT_VELOCITIES = ("left_foot_linvel", "right_foot_linvel")
RewardIndices = namedtuple("RewardIndices", "arms, waists, hips, foot_forces, foot_velocities, other_bodies")  # fmt: skip


def build_reward_indices(robot):
    arms = select_joint_slots(robot.joints, ARM_KEYWORDS)
    waists = select_joint_slots(robot.joints, WAIST_KEYWORDS)
    hips = select_joint_slots(robot.joints, HIP_KEYWORDS)
    forces = read_sensor_addresses(robot.sensors, FOOT_FORCES)
    velocities = read_sensor_addresses(robot.sensors, FOOT_VELOCITIES)
    # the reference exempts every ankle link, pitch included, from the
    # undesired contact penalty
    other = reject_keyword_args(robot.bodies, "ankle")[1:]
    return RewardIndices(arms, waists, hips, forces, velocities, other)


def select_joint_slots(joints, keywords, free_base_size=7):
    slots = []
    for joint in joints:
        if any(keyword in joint.name for keyword in keywords):
            slots.append(joint.qpos_address - free_base_size)
    return jp.array(slots)
