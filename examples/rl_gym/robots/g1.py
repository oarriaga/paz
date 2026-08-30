from collections import namedtuple
from pathlib import Path

import jax.numpy as jp
import mujoco
import numpy as np

from robots import build as build_robot
from robots import read_sensor_addresses, reject_keyword_args

ASSET_PATH = Path(__file__).parent.parent / "assets/g1_29dof.xml"
DEFAULT_ANGLES = jp.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0, 0.3, 0.25, 0.0, 0.97, 0.15, 0.0, 0.0, 0.3, -0.25, 0.0, 0.97, -0.15, 0.0, 0.0])  # fmt: skip
# per-joint actuator velocity limits from the reference articulation, which
# its solver clamps every physics step
VELOCITY_LIMITS = jp.array([32.0, 20.0, 32.0, 20.0, 30.0, 30.0, 32.0, 20.0, 32.0, 20.0, 30.0, 30.0, 32.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 22.0, 22.0, 37.0, 37.0, 37.0, 37.0, 37.0, 22.0, 22.0])  # fmt: skip
FOOT_SUFFIX = "ankle_roll_link"


def G1DoF29():
    mjspec = mujoco.MjSpec.from_file(str(ASSET_PATH))
    return build_robot(mjspec, configure, FOOT_SUFFIX)


def configure(model, simulation_step=0.005):
    model.opt.timestep = simulation_step
    model = configure_actuators(model)
    return configure_joints(model)


def configure_actuators(model):
    gains = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40, 200, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40], "float32")  # fmt: skip
    dampings = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "float32")  # fmt: skip
    limits = np.array([88, 139, 88, 139, 25, 25, 88, 139, 88, 139, 25, 25, 88, 25, 25, 25, 25, 25, 25, 25, 5, 5, 25, 25, 25, 25, 25, 5, 5], "float32")  # fmt: skip
    model.actuator_gainprm[:, 0] = gains
    model.actuator_biasprm[:, 1] = -gains
    model.actuator_biasprm[:, 2] = -dampings
    model.actuator_forcelimited[:] = 1
    model.actuator_forcerange[:, 0] = -limits
    model.actuator_forcerange[:, 1] = limits
    return model


def configure_joints(model):
    model.dof_damping[6:] = 0.0
    model.dof_frictionloss[6:] = 0.0
    model.dof_armature[6:] = 0.01
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
