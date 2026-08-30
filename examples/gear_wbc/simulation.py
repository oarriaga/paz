"""MuJoCo support code for the PAZ GEAR-WBC demonstration.

The constants below are the released deployment values taken from
decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml.
"""

from collections import deque
from collections import namedtuple
from functools import partial

import jax
import mujoco
import numpy as np

from paz.backend.lie.quaternion import to_matrix
from paz.models.foundation.gear_wbc.model import ACTION_DIM
from paz.models.foundation.gear_wbc.model import FRAME_DIM
from paz.models.foundation.gear_wbc.model import NUM_HISTORY_FRAMES

NUM_JOINTS = 29
SIMULATION_STEP = 0.005
CONTROL_DECIMATION = 4
ACTION_SCALE = 0.25
ANGULAR_VELOCITY_SCALE = 0.5
JOINT_VELOCITY_SCALE = 0.05
VELOCITY_COMMAND_SCALE = np.array([2.0, 2.0, 0.5], "float32")
DEFAULT_HEIGHT = 0.74
BALANCE_SPEED = 0.05

# MuJoCo reads a heightfield size as (radius_x, radius_y, peak, base). The
# 20 by 20 metre patch and the 256 cell grid match the rough terrain MuJoCo
# Playground walks its G1 over. Its peak is 0.05, where GEAR-WBC falls on
# every seed, so the default here is lower.
TERRAIN_NAME = "rocky"
TERRAIN_RADIUS = 10.0
TERRAIN_BASE = 1.0
TERRAIN_CELLS = 256
ROCK_HEIGHT = 0.03

LOWER_BODY_ANGLES = np.array(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
     -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
     0.0, 0.0, 0.0], "float32")
LOWER_BODY_GAINS = np.array(
    [150, 150, 150, 200, 40, 40,
     150, 150, 150, 200, 40, 40,
     250, 250, 250], "float32")
LOWER_BODY_DAMPINGS = np.array(
    [2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
     2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
     5.0, 5.0, 5.0], "float32")
UPPER_BODY_GAIN = 100.0
UPPER_BODY_DAMPING = 0.5

# GEAR-WBC is a decoupled controller: it commands the lower body only, so
# the arms are held at the zeros the observation is normalized against.
UPPER_BODY_ANGLES = np.zeros(NUM_JOINTS - ACTION_DIM, "float32")
DEFAULT_ANGLES = np.concatenate([LOWER_BODY_ANGLES, UPPER_BODY_ANGLES])

Command = namedtuple("Command", "velocity height orientation")


def build_command(speed):
    velocity = np.array([speed, 0.0, 0.0], "float32")
    orientation = np.zeros(3, "float32")
    return Command(velocity, DEFAULT_HEIGHT, orientation)


def build_plant(scene_path):
    return compile_plant(mujoco.MjSpec.from_file(str(scene_path)))


def build_rocky_plant(scene_path, seed, rock_height):
    spec = mujoco.MjSpec.from_file(str(scene_path))
    add_rocky_terrain(spec, seed, rock_height)
    model, data = compile_plant(spec)
    data.qpos[2] = data.qpos[2] + rock_height
    return model, data


def compile_plant(spec):
    model = spec.compile()
    model.opt.timestep = SIMULATION_STEP
    return model, mujoco.MjData(model)


def add_rocky_terrain(spec, seed, rock_height):
    size = [TERRAIN_RADIUS, TERRAIN_RADIUS, rock_height, TERRAIN_BASE]
    terrain = spec.add_hfield(name=TERRAIN_NAME, size=size)
    terrain.nrow, terrain.ncol = TERRAIN_CELLS, TERRAIN_CELLS
    terrain.userdata = build_elevations(seed)
    swap_floor_for_terrain(spec)


def swap_floor_for_terrain(spec):
    floor = spec.geom("floor")
    floor.type = mujoco.mjtGeom.mjGEOM_HFIELD
    floor.hfieldname = TERRAIN_NAME


def build_elevations(seed):
    # MuJoCo rescales elevations to [0, 1], so the peak comes from the size.
    return np.random.default_rng(seed).uniform(size=TERRAIN_CELLS ** 2)


def build_history():
    empty_frame = np.zeros(FRAME_DIM, "float32")
    frames = [empty_frame] * NUM_HISTORY_FRAMES
    return deque(frames, maxlen=NUM_HISTORY_FRAMES)


def build_observation_frame(data, command, previous_action):
    joint_end = 13 + NUM_JOINTS
    velocity_end = joint_end + NUM_JOINTS
    frame = np.zeros(FRAME_DIM, "float32")
    frame[0:7] = build_command_frame(command)
    frame[7:10] = data.qvel[3:6] * ANGULAR_VELOCITY_SCALE
    frame[10:13] = compute_gravity_direction(data.qpos[3:7])
    frame[13:joint_end] = compute_joint_positions(data)
    frame[joint_end:velocity_end] = compute_joint_velocities(data)
    frame[velocity_end:] = previous_action
    return frame


def build_command_frame(command):
    frame = np.zeros(7, "float32")
    frame[0:3] = command.velocity * VELOCITY_COMMAND_SCALE
    frame[3] = command.height
    frame[4:7] = command.orientation
    return frame


def compute_joint_positions(data):
    return data.qpos[7:7 + NUM_JOINTS] - DEFAULT_ANGLES


def compute_joint_velocities(data):
    return data.qvel[6:6 + NUM_JOINTS] * JOINT_VELOCITY_SCALE


def compute_gravity_direction(orientation):
    gravity = np.array([0.0, 0.0, -1.0], "float32")
    rotation = np.asarray(to_matrix(orientation))
    return rotation.transpose() @ gravity


def update_history(history, frame):
    history.append(frame)
    return np.concatenate(history)[np.newaxis]


def compile_actors(models):
    # Called eagerly, these actors cost 7.7 ms and dominate the 20 ms
    # control period. Both are compiled here so neither expert stalls the
    # loop the first time a command selects it.
    balance = jax.jit(partial(models.balance, training=False))
    walk = jax.jit(partial(models.walk, training=False))
    observation = np.zeros((1, FRAME_DIM * NUM_HISTORY_FRAMES), "float32")
    balance(observation)
    walk(observation)
    return models._replace(balance=balance, walk=walk)


def select_actor(actors, command):
    # The release disagrees with itself at exactly BALANCE_SPEED: the MuJoCo
    # runner uses <= and the deploy policy uses <. This follows the runner.
    if np.linalg.norm(command.velocity) <= BALANCE_SPEED:
        actor = actors.balance
    else:
        actor = actors.walk
    return actor


def compute_action(actors, observation, command):
    actor = select_actor(actors, command)
    return np.asarray(actor(observation))[0]


def compute_target_angles(action):
    return action * ACTION_SCALE + LOWER_BODY_ANGLES


def compute_control(data, target_angles):
    lower_torques = compute_lower_body_torques(data, target_angles)
    upper_torques = compute_upper_body_torques(data)
    return np.concatenate([lower_torques, upper_torques])


def compute_lower_body_torques(data, target_angles):
    positions = data.qpos[7:7 + ACTION_DIM]
    velocities = data.qvel[6:6 + ACTION_DIM]
    args = target_angles, positions, velocities
    return compute_torques(*args, LOWER_BODY_GAINS, LOWER_BODY_DAMPINGS)


def compute_upper_body_torques(data):
    positions = data.qpos[7 + ACTION_DIM:7 + NUM_JOINTS]
    velocities = data.qvel[6 + ACTION_DIM:6 + NUM_JOINTS]
    args = UPPER_BODY_ANGLES, positions, velocities
    return compute_torques(*args, UPPER_BODY_GAIN, UPPER_BODY_DAMPING)


def compute_torques(targets, positions, velocities, gains, dampings):
    return (targets - positions) * gains - velocities * dampings
