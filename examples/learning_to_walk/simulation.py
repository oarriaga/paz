"""MuJoCo support code for the unitree_rl_lab G1 velocity policy.

The constants are the exported deployment values from a run's
params/deploy.yaml, converted where needed. Two joint orders meet here.
The observation and the action live in the Isaac order the policy was
trained in. Everything touching the plant lives in the Unitree SDK order
the MuJoCo model declares: DEFAULT_ANGLES, GAINS, DAMPINGS, the target
angles and the control vector. SDK_INDICES converts between them.
"""

from collections import namedtuple

import mujoco
import numpy as np

from paz.backend.lie.quaternion import to_matrix
from policy import FRAME_DIM
from policy import NUM_HISTORY_FRAMES
from policy import NUM_JOINTS

SIMULATION_STEP = 0.005
CONTROL_DECIMATION = 4
CONTROL_FREQUENCY = 1.0 / (SIMULATION_STEP * CONTROL_DECIMATION)
SIMULATION_FREQUENCY = 1.0 / SIMULATION_STEP
ACTION_SCALE = 0.25
ANGULAR_VELOCITY_SCALE = 0.2
JOINT_VELOCITY_SCALE = 0.05

SPAWN_HEIGHT = 0.8
SPAWN_RADIUS = 0.5
FALL_ANGLE = 0.8
FALL_HEIGHT = 0.2
SPAWN_JOINT_SPEED = 1.0

MAX_FORWARD_SPEED = 1.0
MAX_BACKWARD_SPEED = 0.5
MAX_LATERAL_SPEED = 0.3
MAX_YAW_RATE = 0.2

PUSH_INTERVAL = (3.0, 8.0)
PUSH_DURATION = 0.2
PUSH_FORCE = 200.0
TRAINED_PUSH_SPEED = 1.0
PUSH_ARROW_SCALE = 0.002
PUSH_ARROW_WIDTH = 0.02
PUSH_ARROW_COLOR = np.array([1.0, 0.35, 0.1, 1.0], "float32")

TERRAIN_NAME = "terrain"
TERRAIN_BASE = 1.0

# deploy.yaml joint_ids_map: for every Isaac joint the policy was trained
# on, its index in the Unitree SDK order the MuJoCo model declares.
SDK_INDICES = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23,
     5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])

DEFAULT_ANGLES = np.array(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
     -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
     0.0, 0.0, 0.0,
     0.3, 0.25, 0.0, 0.97, 0.15, 0.0, 0.0,
     0.3, -0.25, 0.0, 0.97, -0.15, 0.0, 0.0], "float32")

GAINS = np.array(
    [100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
     100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
     200.0, 40.0, 40.0,
     40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
     40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0], "float32")

DAMPINGS = np.array(
    [2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
     2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
     5.0, 5.0, 5.0,
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "float32")

# IsaacLab keeps one history per observation term rather than one history of
# whole frames, so the flat observation groups every term's five samples
# together. These split a frame back into its terms.
TERM_OFFSETS = (3, 6, 9, 38, 67)

Plant = namedtuple("Plant", "model data height")
Rollout = namedtuple("Rollout", "history action targets")


def build_flat_plant(scene_path):
    model, data = compile_plant(mujoco.MjSpec.from_file(str(scene_path)))
    reset_plant(model, data, SPAWN_HEIGHT)
    return Plant(model, data, SPAWN_HEIGHT)


def build_terrain_plant(scene_path, terrain):
    spec = mujoco.MjSpec.from_file(str(scene_path))
    add_terrain(spec, terrain)
    model, data = compile_plant(spec)
    height = SPAWN_HEIGHT + compute_spawn_rise(terrain)
    reset_plant(model, data, height)
    return Plant(model, data, height)


def compile_plant(spec):
    model = spec.compile()
    model.opt.timestep = SIMULATION_STEP
    return model, mujoco.MjData(model)


def reset_plant(model, data, height):
    mujoco.mj_resetData(model, data)
    data.qpos[2] = height
    data.qpos[7:7 + NUM_JOINTS] = DEFAULT_ANGLES
    mujoco.mj_forward(model, data)


def randomize_plant(model, data, height, rng):
    # The training reset drew a heading and joint speeds. Without them a
    # deterministic terrain gives every seed the same rollout.
    reset_plant(model, data, height)
    data.qpos[3:7] = sample_heading(rng)
    speeds = rng.uniform(-SPAWN_JOINT_SPEED, SPAWN_JOINT_SPEED, NUM_JOINTS)
    data.qvel[6:6 + NUM_JOINTS] = speeds
    mujoco.mj_forward(model, data)


def sample_heading(rng):
    # MuJoCo orders a quaternion w, x, y, z, which the paz
    # from_rotation_vector helper does not, so the yaw is written out here.
    yaw = rng.uniform(-np.pi, np.pi)
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def add_terrain(spec, terrain):
    cells = terrain.elevations.shape[0]
    size = [terrain.radius, terrain.radius, terrain.peak, TERRAIN_BASE]
    field = spec.add_hfield(name=TERRAIN_NAME, size=size)
    field.nrow, field.ncol = cells, cells
    field.userdata = terrain.elevations.ravel()
    add_terrain_geom(spec)


def add_terrain_geom(spec):
    # The released ground plane stays underneath. Replacing it leaves
    # nothing to walk on past the edge of the patch, and a run that reaches
    # the edge then falls out of the world rather than off the terrain.
    geom = spec.worldbody.add_geom()
    geom.name = TERRAIN_NAME
    geom.type = mujoco.mjtGeom.mjGEOM_HFIELD
    geom.hfieldname = TERRAIN_NAME
    geom.material = spec.geom("floor").material


def compute_spawn_rise(terrain):
    # Spawn on the tallest cell the feet can reach, so the robot never
    # starts inside a rock or above the far side of a slope.
    cells = terrain.elevations.shape[0]
    span = round(SPAWN_RADIUS * cells / (2 * terrain.radius))
    center = slice(cells // 2 - span, cells // 2 + span)
    return terrain.elevations[center, center].max() * terrain.peak


def build_rollout(data, command):
    action = np.zeros(NUM_JOINTS, "float32")
    frame = build_observation_frame(data, command, action)
    targets = compute_target_angles(action)
    return Rollout(build_history(frame), action, targets)


def update_rollout(rollout, actor, data, command):
    frame = build_observation_frame(data, command, rollout.action)
    history = update_history(rollout.history, frame)
    action = np.asarray(actor(build_observation(history)))[0]
    return Rollout(history, action, compute_target_angles(action))


def build_observation_frame(data, command, action):
    frame = np.zeros(FRAME_DIM, "float32")
    frame[0:3] = data.qvel[3:6] * ANGULAR_VELOCITY_SCALE
    frame[3:6] = compute_gravity_direction(data.qpos[3:7])
    frame[6:9] = command
    frame[9:38] = compute_joint_positions(data)
    frame[38:67] = compute_joint_velocities(data)
    frame[67:96] = action
    return frame


def compute_joint_positions(data):
    positions = data.qpos[7:7 + NUM_JOINTS] - DEFAULT_ANGLES
    return positions[SDK_INDICES]


def compute_joint_velocities(data):
    velocities = data.qvel[6:6 + NUM_JOINTS] * JOINT_VELOCITY_SCALE
    return velocities[SDK_INDICES]


def compute_gravity_direction(orientation):
    gravity = np.array([0.0, 0.0, -1.0], "float32")
    rotation = np.asarray(to_matrix(orientation))
    return rotation.transpose() @ gravity


def build_history(frame):
    # IsaacLab fills every history slot with the first sample on reset.
    return np.repeat(frame[np.newaxis], NUM_HISTORY_FRAMES, axis=0)


def update_history(history, frame):
    return np.concatenate([history[1:], frame[np.newaxis]])


def build_observation(history):
    terms = np.split(history, TERM_OFFSETS, axis=1)
    return np.concatenate([term.ravel() for term in terms])[np.newaxis]


def compute_target_angles(action):
    targets = np.empty(NUM_JOINTS, "float32")
    targets[SDK_INDICES] = action * ACTION_SCALE
    return targets + DEFAULT_ANGLES


def compute_control(data, targets):
    positions = data.qpos[7:7 + NUM_JOINTS]
    velocities = data.qvel[6:6 + NUM_JOINTS]
    return (targets - positions) * GAINS - velocities * DAMPINGS


def find_torso(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")


def apply_push(data, torso, force):
    data.xfrc_applied[torso, 0:3] = force


def sample_push(rng, force):
    angle = rng.uniform(0.0, 2 * np.pi)
    return force * np.array([np.cos(angle), np.sin(angle), 0.0])


def sample_push_step(rng, step):
    interval = rng.uniform(*PUSH_INTERVAL) / SIMULATION_STEP
    return step + round(interval)


def draw_push(scene, data, torso):
    # xfrc_applied is the only record of a shove, so the arrow reads it back
    # and clears itself the moment the shove is released.
    force = data.xfrc_applied[torso, 0:3]
    scene.ngeom = 0
    if np.any(force):
        tip = data.xpos[torso]
        add_arrow(scene, tip - force * PUSH_ARROW_SCALE, tip)


def add_arrow(scene, start, end):
    geom = scene.geoms[scene.ngeom]
    arrow = mujoco.mjtGeom.mjGEOM_ARROW
    # size, pos and mat are all overwritten by the connector below.
    blank = np.zeros(3), np.zeros(3), np.zeros(9)
    mujoco.mjv_initGeom(geom, arrow, *blank, PUSH_ARROW_COLOR)
    mujoco.mjv_connector(geom, arrow, PUSH_ARROW_WIDTH, start, end)
    scene.ngeom = scene.ngeom + 1


def compute_tilt(data):
    upright = compute_gravity_direction(data.qpos[3:7])[2]
    return np.arccos(np.clip(-upright, -1.0, 1.0))


def has_fallen(data):
    return compute_tilt(data) > FALL_ANGLE or data.qpos[2] < FALL_HEIGHT


def compute_push_speed(model, force):
    return force * PUSH_DURATION / sum(model.body_mass)
