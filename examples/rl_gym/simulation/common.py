from collections import namedtuple

import jax
import jax.numpy as jp
from jax import random as jr
import paz

from mujoco import mjx

import rewards
from robots.g1 import ACTION_SCALE, DEFAULT_ANGLES

STATE_FIELDS = "physics_state, history, tile, counters, command, reward_sum"
EPISODE_STEPS = 1000
COMMAND_PERIOD = 500
ACTOR_FIELDS = "angular_velocity, gravity, command, joint_positions, joint_velocities, action"  # fmt: skip
CRITIC_FIELDS = "linear_velocity, " + ACTOR_FIELDS
ObservationHistory = namedtuple("ObservationHistory", "actor, critic")
ActorObservation = namedtuple("ActorObservation", ACTOR_FIELDS)
CriticObservation = namedtuple("CriticObservation", CRITIC_FIELDS)
Tile = namedtuple("Tile", "level, column, origin")
StepCounters = namedtuple("StepCounters", "episode, command, push")
Command = namedtuple("Command", "forward, sideways, turn")
Transition = namedtuple("Transition", "reward, done, timeout, level, terms, diverged")  # fmt: skip


def build_qpos(key, qpos, origin, heights, horizontal_scale, spawn_height=0.78):  # fmt: skip
    key1, key2 = jax.random.split(key)
    position_noise = jr.uniform(key1, (2,), minval=-0.5, maxval=0.5)
    position = origin[:2] + position_noise
    ground = read_ground_height(heights, horizontal_scale, position)
    yaw = jr.uniform(key2, (), minval=-jp.pi, maxval=jp.pi)
    qpos = qpos.at[0:2].set(position)
    qpos = qpos.at[2].set(ground + spawn_height)
    qpos = qpos.at[3:7].set(yaw_quaternion(yaw))
    return qpos.at[7:].set(DEFAULT_ANGLES)


def read_ground_height(heights, horizontal_scale, position, radius_cells=3):
    # the spawn height clears the tallest cell under the footprint, so a
    # rough tile never starts the feet inside a bump; the reference leans
    # on PhysX to absorb spawn penetration, which Euler ejects instead
    width_y = (heights.shape[0] - 1) * horizontal_scale
    width_x = (heights.shape[1] - 1) * horizontal_scale
    row = (position[1] + width_y / 2.0) / horizontal_scale
    column = (position[0] + width_x / 2.0) / horizontal_scale
    row = jp.round(row).astype(jp.int32) - radius_cells
    column = jp.round(column).astype(jp.int32) - radius_cells
    size = 2 * radius_cells + 1
    window = jax.lax.dynamic_slice(heights, (row, column), (size, size))
    return jp.max(window)


def build_qvel(qvel):
    # the reference resets every velocity to zero: its nominal joint
    # velocity range scales a zero default
    return jp.zeros_like(qvel)


def build_physics_state(dynamics, physics_template, qpos, qvel):
    fields = dict(qpos=qpos, qvel=qvel, ctrl=DEFAULT_ANGLES)
    return mjx.forward(dynamics, physics_template.replace(**fields))


def read_action(physics_state):
    offsets = physics_state.ctrl - DEFAULT_ANGLES
    return offsets / ACTION_SCALE


def build_actor_observation(key, physics_state, command, action):
    keys = jr.split(key, 4)
    angular = physics_state.qvel[3:6] + jr.uniform(keys[0], (3,), minval=-0.2, maxval=0.2)  # fmt: skip
    gravity = compute_gravity(physics_state.qpos[3:7])
    gravity = gravity + jr.uniform(keys[1], (3,), minval=-0.05, maxval=0.05)
    positions = physics_state.qpos[7:] - DEFAULT_ANGLES
    shape = positions.shape
    positions = positions + jr.uniform(keys[2], shape, minval=-0.01, maxval=0.01)  # fmt: skip
    velocities = physics_state.qvel[6:] + jr.uniform(keys[3], shape, minval=-1.5, maxval=1.5)  # fmt: skip
    args = angular * 0.2, gravity, jp.stack(command), positions, velocities * 0.05, action  # fmt: skip
    return ActorObservation(*args)


def build_critic_observation(physics_state, command, action):
    linear = rotate_inverse(physics_state.qpos[3:7], physics_state.qvel[:3])
    gravity = compute_gravity(physics_state.qpos[3:7])
    positions = physics_state.qpos[7:] - DEFAULT_ANGLES
    velocities = physics_state.qvel[6:] * 0.05
    args = linear, physics_state.qvel[3:6] * 0.2, gravity, jp.stack(command), positions, velocities, action  # fmt: skip
    return CriticObservation(*args)


def build_observation_history(actor, critic, num_history=5):
    actor_history = build_history(actor, num_history)
    critic_history = build_history(critic, num_history)
    return ObservationHistory(actor_history, critic_history)


def build_history(observation, num_history=5):
    terms = []
    for term in observation:
        stacked = jp.repeat(jp.expand_dims(term, axis=0), num_history, axis=0)
        terms.append(stacked)
    return type(observation)(*terms)


def yaw_quaternion(yaw):
    half = yaw / 2.0
    return jp.array([jp.cos(half), 0.0, 0.0, jp.sin(half)])


def compute_gravity(quaternion):
    return rotate_inverse(quaternion, jp.array([0.0, 0.0, -1.0]))


def rotate_inverse(quaternion, vector):
    return paz.quaternion.to_matrix(quaternion).T @ vector


def rotate_yaw(quaternion, vector):
    yaw = read_yaw(quaternion)
    cosine, sine = jp.cos(yaw), jp.sin(yaw)
    x = cosine * vector[0] - sine * vector[1]
    y = sine * vector[0] + cosine * vector[1]
    return jp.array([x, y, vector[2]])


def rotate_yaw_inverse(quaternion, vector):
    yaw = read_yaw(quaternion)
    cosine, sine = jp.cos(yaw), jp.sin(yaw)
    x = cosine * vector[0] + sine * vector[1]
    y = -sine * vector[0] + cosine * vector[1]
    return jp.array([x, y, vector[2]])


def read_yaw(quaternion):
    w, x, y, z = quaternion
    numerator = 2.0 * (w * z + x * y)
    denominator = 1.0 - 2.0 * (y**2 + z**2)
    return jp.arctan2(numerator, denominator)


def compute_local_phase(step, control_step=0.02):
    global_phase = (step * control_step % 0.8) / 0.8
    return jp.mod(global_phase + jp.array([0.0, 0.5]), 1.0)


def sample_command(key, max_speed, standing_probability=0.02):
    keys = jr.split(key, 2)
    sideways = jp.minimum(max_speed, 0.3)
    turn_rate = jp.minimum(max_speed, 0.2)
    lower = jp.array([-jp.minimum(max_speed, 0.5), -sideways, -turn_rate])
    upper = jp.array([jp.minimum(max_speed, 1.0), sideways, turn_rate])
    sample = jr.uniform(keys[0], (3,), minval=lower, maxval=upper)
    standing = jr.uniform(keys[1]) < standing_probability
    return Command(*jp.where(standing, jp.zeros(3), sample))


def sample_push_step(key, episode_step, control_step=0.02):
    seconds = jax.random.uniform(key, (), minval=3.0, maxval=8.0)
    return episode_step + jp.round(seconds / control_step).astype(jp.int32)


def compute_targets(action):
    return DEFAULT_ANGLES + action * ACTION_SCALE


PUSH_SPEED = 1.0


def apply_scheduled_push(key, physics_state, episode_step, push_step):
    # the reference overwrites the planar velocity and zeroes the vertical
    # and angular rates, as in IsaacLab's push_by_setting_velocity
    keys = jr.split(key, 2)
    kick = jr.uniform(keys[0], (2,), minval=-PUSH_SPEED, maxval=PUSH_SPEED)
    pushing = episode_step >= push_step
    pushed = jp.concatenate([kick, jp.zeros(4)])
    qvel = physics_state.qvel.at[:6].set(jp.where(pushing, pushed, physics_state.qvel[:6]))  # fmt: skip
    next_push = sample_push_step(keys[1], episode_step)
    push_step = jp.where(pushing, next_push, push_step)
    return physics_state.replace(qvel=qvel), push_step


def sanitize_diverged(physics_state, dynamics, velocity_bound=100.0, position_bound=10.0):  # fmt: skip
    # a non-finite, unphysically fast, or mechanically impossible state is
    # a solver failure, not an outcome: replace it so no poison reaches
    # the observation history, and report the divergence; the bounds sit
    # several times above the joint limits, so they also cap how large any
    # reward term or observation can get before an explosion is caught
    finite_qpos = jp.all(jp.isfinite(physics_state.qpos))
    stretch = jp.max(jp.abs(physics_state.qpos[7:]))
    stable_joints = finite_qpos & (stretch < position_bound)
    speed = jp.max(jp.abs(physics_state.qvel))
    stable_qvel = jp.isfinite(speed) & (speed < velocity_bound)
    diverged = ~(stable_joints & stable_qvel)
    qpos = jp.where(diverged, dynamics.qpos0, physics_state.qpos)
    qvel = jp.where(diverged, 0.0, physics_state.qvel)
    return physics_state.replace(qpos=qpos, qvel=qvel), diverged


def discard_diverged(diverged, reward, terms, reward_bound=1e3):
    # a diverged step carries no information, so it must not be learned
    # from; the magnitude bound also catches solver corruption that only
    # shows in sensor-driven reward channels, where the state looks legal
    # but no physically reachable state can produce such a reward
    implausible = ~jp.isfinite(reward) | (jp.abs(reward) > reward_bound)
    diverged = diverged | implausible
    reward = jp.where(diverged, 0.0, reward)
    terms = jp.where(diverged, jp.zeros_like(terms), terms)
    return diverged, reward, terms


def run_physics(dynamics, physics_state, targets, decimation=4):
    # the sensor history over the substeps feeds the contact detection,
    # which the reference pools over the last three physics steps

    def advance(physics_state, _):
        stepped = mjx.step(dynamics, physics_state.replace(ctrl=targets))
        return stepped, stepped.sensordata

    args = advance, physics_state, None
    physics_state, sensor_history = jax.lax.scan(*args, length=decimation)
    return physics_state, sensor_history


def update_observation_history(history, actor, critic):
    actor_history = update_history(history.actor, actor)
    critic_history = update_history(history.critic, critic)
    return ObservationHistory(actor_history, critic_history)


def update_history(history, observation):
    terms = []
    for past, term in zip(history, observation):
        newest = jp.expand_dims(term, axis=0)
        terms.append(jp.concatenate((past[1:], newest), axis=0))
    return type(observation)(*terms)


def resample_command(key, command, command_step, max_speed, period=COMMAND_PERIOD):  # fmt: skip
    resample = command_step >= period
    sampled = sample_command(key, max_speed)
    values = jp.where(resample, jp.stack(sampled), jp.stack(command))
    return Command(*values), jp.where(resample, 0, command_step)


def decorrelate_counters(key, state):
    # stagger the initial episode and command phases so the parallel
    # environments do not time out and resample in lockstep; the push
    # schedule shifts with the episode so no push fires at spawn
    keys = jr.split(key)
    num_envs = state.counters.episode.shape[0]
    episode = jr.randint(keys[0], (num_envs,), 0, EPISODE_STEPS)
    command = jr.randint(keys[1], (num_envs,), 0, COMMAND_PERIOD)
    counters = StepCounters(episode, command, episode + state.counters.push)
    return state._replace(counters=counters)


def read_foot_contact(sensor_history, force_addresses, threshold=1.0, history=3):  # fmt: skip
    # the reference thresholds the net foot force at one newton and pools
    # it over the last three physics substeps, so touchdown detection
    # neither counts a feather-light graze nor drops between substeps
    force = sensor_history[-history:, force_addresses].reshape(history, 2, 4, 3)  # fmt: skip
    net = jp.sum(force, axis=2)
    contact = jp.linalg.norm(net, axis=-1) > threshold
    return jp.any(contact, axis=0)


def read_foot_velocities(physics_state, velocity_addresses):
    return physics_state.sensordata[velocity_addresses].reshape(2, 3)


def count_undesired_contacts(physics_state, other_bodies, threshold=1.0):
    force = jp.linalg.norm(physics_state.cfrc_ext[other_bodies, 3:6], axis=-1)
    return jp.sum(force > threshold)


def compute_tilt(gravity):
    return jp.arccos(jp.clip(-gravity[2], -1.0, 1.0))


def is_fallen(physics_state, gravity, min_height=0.2, max_tilt=0.8):
    tilted = compute_tilt(gravity) > max_tilt
    return (physics_state.qpos[2] < min_height) | tilted


def update_level(key, state, tile_size, max_level, episode_seconds=20.0):
    travelled = state.physics_state.qpos[:2] - state.tile.origin[:2]
    distance = jp.linalg.norm(travelled)
    promote = distance > tile_size / 2
    speed = jp.linalg.norm(jp.stack(state.command)[:2])
    demote = (distance < speed * episode_seconds * 0.5) & ~promote
    level = state.tile.level + promote.astype(jp.int32)
    # environments promoted past the top respawn at a random level, as in
    # the reference, instead of piling up at the hardest tiles
    random_level = jr.randint(key, (), 0, max_level + 1)
    level = jp.where(level > max_level, random_level, level)
    return jp.clip(level - demote.astype(jp.int32), 0, max_level)


def compute_robust_reward(physics_state, sensor_history, state, action, robot, indices, step):  # fmt: skip
    command = jp.stack(state.command)
    quaternion = physics_state.qpos[3:7]
    gravity = compute_gravity(quaternion)
    # the reference pays no alive reward on the step a fall terminates
    alive = 1.0 - is_fallen(physics_state, gravity).astype(jp.float32)
    linear = rotate_inverse(quaternion, physics_state.qvel[:3])
    yaw_linear = rotate_yaw_inverse(quaternion, physics_state.qvel[:3])
    angular = physics_state.qvel[3:6]
    tracking = rewards.compute_tracking_terms(yaw_linear, angular, command)
    base_args = linear, angular, gravity, physics_state.qpos[2]
    base = rewards.compute_base_terms(*base_args)
    joint_args = read_joint_arguments(physics_state, state.physics_state, action, robot, indices)  # fmt: skip
    joint = rewards.compute_joint_terms(*joint_args)
    feet_args = read_foot_arguments(physics_state, sensor_history, robot, indices, step, command)  # fmt: skip
    feet = rewards.compute_foot_terms(*feet_args)
    posture = rewards.posture(physics_state.qpos[7:], DEFAULT_ANGLES, command)
    stability = rewards.upright(gravity), posture
    return rewards.compute_reward(tracking, joint, base, feet, alive, stability)  # fmt: skip


def read_joint_arguments(physics_state, previous, action, robot, indices, control_step=0.02):  # fmt: skip
    positions = physics_state.qpos[7:]
    velocities = physics_state.qvel[6:]
    accelerations = (velocities - previous.qvel[6:]) / control_step
    torque = physics_state.actuator_force
    last_action = read_action(previous)
    limits = compute_soft_limits(robot.joint_limits)
    return positions, velocities, accelerations, torque, action, last_action, DEFAULT_ANGLES, limits, indices  # fmt: skip


def compute_soft_limits(limits, factor=0.9):
    # the reference penalizes crossing 90% of the joint range; against the
    # hard limits the penalty would never fire
    lower, upper = limits
    midpoint = (lower + upper) / 2.0
    half = (upper - lower) * (factor / 2.0)
    return midpoint - half, midpoint + half


def read_foot_arguments(physics_state, sensor_history, robot, indices, step, command):  # fmt: skip
    contact = read_foot_contact(sensor_history, indices.foot_forces)
    velocities = read_foot_velocities(physics_state, indices.foot_velocities)
    heights = physics_state.xpos[robot.contact_bodies, 2]
    undesired = count_undesired_contacts(physics_state, indices.other_bodies)
    return step, contact, velocities, heights, command, undesired
