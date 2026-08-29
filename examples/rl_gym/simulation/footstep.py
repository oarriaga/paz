from collections import namedtuple

import jax.numpy as jp

import rewards
from jax import random as jr

from .common import EPISODE_STEPS, STATE_FIELDS
from .common import ACTOR_FIELDS, CRITIC_FIELDS
from .common import StepCounters, Tile
from .common import build_qpos, build_qvel, build_physics_state
from .common import sample_command, sample_push_step
from .common import build_actor_observation, build_critic_observation
from .common import build_observation_history, compute_local_phase
from .common import rotate_yaw, rotate_yaw_inverse
from .common import Transition, apply_scheduled_push, compute_targets
from .common import compute_gravity, compute_robust_reward, is_fallen
from .common import discard_diverged, sanitize_diverged
from .common import read_foot_contact, resample_command, run_physics
from .common import update_level, update_observation_history

State = namedtuple("State", STATE_FIELDS + ", feet")
FootState = namedtuple("FootState", "targets, contact, air_time, phase, switch_phase")  # fmt: skip
ActorObservation = namedtuple("ActorObservation", ACTOR_FIELDS + ", footstep")  # fmt: skip
CriticObservation = namedtuple("CriticObservation", CRITIC_FIELDS + ", footstep")  # fmt: skip


def reset(key, dynamics, level, max_speed, physics_template, origins, feet):  # fmt: skip
    keys = jr.split(key, 6)
    column = jr.randint(keys[0], (), 0, origins.shape[1])
    origin = origins[level, column]
    qpos = build_qpos(keys[1], dynamics.qpos0, origin)
    num_joints = physics_template.ctrl.shape[0]
    qvel = build_qvel(physics_template.qvel)
    physics_state = build_physics_state(dynamics, physics_template, qpos, qvel)
    command = sample_command(keys[2], max_speed)
    push_step = sample_push_step(keys[3], 0)
    phase = compute_local_phase(0)
    targets = generate_both_targets(keys[4], physics_state, command, feet)
    action = jp.zeros(num_joints)
    history_args = keys[5], physics_state, command, action, targets, phase
    history = build_initial_history(*history_args)
    tile = Tile(level, column, origin)
    counters = StepCounters(jp.array(0), jp.array(0), push_step)
    reward_sum = jp.zeros(())
    foot_state = build_initial_feet(targets, phase)
    state_args = physics_state, history, tile, counters, command, reward_sum, foot_state  # fmt: skip
    return State(*state_args)


def build_initial_history(key, physics_state, command, action, targets, phase):
    actor = build_actor_observation(key, physics_state, command, action)
    critic = build_critic_observation(physics_state, command, action)
    term = build_target_term(physics_state, targets, phase)
    actor = ActorObservation(*actor, term)
    critic = CriticObservation(*critic, term)
    return build_observation_history(actor, critic)


def build_initial_feet(targets, phase, switch_phase=2.0):
    contact = jp.zeros(2, dtype=bool)
    air_time, switch = jp.zeros(2), jp.full(2, switch_phase)
    return FootState(targets, contact, air_time, phase, switch)


def generate_both_targets(key, physics_state, command, feet):
    keys = jr.split(key, 2)
    left = generate_target(keys[0], physics_state, command, feet, 0)
    right = generate_target(keys[1], physics_state, command, feet, 1)
    return jp.stack((left, right))


def generate_target(key, physics_state, command, feet, foot, stride=0.58, offset=0.12):  # fmt: skip
    low, high = jp.array([-0.15, -0.10]), jp.array([0.15, 0.10])
    noise = jr.uniform(key, (2,), minval=low, maxval=high)
    hip_sign = 1.0 if foot == 0 else -1.0
    forward = command.forward * stride + noise[0]
    lateral = command.sideways * stride + offset * hip_sign + noise[1]
    local = jp.array([forward, lateral, 0.0])
    displacement = rotate_yaw(physics_state.qpos[3:7], local)
    target = physics_state.qpos[:3] + displacement
    return target.at[2].set(physics_state.xpos[feet[foot], 2])


def build_target_term(physics_state, targets, phase, stance_threshold=0.55):
    offsets = targets - physics_state.qpos[:3]
    left = rotate_yaw_inverse(physics_state.qpos[3:7], offsets[0])[:2]
    right = rotate_yaw_inverse(physics_state.qpos[3:7], offsets[1])[:2]
    clock = 2.0 * jp.pi * jp.mod(phase[0], 1.0)
    swing = (phase >= stance_threshold).astype(jp.float32)
    gait = jp.array([jp.sin(clock), jp.cos(clock), swing[0] - swing[1]])
    return jp.concatenate((left, right, gait))


def step(key, dynamics, state, action, max_speed, robot, indices, tile_size, max_level, episode_steps=EPISODE_STEPS):  # fmt: skip
    keys = jr.split(key, 12)
    counters = state.counters
    push_args = keys[0], state.physics_state, counters.episode, counters.push
    pushed, push_step = apply_scheduled_push(*push_args)
    physics_state = run_physics(dynamics, pushed, compute_targets(action))
    physics_state, diverged = sanitize_diverged(physics_state, dynamics)
    episode = counters.episode + 1
    phase = compute_local_phase(episode)
    feet_args = keys[3:11], physics_state, state, phase, robot, indices
    feet = update_feet(*feet_args)
    reward_args = physics_state, state, action, robot, indices, episode
    reward, terms = compute_robust_reward(*reward_args)
    diverged, reward, terms = discard_diverged(diverged, reward, terms)
    bonus = compute_touchdown_bonus(physics_state, state, feet, robot)
    bonus = jp.where(diverged, 0.0, bonus)
    command_args = keys[1], state.command, counters.command + 1, max_speed
    command, command_step = resample_command(*command_args)
    term = build_target_term(physics_state, feet.targets, phase)
    # foot targets come from xpos, which sanitize_diverged does not cover
    term = jp.where(diverged, jp.zeros_like(term), term)
    history_args = keys[2], physics_state, command, action, term
    history = update_observation_history(state.history, *build_observations(*history_args))  # fmt: skip
    counters = StepCounters(episode, command_step, push_step)
    reward_sum = state.reward_sum + reward + bonus
    state_args = physics_state, history, state.tile, counters, command
    state = State(*state_args, reward_sum, feet)
    gravity = compute_gravity(physics_state.qpos[3:7])
    timeout = (episode >= episode_steps) & ~diverged
    done = is_fallen(physics_state, gravity) | timeout | diverged
    fresh_level = update_level(keys[11], state, tile_size, max_level)
    level = jp.where(diverged, state.tile.level, fresh_level)
    transition = Transition(reward + bonus, done, timeout, level, terms, diverged)  # fmt: skip
    return state, transition


def build_observations(key, physics_state, command, action, term):
    actor = build_actor_observation(key, physics_state, command, action)
    critic = build_critic_observation(physics_state, command, action)
    return ActorObservation(*actor, term), CriticObservation(*critic, term)


def update_feet(keys, physics_state, state, phase, robot, indices, control_step=0.02):  # fmt: skip
    contact = read_foot_contact(physics_state, indices.foot_forces)
    air_time = jp.where(contact, 0.0, state.feet.air_time + control_step)
    target_args = keys, physics_state, state, phase, robot.contact_bodies
    targets, switch_phase = update_targets(*target_args)
    return FootState(targets, contact, air_time, phase, switch_phase)


def update_targets(keys, physics_state, state, phase, feet, stance=0.55):
    targets = state.feet.targets
    switch_phase = state.feet.switch_phase
    entered = (state.feet.phase < stance) & (phase >= stance)
    for foot in range(2):
        args = physics_state, state.command, feet, foot
        fresh = generate_target(keys[foot], *args)
        targets = set_foot(targets, foot, entered[foot], fresh)
        switch = sample_switch_phase(keys[foot + 2], keys[foot + 4])
        switch_phase = set_foot(switch_phase, foot, entered[foot], switch)
        switching = phase[foot] >= switch_phase[foot]
        fresh = generate_target(keys[foot + 6], *args)
        targets = set_foot(targets, foot, switching, fresh)
        switch_phase = set_foot(switch_phase, foot, switching, 2.0)
    return targets, switch_phase


def set_foot(values, foot, condition, value):
    return values.at[foot].set(jp.where(condition, value, values[foot]))


def sample_switch_phase(draw_key, phase_key, probability=0.15, low=0.6, high=0.8):  # fmt: skip
    switch = jr.uniform(phase_key, (), minval=low, maxval=high)
    return jp.where(jr.uniform(draw_key) < probability, switch, 2.0)


def compute_touchdown_bonus(physics_state, state, feet, robot, weight=20.0, control_step=0.02):  # fmt: skip
    touchdown = compute_touchdown(state, feet)
    positions = physics_state.xpos[robot.contact_bodies, :2]
    error = jp.linalg.norm(positions - feet.targets[:, :2], axis=-1)
    accuracy = rewards.touchdown_accuracy(touchdown, error)
    return accuracy * weight * control_step


def compute_touchdown(state, feet, minimum_air_time=0.15):
    entered = feet.contact & ~state.feet.contact
    return entered & (state.feet.air_time > minimum_air_time)
