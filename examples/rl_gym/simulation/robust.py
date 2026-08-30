from collections import namedtuple

import jax.numpy as jp
from jax import random as jr

from .common import STATE_FIELDS
from .common import StepCounters, Tile
from .common import build_qpos, build_qvel, build_physics_state
from .common import sample_joint_velocity, sample_command, sample_push_step
from .common import build_actor_observation, build_critic_observation
from .common import build_observation_history
from .common import Transition, apply_scheduled_push, compute_targets
from .common import compute_gravity, compute_robust_reward, is_fallen
from .common import resample_command, run_physics, update_level
from .common import update_observation_history

State = namedtuple("State", STATE_FIELDS)


def reset(key, dynamics, level, max_speed, physics_template, origins):
    keys = jr.split(key, 6)
    column = jr.randint(keys[0], (), 0, origins.shape[1])
    origin = origins[level, column]
    tile = Tile(level, column, origin)
    qpos = build_qpos(keys[1], dynamics.qpos0, origin)
    num_joints = physics_template.ctrl.shape[0]
    joint_velocity = sample_joint_velocity(keys[2], num_joints)
    qvel = build_qvel(physics_template.qvel, joint_velocity)
    physics_state = build_physics_state(dynamics, physics_template, qpos, qvel)
    command = sample_command(keys[3], max_speed)
    push_step = sample_push_step(keys[4], 0)
    action = jp.zeros(num_joints)
    actor = build_actor_observation(keys[5], physics_state, command, action)
    critic = build_critic_observation(physics_state, command, action)
    history = build_observation_history(actor, critic)
    counters = StepCounters(jp.array(0), jp.array(0), push_step)
    reward_sum = jp.zeros(())
    return State(physics_state, history, tile, counters, command, reward_sum)


def step(key, dynamics, state, action, max_speed, robot, indices, tile_size, max_level, episode_steps=1000):  # fmt: skip
    keys = jr.split(key, 3)
    counters = state.counters
    push_args = keys[0], state.physics_state, counters.episode, counters.push
    pushed, push_step = apply_scheduled_push(*push_args)
    targets = compute_targets(action)
    physics_state = run_physics(dynamics, pushed, targets)
    episode = counters.episode + 1
    reward_args = physics_state, state, action, robot, indices, episode
    reward, terms = compute_robust_reward(*reward_args)
    command_args = keys[1], state.command, counters.command + 1, max_speed
    command, command_step = resample_command(*command_args)
    actor = build_actor_observation(keys[2], physics_state, command, action)
    critic = build_critic_observation(physics_state, command, action)
    history = update_observation_history(state.history, actor, critic)
    counters = StepCounters(episode, command_step, push_step)
    reward_sum = state.reward_sum + reward
    state_args = physics_state, history, state.tile, counters, command
    state = State(*state_args, reward_sum)
    gravity = compute_gravity(physics_state.qpos[3:7])
    fallen = is_fallen(physics_state, gravity)
    timeout = episode >= episode_steps
    level = update_level(state, tile_size, max_level)
    done = fallen | timeout
    return state, Transition(reward, done, timeout, level, terms)
