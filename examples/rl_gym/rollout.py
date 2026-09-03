from collections import namedtuple

import jax
import jax.numpy as jp
from jax import random as jr

from networks import call_actor
from networks import call_critic
from networks import read_stdv
from ppo import Experience
from ppo import compute_value_targets
from ppo import sample_actions
from ppo import standardize_advantages

Rollout = namedtuple("Rollout", "actor_observation, critic_observation, action, log_probability, mean, stdv, value, reward, done, terms, reward_sum, tracking_sum, level, diverged")  # fmt: skip
Metrics = namedtuple("Metrics", "reward, episode_return, episodic_tracking, terms, level, episode_length, divergences")  # fmt: skip


def build_collect(actor, critic, reset, step, num_steps=24, gamma=0.99):

    def collect(state, parameters, key, max_speed):

        def advance(carry, _):
            state, key = carry
            keys = jr.split(key, 4)
            history = state.history
            stdv = read_stdv(parameters)
            mean = call_actor(actor, parameters.actor, history.actor)
            action, log_probability = sample_actions(keys[1], mean, stdv)
            value = call_critic(critic, parameters.critic, history.critic)
            state, transition = step(keys[2], state, action, max_speed)
            sums = state.reward_sum, state.tracking_sum
            reward = bootstrap(critic, parameters, state, transition, gamma)
            fresh = reset(keys[3], transition.level, state.tile.column, max_speed)  # fmt: skip
            state = select_done(transition.done, fresh, state)
            stdv = jp.broadcast_to(stdv, mean.shape)
            done = transition.done.astype(jp.float32)
            step_args = history.actor, history.critic, action, log_probability, mean, stdv, value, reward, done, transition.terms, *sums, state.tile.level, transition.diverged  # fmt: skip
            return (state, keys[0]), Rollout(*step_args)

        (state, key), rollout = jax.lax.scan(advance, (state, key), None, length=num_steps)  # fmt: skip
        last_value = call_critic(critic, parameters.critic, state.history.critic)  # fmt: skip
        target_args = rollout.reward, rollout.done, rollout.value, last_value
        value_target = compute_value_targets(*target_args)
        advantage = standardize_advantages(value_target - rollout.value)
        experience = build_experience(rollout, value_target, advantage)
        return state, key, experience, compute_metrics(rollout)

    return collect


def bootstrap(critic, parameters, state, transition, gamma):
    # a diverged state may hold a non-finite value; it never times out, so
    # selecting instead of multiplying keeps the NaN out of the reward
    values = call_critic(critic, parameters.critic, state.history.critic)
    bootstrapped = jp.where(transition.timeout > 0.0, values, 0.0)
    return transition.reward + gamma * bootstrapped


def select_done(done, fresh, state):

    def select(new, old):
        if is_per_environment(old, done):
            shape = (done.shape[0],) + (1,) * (old.ndim - 1)
            selected = jp.where(done.reshape(shape), new, old)
        else:
            selected = old
        return selected

    return jax.tree.map(select, fresh, state)


def is_per_environment(values, done):
    return values.ndim > 0 and values.shape[0] == done.shape[0]


def build_experience(rollout, value_target, advantage):
    args = rollout.actor_observation, rollout.critic_observation, rollout.action, rollout.log_probability, rollout.mean, rollout.stdv, rollout.value, value_target, advantage  # fmt: skip
    return jax.tree.map(merge_steps, Experience(*args))


def merge_steps(values):
    return values.reshape((-1,) + values.shape[2:])


def compute_metrics(rollout, control_step=0.02, episode_seconds=20.0):
    completed = jp.sum(rollout.done)
    total_return = jp.sum(rollout.reward_sum * rollout.done)
    episode_return = total_return / jp.maximum(completed, 1.0)
    # the episodic tracking return rated against the full horizon, as the
    # reference's command curriculum reads it from the resetting episodes
    total_tracking = jp.sum(rollout.tracking_sum * rollout.done)
    episodic_tracking = total_tracking / jp.maximum(completed, 1.0) * control_step / episode_seconds  # fmt: skip
    terms = jp.mean(rollout.terms, axis=(0, 1))
    level = jp.mean(rollout.level.astype(jp.float32))
    episode_length = rollout.done.size / jp.maximum(completed, 1.0)
    divergences = jp.sum(rollout.diverged)
    return Metrics(jp.mean(rollout.reward), episode_return, episodic_tracking, terms, level, episode_length, divergences)  # fmt: skip
