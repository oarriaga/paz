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

Rollout = namedtuple("Rollout", "actor_observation, critic_observation, action, log_probability, mean, stdv, value, reward, done, terms, reward_sum, diverged, level")  # fmt: skip
Metrics = namedtuple("Metrics", "reward, episode_return, terms, divergences, level, episode_length")  # fmt: skip
Normalizer = namedtuple("Normalizer", "mean, variance, count")
Normalizers = namedtuple("Normalizers", "actor, critic")


def build_collect(actor, critic, reset, step, num_steps=24, gamma=0.99):

    def collect(state, parameters, normalizers, key, max_speed):

        def advance(carry, _):
            state, key = carry
            keys = jr.split(key, 4)
            history = state.history
            stdv = read_stdv(parameters)
            actor_input = normalize(history.actor, normalizers.actor)
            mean = call_actor(actor, parameters.actor, actor_input)
            action, log_probability = sample_actions(keys[1], mean, stdv)
            critic_input = normalize(history.critic, normalizers.critic)
            value = call_critic(critic, parameters.critic, critic_input)
            state, transition = step(keys[2], state, action, max_speed)
            reward_sum = state.reward_sum
            bootstrap_args = critic, parameters, normalizers, state, transition  # fmt: skip
            reward = bootstrap(*bootstrap_args, gamma)
            fresh = reset(keys[3], transition.level, state.tile.column, max_speed)  # fmt: skip
            state = select_done(transition.done, fresh, state)
            stdv = jp.broadcast_to(stdv, mean.shape)
            done = transition.done.astype(jp.float32)
            step_args = history.actor, history.critic, action, log_probability, mean, stdv, value, reward, done, transition.terms, reward_sum, transition.diverged, transition.level  # fmt: skip
            return (state, keys[0]), Rollout(*step_args)

        (state, key), rollout = jax.lax.scan(advance, (state, key), None, length=num_steps)  # fmt: skip
        critic_input = normalize(state.history.critic, normalizers.critic)
        last_value = call_critic(critic, parameters.critic, critic_input)
        target_args = rollout.reward, rollout.done, rollout.value, last_value
        value_target = compute_value_targets(*target_args)
        valid = 1.0 - rollout.diverged.astype(jp.float32)
        advantage = standardize_advantages(value_target - rollout.value, valid)  # fmt: skip
        experience = build_experience(rollout, normalizers, value_target, advantage, valid)  # fmt: skip
        actor_stats = update_normalizer(normalizers.actor, rollout.actor_observation)  # fmt: skip
        critic_stats = update_normalizer(normalizers.critic, rollout.critic_observation)  # fmt: skip
        normalizers = Normalizers(actor_stats, critic_stats)
        return state, key, experience, normalizers, compute_metrics(rollout)

    return collect


def build_normalizers(history):
    return Normalizers(build_normalizer(history.actor), build_normalizer(history.critic))  # fmt: skip


def build_normalizer(observation):
    # one running statistic per feature, shared across the history slots

    def build(term):
        features = term.shape[-1]
        return Normalizer(jp.zeros(features), jp.ones(features), jp.asarray(1e-4))  # fmt: skip

    return type(observation)(*map(build, observation))


def normalize(observation, normalizers, epsilon=1e-8):

    def apply(term, normalizer):
        scale = jp.sqrt(normalizer.variance + epsilon)
        return (term - normalizer.mean) / scale

    return type(observation)(*map(apply, observation, normalizers))


def update_normalizer(normalizers, observations):
    # fold the rollout's per-feature statistics into the running ones

    def update(normalizer, term):
        axes = tuple(range(term.ndim - 1))
        batch_mean = jp.mean(term, axis=axes)
        batch_variance = jp.var(term, axis=axes)
        batch_count = term.size // term.shape[-1]
        total = normalizer.count + batch_count
        delta = batch_mean - normalizer.mean
        mean = normalizer.mean + delta * batch_count / total
        blended = normalizer.variance * normalizer.count + batch_variance * batch_count  # fmt: skip
        correction = delta**2 * normalizer.count * batch_count / total
        return Normalizer(mean, (blended + correction) / total, total)

    return type(normalizers)(*map(update, normalizers, observations))


def bootstrap(critic, parameters, normalizers, state, transition, gamma):
    critic_input = normalize(state.history.critic, normalizers.critic)
    values = call_critic(critic, parameters.critic, critic_input)
    return transition.reward + gamma * values * transition.timeout


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


def build_experience(rollout, normalizers, value_target, advantage, valid):
    # the stored observations are normalized with the statistics the
    # policy saw at collection time, so the update needs no normalizer
    actor_input = normalize(rollout.actor_observation, normalizers.actor)
    critic_input = normalize(rollout.critic_observation, normalizers.critic)
    args = actor_input, critic_input, rollout.action, rollout.log_probability, rollout.mean, rollout.stdv, rollout.value, value_target, advantage, valid  # fmt: skip
    return jax.tree.map(merge_steps, Experience(*args))


def merge_steps(values):
    return values.reshape((-1,) + values.shape[2:])


def compute_metrics(rollout):
    completed = jp.sum(rollout.done)
    total_return = jp.sum(rollout.reward_sum * rollout.done)
    episode_return = total_return / jp.maximum(completed, 1.0)
    terms = jp.mean(rollout.terms, axis=(0, 1))
    divergences = jp.sum(rollout.diverged)
    level = jp.mean(rollout.level.astype(jp.float32))
    episode_length = rollout.done.size / jp.maximum(completed, 1.0)
    metric_args = jp.mean(rollout.reward), episode_return, terms
    return Metrics(*metric_args, divergences, level, episode_length)
