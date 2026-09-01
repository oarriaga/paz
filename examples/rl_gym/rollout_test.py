import os
from collections import namedtuple

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax.numpy as jp
import numpy as np
from jax import random as jr

import rollout

FakeState = namedtuple("FakeState", "positions, level, constant")


def test_select_done_replaces_only_done_environments():
    done = jp.array([True, False, True])
    old = FakeState(jp.zeros((3, 2)), jp.array([0, 1, 2]), jp.array(7.0))
    fresh = FakeState(jp.ones((3, 2)), jp.array([5, 5, 5]), jp.array(9.0))
    selected = rollout.select_done(done, fresh, old)
    assert np.allclose(np.asarray(selected.positions[:, 0]), [1.0, 0.0, 1.0])
    assert np.allclose(np.asarray(selected.level), [5, 1, 5])
    assert float(selected.constant) == 7.0


def test_bootstrap_only_on_timeout():
    Parameters = namedtuple("Parameters", "critic")
    History = namedtuple("History", "critic")
    State = namedtuple("State", "history")
    Transition = namedtuple("Transition", "reward, timeout")

    def critic_call(critic, parameters, observation):
        return observation.term

    original = rollout.call_critic
    rollout.call_critic = critic_call
    try:
        observation = FakeObservation(jp.array([2.0, 3.0]))
        state = State(History(observation))
        normalizer = rollout.build_normalizer(observation)
        normalizers = rollout.Normalizers(None, normalizer)
        transition = Transition(jp.array([1.0, 1.0]), jp.array([1.0, 0.0]))
        args = None, Parameters(None), normalizers, state, transition, 0.99
        rewards = rollout.bootstrap(*args)
    finally:
        rollout.call_critic = original
    assert np.allclose(np.asarray(rewards), [1.0 + 0.99 * 2.0, 1.0], atol=1e-4)  # fmt: skip


FakeObservation = namedtuple("FakeObservation", "term")


def test_normalizer_merge_matches_batch_statistics():
    first = jp.reshape(jp.arange(24.0), (2, 3, 2, 2))
    second = first * 3.0 + 1.0
    normalizer = rollout.build_normalizer(FakeObservation(first[0]))
    normalizer = rollout.update_normalizer(normalizer, FakeObservation(first))
    normalizer = rollout.update_normalizer(normalizer, FakeObservation(second))  # fmt: skip
    stacked = np.concatenate([np.asarray(first), np.asarray(second)])
    flat = stacked.reshape(-1, 2)
    assert np.allclose(np.asarray(normalizer.term.mean), flat.mean(0), atol=1e-3)  # fmt: skip
    assert np.allclose(np.asarray(normalizer.term.variance), flat.var(0), rtol=1e-2)  # fmt: skip


def test_normalize_is_identity_before_any_update():
    observation = FakeObservation(jp.array([[3.0, -2.0]]))
    normalizer = rollout.build_normalizer(observation)
    normalized = rollout.normalize(observation, normalizer)
    assert np.allclose(np.asarray(normalized.term), [[3.0, -2.0]], atol=1e-4)


def test_normalize_standardizes_after_updates():
    values = jr.normal(jr.key(0), (100, 4, 2)) * 5.0 + 3.0
    normalizer = rollout.build_normalizer(FakeObservation(values[0]))
    normalizer = rollout.update_normalizer(normalizer, FakeObservation(values))  # fmt: skip
    normalized = rollout.normalize(FakeObservation(values), normalizer)
    flat = np.asarray(normalized.term).reshape(-1, 2)
    assert np.allclose(flat.mean(0), 0.0, atol=1e-2)
    assert np.allclose(flat.std(0), 1.0, atol=1e-2)


def test_merge_steps_flattens_time_and_environment():
    values = jp.arange(12.0).reshape(3, 2, 2)
    merged = rollout.merge_steps(values)
    assert merged.shape == (6, 2)
    assert np.allclose(np.asarray(merged[0]), [0.0, 1.0])


def build_rollout(**fields):
    values = {}
    for name in rollout.Rollout._fields:
        values[name] = jp.zeros((2, 3))
    values.update(fields)
    return rollout.Rollout(**values)


def test_compute_metrics_counts_divergences():
    diverged = jp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    done = jp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    reward_sum = jp.array([[0.0, 4.0, 0.0], [2.0, 0.0, 6.0]])
    fields = dict(diverged=diverged, done=done, reward_sum=reward_sum)
    metrics = rollout.compute_metrics(build_rollout(**fields))
    assert int(metrics.divergences) == 2
    assert np.isclose(float(metrics.episode_return), 12.0 / 3.0)
    assert np.isclose(float(metrics.episode_length), 6.0 / 3.0)
