import os
from collections import namedtuple

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax.numpy as jp
import numpy as np

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
        return observation

    original = rollout.call_critic
    rollout.call_critic = critic_call
    try:
        state = State(History(jp.array([2.0, 3.0])))
        transition = Transition(jp.array([1.0, 1.0]), jp.array([1.0, 0.0]))
        args = None, Parameters(None), state, transition, 0.99
        rewards = rollout.bootstrap(*args)
    finally:
        rollout.call_critic = original
    assert np.allclose(np.asarray(rewards), [1.0 + 0.99 * 2.0, 1.0], atol=1e-4)  # fmt: skip


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


def test_compute_metrics_reports_episode_length_and_level():
    done = jp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    reward_sum = jp.array([[0.0, 4.0, 0.0], [2.0, 0.0, 6.0]])
    level = jp.array([[0, 2, 4], [0, 2, 4]])
    tracking_sum = jp.array([[0.0, 800.0, 0.0], [1000.0, 0.0, 600.0]])
    fields = dict(done=done, reward_sum=reward_sum, tracking_sum=tracking_sum, level=level)  # fmt: skip
    metrics = rollout.compute_metrics(build_rollout(**fields))
    assert np.isclose(float(metrics.episode_return), 12.0 / 3.0)
    assert np.isclose(float(metrics.episode_length), 6.0 / 3.0)
    assert np.isclose(float(metrics.level), 2.0)
    # the episodic tracking sums of 1000-step episodes, rated over 20 s
    assert np.isclose(float(metrics.episodic_tracking), 0.8)


def test_bootstrap_keeps_a_diverged_value_out_of_the_reward():
    Parameters = namedtuple("Parameters", "critic")
    History = namedtuple("History", "critic")
    State = namedtuple("State", "history")
    Transition = namedtuple("Transition", "reward, timeout")

    def critic_call(critic, parameters, observation):
        return observation

    original = rollout.call_critic
    rollout.call_critic = critic_call
    try:
        state = State(History(jp.array([jp.nan, 3.0])))
        transition = Transition(jp.array([0.0, 1.0]), jp.array([0.0, 1.0]))
        args = None, Parameters(None), state, transition, 0.99
        rewards = rollout.bootstrap(*args)
    finally:
        rollout.call_critic = original
    assert np.allclose(np.asarray(rewards), [0.0, 1.0 + 0.99 * 3.0], atol=1e-4)  # fmt: skip


def test_compute_metrics_counts_divergences():
    diverged = jp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    metrics = rollout.compute_metrics(build_rollout(diverged=diverged))
    assert int(metrics.divergences) == 2
