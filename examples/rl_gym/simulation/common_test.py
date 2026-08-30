import os
from collections import namedtuple

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax.numpy as jp
import numpy as np
from jax import random as jr

from simulation import common

FakeDynamics = namedtuple("FakeDynamics", "qpos0")


class FakePhysics(namedtuple("FakePhysics", "qpos, qvel")):
    def replace(self, **kwargs):
        return self._replace(**kwargs)


def test_sanitize_diverged_replaces_non_finite_states():
    dynamics = FakeDynamics(jp.full(9, 0.5))
    poisoned = FakePhysics(jp.ones(9).at[1].set(jp.nan), jp.zeros(9))
    sanitized, diverged = common.sanitize_diverged(poisoned, dynamics)
    assert bool(diverged)
    assert np.allclose(np.asarray(sanitized.qpos), 0.5)
    healthy = FakePhysics(jp.ones(9), jp.ones(9))
    sanitized, diverged = common.sanitize_diverged(healthy, dynamics)
    assert not bool(diverged)
    assert np.allclose(np.asarray(sanitized.qpos), 1.0)


def test_sanitize_diverged_catches_infinite_velocity():
    dynamics = FakeDynamics(jp.zeros(9))
    poisoned = FakePhysics(jp.ones(9), jp.zeros(9).at[1].set(jp.inf))
    sanitized, diverged = common.sanitize_diverged(poisoned, dynamics)
    assert bool(diverged)
    assert np.allclose(np.asarray(sanitized.qvel), 0.0)


def test_sanitize_diverged_catches_unphysical_speeds():
    dynamics = FakeDynamics(jp.zeros(9))
    exploding = FakePhysics(jp.ones(9), jp.zeros(9).at[1].set(2e3))
    sanitized, diverged = common.sanitize_diverged(exploding, dynamics)
    assert bool(diverged)
    assert np.allclose(np.asarray(sanitized.qvel), 0.0)
    fast = FakePhysics(jp.ones(9), jp.full(9, 50.0))
    sanitized, diverged = common.sanitize_diverged(fast, dynamics)
    assert not bool(diverged)


def test_sanitize_diverged_catches_runaway_joint_positions():
    dynamics = FakeDynamics(jp.zeros(9))
    qpos = jp.zeros(9).at[8].set(300.0)
    runaway = FakePhysics(qpos, jp.zeros(9))
    sanitized, diverged = common.sanitize_diverged(runaway, dynamics)
    assert bool(diverged)
    assert np.allclose(np.asarray(sanitized.qpos), 0.0)
    bent = FakePhysics(jp.zeros(9).at[8].set(2.5), jp.zeros(9))
    sanitized, diverged = common.sanitize_diverged(bent, dynamics)
    assert not bool(diverged)


def test_discard_diverged_zeroes_reward_and_terms():
    terms = jp.ones(3)
    diverged, reward, terms = common.discard_diverged(jp.asarray(True), jp.asarray(1.0), terms)  # fmt: skip
    assert bool(diverged)
    assert float(reward) == 0.0
    assert np.allclose(np.asarray(terms), 0.0)


def test_discard_diverged_catches_non_finite_reward():
    terms = jp.ones(3)
    args = jp.asarray(False), jp.asarray(jp.nan), terms
    diverged, reward, terms = common.discard_diverged(*args)
    assert bool(diverged)
    assert float(reward) == 0.0


def test_discard_diverged_catches_implausible_reward_magnitudes():
    # corrupted contact sensors can inject huge finite rewards from
    # states whose positions and velocities still look legal
    terms = jp.ones(3)
    args = jp.asarray(False), jp.asarray(-7e34), terms
    diverged, reward, terms = common.discard_diverged(*args)
    assert bool(diverged)
    assert float(reward) == 0.0
    assert np.allclose(np.asarray(terms), 0.0)


def test_discard_diverged_keeps_healthy_steps():
    terms = jp.ones(3)
    args = jp.asarray(False), jp.asarray(0.7), terms
    diverged, reward, terms = common.discard_diverged(*args)
    assert not bool(diverged)
    assert np.isclose(float(reward), 0.7)
    assert np.allclose(np.asarray(terms), 1.0)


def test_soft_limits_shrink_the_range_around_the_midpoint():
    limits = jp.array([-1.0, 0.0]), jp.array([1.0, 4.0])
    lower, upper = common.compute_soft_limits(limits)
    assert np.allclose(np.asarray(lower), [-0.9, 0.2])
    assert np.allclose(np.asarray(upper), [0.9, 3.8])


def test_scheduled_push_adds_to_the_current_velocity():
    qvel = jp.zeros(9).at[0].set(0.4)
    physics_state = FakePhysics(jp.zeros(9), qvel)
    args = jr.key(0), physics_state, jp.asarray(10), jp.asarray(5)
    pushed, next_push = common.apply_scheduled_push(*args)
    kick = float(pushed.qvel[0]) - 0.4
    assert kick != 0.0 and abs(kick) <= 0.5
    assert float(jp.abs(pushed.qvel[2])) == 0.0
    assert float(jp.abs(pushed.qvel[5])) <= 0.78
    assert int(next_push) > 10
    args = jr.key(0), physics_state, jp.asarray(3), jp.asarray(5)
    unpushed, same_push = common.apply_scheduled_push(*args)
    assert np.allclose(np.asarray(unpushed.qvel), np.asarray(qvel))
    assert int(same_push) == 5


def test_yaw_quaternion_roundtrip():
    yaw = 1.2
    quaternion = common.yaw_quaternion(yaw)
    assert np.isclose(float(common.read_yaw(quaternion)), yaw, atol=1e-5)


def test_rotate_yaw_inverse_inverts_rotate_yaw():
    quaternion = common.yaw_quaternion(0.7)
    vector = jp.array([0.3, -0.2, 0.5])
    rotated = common.rotate_yaw(quaternion, vector)
    recovered = common.rotate_yaw_inverse(quaternion, rotated)
    assert np.allclose(np.asarray(recovered), np.asarray(vector), atol=1e-5)


def test_gravity_is_unit_and_down_when_upright():
    quaternion = jp.array([1.0, 0.0, 0.0, 0.0])
    gravity = common.compute_gravity(quaternion)
    assert np.allclose(np.asarray(gravity), [0.0, 0.0, -1.0], atol=1e-6)


def test_is_fallen_thresholds():
    upright = jp.array([0.0, 0.0, -1.0])
    standing = FakePhysics(jp.array([0.0, 0.0, 0.7]), jp.zeros(3))
    assert not bool(common.is_fallen(standing, upright))
    low = FakePhysics(jp.array([0.0, 0.0, 0.1]), jp.zeros(3))
    assert bool(common.is_fallen(low, upright))
    tilted = jp.array([jp.sin(1.0), 0.0, -jp.cos(1.0)])
    assert bool(common.is_fallen(standing, tilted))


def test_sample_command_bounds():
    for seed in range(20):
        command = common.sample_command(jr.key(seed), jp.asarray(0.8))
        assert -0.5 <= float(command.forward) <= 0.8
        assert -0.3 <= float(command.sideways) <= 0.3
        assert -0.1 <= float(command.turn) <= 0.1


def test_decorrelate_counters_stays_in_range():
    State = namedtuple("State", "counters")
    push = jp.full(50, 150, jp.int32)
    counters = common.StepCounters(jp.zeros(50, jp.int32), jp.zeros(50, jp.int32), push)  # fmt: skip
    state = common.decorrelate_counters(jr.key(0), State(counters))
    episode = np.asarray(state.counters.episode)
    command = np.asarray(state.counters.command)
    assert episode.min() >= 0 and episode.max() < common.EPISODE_STEPS
    assert command.min() >= 0 and command.max() < common.COMMAND_PERIOD
    assert len(np.unique(episode)) > 10
    assert np.all(np.asarray(state.counters.push) == episode + 150)


def test_update_history_appends_newest_last():
    Observation = namedtuple("Observation", "term")
    history = common.build_history(Observation(jp.zeros(2)), num_history=3)
    updated = common.update_history(history, Observation(jp.ones(2)))
    assert np.allclose(np.asarray(updated.term[-1]), 1.0)
    assert np.allclose(np.asarray(updated.term[:-1]), 0.0)
