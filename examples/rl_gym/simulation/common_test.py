import os
from collections import namedtuple

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax.numpy as jp
import numpy as np
from jax import random as jr

from simulation import common


class FakePhysics(namedtuple("FakePhysics", "qpos, qvel")):
    def replace(self, **kwargs):
        return self._replace(**kwargs)


def test_soft_limits_shrink_the_range_around_the_midpoint():
    limits = jp.array([-1.0, 0.0]), jp.array([1.0, 4.0])
    lower, upper = common.compute_soft_limits(limits)
    assert np.allclose(np.asarray(lower), [-0.9, 0.2])
    assert np.allclose(np.asarray(upper), [0.9, 3.8])


def test_scheduled_push_adds_to_the_current_velocity():
    qvel = jp.zeros(9).at[0].set(0.4).at[2].set(-0.2).at[5].set(0.78)
    physics_state = FakePhysics(jp.zeros(9), qvel)
    args = jr.key(0), physics_state, jp.asarray(10), jp.asarray(5)
    pushed, next_push = common.apply_scheduled_push(*args)
    kick = float(pushed.qvel[0]) - 0.4
    assert kick != 0.0 and abs(kick) <= 1.0
    assert np.isclose(float(pushed.qvel[2]), -0.2)
    assert np.isclose(float(pushed.qvel[5]), 0.78)
    assert int(next_push) > 10
    args = jr.key(0), physics_state, jp.asarray(3), jp.asarray(5)
    unpushed, same_push = common.apply_scheduled_push(*args)
    assert np.allclose(np.asarray(unpushed.qvel), np.asarray(qvel))
    assert int(same_push) == 5


def test_foot_contact_pools_the_net_force_over_three_substeps():
    addresses = jp.arange(24)
    history = jp.zeros((4, 24))
    # two opposite 3 N corner forces cancel on the left foot; the right
    # foot touches on the second-to-last substep only
    history = history.at[3, 0].set(3.0).at[3, 3].set(-3.0)
    history = history.at[2, 12].set(2.0)
    contact = common.read_foot_contact(history, addresses)
    assert np.array_equal(np.asarray(contact), [False, True])
    early = jp.zeros((4, 24)).at[0, 12].set(2.0)
    contact = common.read_foot_contact(early, addresses)
    assert not np.any(np.asarray(contact))


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


def test_update_history_appends_newest_last():
    Observation = namedtuple("Observation", "term")
    history = common.build_history(Observation(jp.zeros(2)), num_history=3)
    updated = common.update_history(history, Observation(jp.ones(2)))
    assert np.allclose(np.asarray(updated.term[-1]), 1.0)
    assert np.allclose(np.asarray(updated.term[:-1]), 0.0)


def test_detect_divergence_flags_non_finite_states_and_rewards():
    healthy = FakePhysics(jp.ones(9), jp.ones(9))
    assert not bool(common.detect_divergence(healthy, jp.asarray(0.7)))
    poisoned = FakePhysics(jp.ones(9).at[1].set(jp.nan), jp.zeros(9))
    assert bool(common.detect_divergence(poisoned, jp.asarray(0.0)))
    exploding = FakePhysics(jp.ones(9), jp.zeros(9).at[1].set(jp.inf))
    assert bool(common.detect_divergence(exploding, jp.asarray(0.0)))
    assert bool(common.detect_divergence(healthy, jp.asarray(jp.nan)))


def test_detect_divergence_flags_unreachable_rewards():
    # a joint kicked far past its limit sends the action-rate term to
    # hundreds while the state itself still looks legal
    healthy = FakePhysics(jp.ones(9), jp.ones(9))
    assert bool(common.detect_divergence(healthy, jp.asarray(-679.0)))
    assert not bool(common.detect_divergence(healthy, jp.asarray(-0.25)))
