import os
from collections import namedtuple

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax.numpy as jp
import numpy as np

import networks

Shapes = namedtuple("Shapes", "first, second")
SHAPES = Shapes((5, 3), (5, 2))


def test_actor_and_critic_shapes():
    actor, critic, stdv = networks.PPO(SHAPES, SHAPES, num_actions=4)
    observations = [jp.zeros((7, 5, 3)), jp.zeros((7, 5, 2))]
    parameters = networks.read_variables(actor)
    outputs = networks.call_actor(actor, parameters, observations)
    assert outputs.shape == (7, 4)
    parameters = networks.read_variables(critic)
    values = networks.call_critic(critic, parameters, observations)
    assert values.shape == (7,)
    assert np.allclose(np.asarray(stdv.value), 1.0)


def test_read_stdv_floors_non_positive_values():
    parameters = networks.Parameters(jp.array([0.5, -1.0]), [], [])
    values = np.asarray(networks.read_stdv(parameters))
    assert np.allclose(values, [0.5, 0.01])


def test_pack_unpack_roundtrip():
    actor, critic, stdv = networks.PPO(SHAPES, SHAPES, num_actions=4)
    parameters = networks.snapshot_parameters(actor, critic, stdv)
    packed = networks.pack_parameters(parameters)
    unpacked = networks.unpack_parameters(packed)
    assert np.allclose(np.asarray(unpacked.stdv), np.asarray(parameters.stdv))  # fmt: skip
    assert len(unpacked.actor) == len(parameters.actor)
    for one, other in zip(unpacked.critic, parameters.critic):
        assert np.allclose(np.asarray(one), np.asarray(other))


def test_optimizer_state_learning_rate_slot():
    import ppo

    actor, critic, stdv = networks.PPO(SHAPES, SHAPES, num_actions=4)
    _, optimizer_state = networks.Optimizer(actor, critic, stdv, 1e-3)
    assert np.isclose(float(optimizer_state[1]), 1e-3)
    updated = ppo.set_learning_rate(optimizer_state, 5e-4)
    assert np.isclose(float(updated[1]), 5e-4)


def test_compute_shapes_drops_batch_axis():
    observations = Shapes(jp.zeros((9, 5, 3)), jp.zeros((9, 5, 2)))
    shapes = networks.compute_shapes(observations)
    assert shapes == Shapes((5, 3), (5, 2))
