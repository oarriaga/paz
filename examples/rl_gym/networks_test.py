import os
from collections import namedtuple

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax.numpy as jp
import numpy as np

import networks

Shapes = namedtuple("Shapes", "first, second")


def build_models():
    actor_shapes = Shapes((5, 3), (5, 2))
    critic_shapes = Shapes((5, 4), (5, 2))
    return networks.PPO(actor_shapes, critic_shapes, num_actions=4)


def test_actor_and_critic_shapes():
    actor, critic, stdv = build_models()
    parameters = networks.snapshot_parameters(actor, critic, stdv)
    observation = Shapes(jp.zeros((7, 5, 3)), jp.zeros((7, 5, 2)))
    assert networks.call_actor(actor, parameters.actor, observation).shape == (7, 4)  # fmt: skip
    critic_observation = Shapes(jp.zeros((7, 5, 4)), jp.zeros((7, 5, 2)))
    assert networks.call_critic(critic, parameters.critic, critic_observation).shape == (7,)  # fmt: skip


def test_read_stdv_floors_non_positive_values():
    parameters = networks.Parameters(jp.array([0.5, 0.0, -1.0]), None, None)
    assert np.allclose(np.asarray(networks.read_stdv(parameters)), [0.5, 0.01, 0.01])  # fmt: skip


def test_pack_unpack_roundtrip():
    actor, critic, stdv = build_models()
    parameters = networks.snapshot_parameters(actor, critic, stdv)
    unpacked = networks.unpack_parameters(networks.pack_parameters(parameters))
    assert len(unpacked.actor) == len(parameters.actor)
    assert len(unpacked.critic) == len(parameters.critic)
    for original, restored in zip(parameters.actor, unpacked.actor):
        assert np.allclose(np.asarray(original), np.asarray(restored))


def test_optimizer_state_learning_rate_slot():
    actor, critic, stdv = build_models()
    _, optimizer_state = networks.Optimizer(actor, critic, stdv, 3e-4)
    assert np.isclose(float(optimizer_state[1]), 3e-4)


def test_compute_shapes_drops_batch_axis():
    observation = Shapes(jp.zeros((7, 5, 3)), jp.zeros((7, 5, 2)))
    assert networks.compute_shapes(observation) == Shapes((5, 3), (5, 2))
