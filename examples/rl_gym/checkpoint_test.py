import os
from collections import namedtuple

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax.numpy as jp
import numpy as np

import checkpoint
import networks
import ppo
import rollout

Shapes = namedtuple("Shapes", "first, second")
SHAPES = Shapes((5, 3), (5, 2))
History = namedtuple("History", "actor, critic")


def build_normalizers():
    observation = Shapes(jp.zeros((5, 3)), jp.zeros((5, 2)))
    history = History(observation, observation)
    return rollout.build_normalizers(history)


def test_save_and_load_roundtrip(tmp_path):
    actor, critic, stdv = networks.PPO(SHAPES, SHAPES, num_actions=4)
    optimizer_args = actor, critic, stdv, 1e-3
    optimizer, optimizer_state = networks.Optimizer(*optimizer_args)
    stdv.assign(jp.full(4, -0.5))
    parameters = networks.snapshot_parameters(actor, critic, stdv)
    learning_rate = jp.asarray(3e-4)
    training = ppo.TrainingState(parameters, optimizer_state, learning_rate)
    normalizers = build_normalizers()
    normalizers = normalizers._replace(actor=rollout.update_normalizer(normalizers.actor, Shapes(jp.ones((4, 5, 3)), jp.ones((4, 5, 2)))))  # fmt: skip
    save_args = tmp_path, 7, actor, critic, training, jp.asarray(0.4)
    checkpoint.save(*save_args, normalizers)
    stdv.assign(jp.zeros(4))
    loaded = checkpoint.load(tmp_path, actor, critic)
    assert loaded.iteration == 7
    assert np.isclose(loaded.learning_rate, 3e-4)
    assert np.isclose(loaded.max_speed, 0.4)
    assert np.allclose(loaded.stdv, -0.5)
    assert len(loaded.optimizer_state) == len(optimizer_state)
    restored = checkpoint.restore_normalizers(loaded, build_normalizers())
    original_mean = np.asarray(normalizers.actor.first.mean)
    assert np.allclose(np.asarray(restored.actor.first.mean), original_mean)
    parameters_back = networks.snapshot_parameters(actor, critic, stdv)
    for one, other in zip(parameters_back.actor, parameters.actor):
        assert np.allclose(np.asarray(one), np.asarray(other))


def test_find_latest_iteration(tmp_path):
    for iteration in (2, 10, 4):
        (tmp_path / f"training_{iteration:06d}.npz").touch()
    assert checkpoint.find_latest_iteration(tmp_path) == 10


def test_find_latest_iteration_reports_empty_directory(tmp_path):
    try:
        checkpoint.find_latest_iteration(tmp_path)
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised
