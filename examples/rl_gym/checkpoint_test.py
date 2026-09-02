import os
from collections import namedtuple

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax.numpy as jp
import keras
import numpy as np
import pytest

import checkpoint
import networks

Shapes = namedtuple("Shapes", "first, second")
SHAPES = Shapes((5, 3), (5, 2))


def test_saved_actor_reproduces_the_training_parameters(tmp_path):
    actor, critic, stdv = networks.PPO(SHAPES, SHAPES, num_actions=4)
    parameters = networks.snapshot_parameters(actor, critic, stdv)
    shifted = [value + 0.5 for value in parameters.actor]
    parameters = parameters._replace(actor=shifted, stdv=jp.full(4, 0.3))
    checkpoint.save(tmp_path, 7, actor, critic, parameters)
    assert checkpoint.find_latest_iteration(tmp_path) == 7
    loaded = keras.models.load_model(tmp_path / "actor_000007.keras")
    inputs = [jp.ones((2, 5, 3)), jp.ones((2, 5, 2))]
    expected = networks.call_actor(actor, shifted, Shapes(*inputs))
    assert np.allclose(np.asarray(loaded(inputs)), np.asarray(expected), atol=1e-6)  # fmt: skip
    assert np.allclose(np.load(tmp_path / "stdv_000007.npy"), 0.3)


def test_find_latest_iteration_reports_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        checkpoint.find_latest_iteration(tmp_path)
