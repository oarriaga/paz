import os

os.environ.setdefault("KERAS_BACKEND", "jax")

from pathlib import Path

import jax
import numpy as np
import pytest
import torch

from policy import HIDDEN_UNITS
from policy import NUM_JOINTS
from policy import OBSERVATION_DIM
from policy import build_actor
from policy import compile_actor
from policy import find_latest_checkpoint
from policy import load_actor
from policy import order_by_run

EXPERIMENT = Path(__file__).parent / "unitree_g1_29dof_velocity_robust"

needs_checkpoint = pytest.mark.skipif(
    not list(EXPERIMENT.glob("*/model_*.pt")),
    reason="no unitree_rl_lab run directory next to this example")


def build_torch_actor(weights):
    widths = (OBSERVATION_DIM,) + HIDDEN_UNITS + (NUM_JOINTS,)
    modules = []
    for index in range(len(widths) - 1):
        linear = torch.nn.Linear(widths[index], widths[index + 1])
        linear.weight.data = weights[f"actor.{2 * index}.weight"]
        linear.bias.data = weights[f"actor.{2 * index}.bias"]
        modules.append(linear)
        modules.append(torch.nn.ELU())
    return torch.nn.Sequential(*modules[:-1])


def build_observations(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(4, OBSERVATION_DIM)).astype("float32")


def test_actor_maps_the_trained_observation_onto_one_action_per_joint():
    outputs = build_actor()(build_observations())
    assert outputs.shape == (4, NUM_JOINTS)


def test_actor_has_the_widths_the_run_configuration_asked_for():
    units = [layer.units for layer in build_actor().layers[1:]]
    assert units == list(HIDDEN_UNITS) + [NUM_JOINTS]


def test_actor_keeps_full_float32_precision_against_the_checkpoint():
    # XLA defaults float32 matmuls to TF32 on a GPU, which is worth about
    # three decimal digits of every action.
    assert jax.config.jax_default_matmul_precision == "float32"


def test_checkpoints_order_by_run_then_by_iteration():
    older = Path("2026-08-18_08-10-46/model_900.pt")
    newer = Path("2026-08-18_08-10-46/model_1000.pt")
    latest = Path("2026-08-18_08-19-08/model_100.pt")
    assert order_by_run(older) < order_by_run(newer) < order_by_run(latest)


@needs_checkpoint
def test_loaded_actor_matches_the_torch_checkpoint_it_came_from():
    checkpoint = find_latest_checkpoint(EXPERIMENT)
    weights = torch.load(checkpoint, map_location="cpu")["model_state_dict"]
    observations = build_observations()
    expected = build_torch_actor(weights)(torch.from_numpy(observations))
    outputs = np.asarray(load_actor(checkpoint)(observations))
    assert np.allclose(outputs, expected.detach().numpy(), atol=1e-4)


@needs_checkpoint
def test_compiled_actor_returns_what_the_eager_actor_returns():
    actor = load_actor(find_latest_checkpoint(EXPERIMENT))
    observations = build_observations()
    expected = np.asarray(actor(observations))
    outputs = np.asarray(compile_actor(actor)(observations))
    assert np.allclose(outputs, expected, atol=1e-4)
