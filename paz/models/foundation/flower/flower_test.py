import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest

from paz.models.foundation.flower import losses, sampling
from paz.models.foundation.flower.configuration import FlowerArgs, to_config
from paz.models.foundation.flower.model import build

TINY = FlowerArgs(
    context_dim=16, hidden_dim=32, num_layers=2, num_heads=4, head_dim=8,
    mlp_dim=24, adaln_dim=8, num_shared_signals=9, action_dim=3,
    num_actions=5, action_space="eef_delta", rope_wavelength=32.0,
    rope_max_positions=20, sinusoidal_dim=8, flow_time_max_period=10_000.0,
    flow_time_frequency_scale=1000.0, frequency_max_period=1000.0,
    control_frequency=3.0, num_sampling_steps=4)


def build_tiny_inputs(seed=0):
    rng = np.random.default_rng(seed)
    context = rng.normal(size=(2, 7, TINY.context_dim)).astype("float32")
    context_mask = np.ones((2, 7), "float32")
    shape = (2, TINY.num_actions, TINY.action_dim)
    actions = rng.normal(size=shape).astype("float32")
    flow_time = np.full((2,), 0.5, "float32")
    return [context, context_mask, actions, flow_time]


def test_velocity_shape():
    model = build(TINY)
    velocity = model(build_tiny_inputs())
    assert velocity.shape == (2, TINY.num_actions, TINY.action_dim)


def test_deterministic_outputs():
    model = build(TINY)
    inputs = build_tiny_inputs()
    first = np.asarray(model(inputs))
    second = np.asarray(model(inputs))
    assert np.array_equal(first, second)


def test_causal_self_attention():
    model = build(TINY)
    inputs = build_tiny_inputs()
    reference = np.asarray(model(inputs))
    perturbed = [array.copy() for array in inputs]
    perturbed[2][:, -1, :] = perturbed[2][:, -1, :] + 10.0
    output = np.asarray(model(perturbed))
    assert np.allclose(output[:, :-1], reference[:, :-1], atol=1e-5)
    assert not np.allclose(output[:, -1], reference[:, -1], atol=1e-3)


def test_interpolation_endpoints():
    rng = np.random.default_rng(1)
    actions = rng.normal(size=(2, 5, 3)).astype("float32")
    noise = rng.normal(size=(2, 5, 3)).astype("float32")
    at_data = losses.interpolate_actions(actions, noise, np.zeros(2, "f"))
    at_noise = losses.interpolate_actions(actions, noise, np.ones(2, "f"))
    assert np.allclose(np.asarray(at_data), actions)
    assert np.allclose(np.asarray(at_noise), noise)


def test_target_velocity_moves_noise_to_actions():
    rng = np.random.default_rng(2)
    actions = rng.normal(size=(2, 5, 3)).astype("float32")
    noise = rng.normal(size=(2, 5, 3)).astype("float32")
    velocity = np.asarray(losses.build_target_velocity(actions, noise))
    assert np.allclose(noise - velocity, actions, atol=1e-6)


def test_rectified_flow_loss_zero_at_target():
    rng = np.random.default_rng(3)
    target = rng.normal(size=(2, 5, 3)).astype("float32")
    mask = np.ones((2, 5), "float32")
    loss = losses.rectified_flow_loss(target, target, mask)
    assert np.allclose(np.asarray(loss), 0.0)


def test_rectified_flow_loss_masks_positions():
    velocity = np.zeros((1, 2, 3), "float32")
    target = np.ones((1, 2, 3), "float32")
    target[0, 1] = 3.0
    mask = np.array([[1.0, 0.0]], "float32")
    loss = losses.rectified_flow_loss(velocity, target, mask)
    assert np.allclose(np.asarray(loss), 1.0)


def test_euler_sampler_integrates_constant_velocity():
    rng = np.random.default_rng(4)
    context = rng.normal(size=(1, 7, TINY.context_dim)).astype("float32")
    noise = rng.normal(size=(1, 5, 3)).astype("float32")

    def constant_velocity(inputs, training=False):
        return np.ones_like(inputs[2])

    chunk = sampling.sample_actions(constant_velocity, context, noise, 4)
    expected = np.clip(noise - 1.0, -1.0, 1.0)
    assert np.allclose(np.asarray(chunk), expected, atol=1e-6)


def missing_weight_variables():
    return not (os.environ.get("FLOWER_WEIGHTS_TEST") == "1"
                and os.environ.get("FLOWER_WEIGHTS")
                and os.environ.get("FLOWER_FIXTURES"))


@pytest.mark.skipif(missing_weight_variables(),
                    reason="set FLOWER_WEIGHTS_TEST=1, FLOWER_WEIGHTS and "
                           "FLOWER_FIXTURES to run checkpoint parity")
def test_checkpoint_parity_against_torch_fixtures():
    config = to_config("flower_libero_object")
    model = build(config)
    model.load_weights(os.environ["FLOWER_WEIGHTS"])
    fixtures = np.load(os.environ["FLOWER_FIXTURES"])
    context = fixtures["encoder_hidden_states"]
    context_mask = np.ones(context.shape[:2], "float32")
    noise = fixtures["noise_action_chunk"]
    flow_time = np.full((1,), 0.5, "float32")
    velocity = model([context, context_mask, noise, flow_time])
    error = np.abs(np.asarray(velocity) - fixtures["velocity_t0_5"]).max()
    assert error < 1e-4
    chunk = sampling.sample_actions(model, context, noise, 4)
    chunk_error = np.abs(
        np.asarray(chunk) - fixtures["final_action_chunk"]).max()
    assert chunk_error < 1e-4
