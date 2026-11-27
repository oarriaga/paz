import numpy as np

from paz.models.foundation.sonic.layout import ObservationSpan
from paz.models.foundation.sonic.layout import compute_encoder_input_dim
from paz.models.foundation.sonic.layout import compute_policy_tail_dim
from paz.models.foundation.sonic.model import build_actor
from paz.models.foundation.sonic.model import build_decoder
from paz.models.foundation.sonic.model import build_encoder
from paz.models.foundation.sonic.model import compute_release_fsq
from paz.models.foundation.sonic.model import compute_temporal_part


def test_actor_decodes_encoder_tokens_followed_by_policy_tail(toy_layout):
    encoder = build_encoder(toy_layout)
    decoder = build_decoder(toy_layout)
    actor = build_actor(toy_layout, encoder, decoder)
    random = np.random.default_rng(0)
    encoder_dim = compute_encoder_input_dim(toy_layout)
    tail_dim = compute_policy_tail_dim(toy_layout)
    encoder_obs = random.normal(size=(1, encoder_dim)).astype("float32")
    encoder_obs[0, 0] = 0
    tail = random.normal(size=(1, tail_dim)).astype("float32")
    inputs = {"encoder_obs": encoder_obs, "policy_obs_tail": tail}
    action = np.array(actor(inputs, training=False))
    tokens = np.array(encoder(encoder_obs, training=False))
    decoder_input = np.concatenate([tokens, tail], axis=-1)
    expected = np.array(decoder(decoder_input, training=False))
    assert action.shape == (1, toy_layout.action_dim)
    assert np.allclose(action, expected, atol=1e-6)


def test_mode_routing_selects_matching_branch(toy_layout):
    encoder = build_encoder(toy_layout)
    x_flat = np.random.default_rng(0).normal(
        size=(1, compute_encoder_input_dim(toy_layout))).astype("float32")
    x_temporal = x_flat.copy()
    x_flat[0, 0] = 0
    x_temporal[0, 0] = 1
    tokens_flat = np.array(encoder(x_flat, training=False))
    tokens_temporal = np.array(encoder(x_temporal, training=False))
    assert not np.allclose(tokens_flat, tokens_temporal)


def test_fsq_output_lies_on_quantization_grid():
    inputs = np.random.default_rng(1).normal(size=(2, 8)).astype("float32")
    output = np.array(
        compute_release_fsq(inputs, 2, 4, 0.032237, 15.515501, 0.5, 16.0))
    scaled = output * 16.0
    assert np.allclose(scaled, np.round(scaled))


def test_temporal_part_concatenates_before_reshaping():
    # Pins compute_temporal_part's concatenate-then-reshape order for a
    # multi-span group: each span's values stay contiguous across frames
    # rather than interleaving per timestep. Matches the released model's
    # training-time layout (e.g. the real "g1" position+velocity group).
    span_a = ObservationSpan("a", 0, 4, 4)
    span_b = ObservationSpan("b", 4, 8, 4)
    encoder_input = np.array([[1, 2, 3, 4, 10, 20, 30, 40]], dtype="float32")
    output = np.array(
        compute_temporal_part(encoder_input, (span_a, span_b), 2))
    expected = np.array([[[1, 2, 3, 4], [10, 20, 30, 40]]], dtype="float32")
    assert np.array_equal(output, expected)
