import numpy as np

from paz.models.foundation.sonic.layout import ObservationSpan
from paz.models.foundation.sonic.layout import compute_decoder_input_dim
from paz.models.foundation.sonic.layout import compute_encoder_input_dim
from paz.models.foundation.sonic.layout import compute_policy_tail_dim
from paz.models.foundation.sonic.model import FSQArgs
from paz.models.foundation.sonic.model import build_sonic_actor
from paz.models.foundation.sonic.model import build_sonic_decoder
from paz.models.foundation.sonic.model import build_sonic_encoder
from paz.models.foundation.sonic.model import build_toy_layout
from paz.models.foundation.sonic.model import compute_release_fsq
from paz.models.foundation.sonic.model import compute_temporal_part


def test_encoder_output_shape_matches_token_dim():
    layout = build_toy_layout()
    encoder = build_sonic_encoder(layout)
    x = np.zeros((1, compute_encoder_input_dim(layout)), dtype="float32")
    tokens = np.array(encoder(x, training=False))
    assert tokens.shape == (1, layout.token_dim)


def test_decoder_output_shape_matches_action_dim():
    layout = build_toy_layout()
    decoder = build_sonic_decoder(layout)
    x = np.zeros((1, compute_decoder_input_dim(layout)), dtype="float32")
    action = np.array(decoder(x, training=False))
    assert action.shape == (1, layout.action_dim)


def test_actor_composes_encoder_and_decoder():
    layout = build_toy_layout()
    encoder = build_sonic_encoder(layout)
    decoder = build_sonic_decoder(layout)
    actor = build_sonic_actor(layout, encoder, decoder)
    inputs = {
        "encoder_obs": np.zeros(
            (1, compute_encoder_input_dim(layout)), dtype="float32"),
        "policy_obs_tail": np.zeros(
            (1, compute_policy_tail_dim(layout)), dtype="float32"),
    }
    action = np.array(actor(inputs, training=False))
    assert action.shape == (1, layout.action_dim)


def test_mode_routing_selects_matching_branch():
    layout = build_toy_layout()
    encoder = build_sonic_encoder(layout)
    x_flat = np.random.default_rng(0).normal(
        size=(1, compute_encoder_input_dim(layout))).astype("float32")
    x_temporal = x_flat.copy()
    x_flat[0, 0] = 0
    x_temporal[0, 0] = 1
    tokens_flat = np.array(encoder(x_flat, training=False))
    tokens_temporal = np.array(encoder(x_temporal, training=False))
    assert not np.allclose(tokens_flat, tokens_temporal)


def test_fsq_output_lies_on_quantization_grid():
    inputs = np.random.default_rng(1).normal(size=(2, 8)).astype("float32")
    fsq_args = FSQArgs(2, 4, 0.032237, 15.515501, 0.5, 16.0)
    output = np.array(compute_release_fsq(inputs, fsq_args))
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
    output = np.array(compute_temporal_part(encoder_input, (span_a, span_b), 2))
    expected = np.array([[[1, 2, 3, 4], [10, 20, 30, 40]]], dtype="float32")
    assert np.array_equal(output, expected)
