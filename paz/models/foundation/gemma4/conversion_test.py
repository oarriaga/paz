import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import jax.numpy as jp
import pytest

from paz.models.foundation.gemma4.configuration import load_config
from paz.models.foundation.gemma4.conversion import (
    build_paz_config, build_target_backbone, save_paz_models, transfer)
from paz.models.foundation.gemma4.model import Gemma4Backbone

gemma4 = pytest.importorskip(
    "keras_hub.src.models.gemma4.gemma4_backbone")
KerasHubGemma4Backbone = gemma4.Gemma4Backbone


def build_reference():
    return KerasHubGemma4Backbone(
        vocabulary_size=64, image_size=16, num_layers=6, num_query_heads=2,
        num_key_value_heads=1, hidden_dim=16, intermediate_dim=32,
        head_dim=16, use_sliding_window_attention=True, sliding_window_size=4,
        sliding_window_pattern=3, global_head_dim=None,
        hidden_size_per_layer_input=2, num_kv_shared_layers=2,
        global_rope_partial_rotary_factor=0.25, use_double_wide_mlp=True,
        dtype="float32")


def build_inputs():
    rng = np.random.default_rng(0)
    tokens = rng.integers(1, 64, (2, 7)).astype("int32")
    padding = np.ones((2, 7), dtype="int32")
    padding[0, 6] = 0
    positions = np.broadcast_to(np.arange(7, dtype="int32"), (2, 7)).copy()
    return tokens, padding, positions


def test_transfer_matches_keras_hub():
    reference = build_reference()
    config = build_paz_config(reference)
    model = build_target_backbone(config)
    transfer(reference, model)
    tokens, padding, positions = build_inputs()
    expected = reference(
        {"token_ids": tokens, "padding_mask": padding,
         "position_ids": positions})
    output = model({"token_ids": tokens, "padding_mask": padding})
    diff = float(np.max(np.abs(np.array(expected) - np.array(output))))
    assert diff < 1e-4


def test_converter_writes_loadable_artifacts(tmp_path):
    reference = build_reference()
    save_paz_models(reference, tmp_path)
    config = load_config(tmp_path / "config.json")
    model = Gemma4Backbone(config)
    model({"token_ids": jp.zeros((1, 1), "int32"),
           "padding_mask": jp.ones((1, 1), "int32")})
    model.load_weights(str(tmp_path / "backbone.weights.h5"))
