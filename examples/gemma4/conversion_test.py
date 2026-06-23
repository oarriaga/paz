import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest

from .configuration import load_config
from .conversion import build_paz_config, save_paz_models, transfer
from .inference import Gemma4DecoderStep, Gemma4PerLayerEmbeddingStep
from .model import build_text_backbone

gemma4 = pytest.importorskip(
    "keras_hub.src.models.gemma4.gemma4_backbone")
Gemma4Backbone = gemma4.Gemma4Backbone


def build_reference():
    return Gemma4Backbone(
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
    backbone = build_reference()
    config = build_paz_config(backbone)
    model = build_text_backbone(config)
    transfer(backbone, model)
    tokens, padding, positions = build_inputs()
    reference = backbone(
        {"token_ids": tokens, "padding_mask": padding,
         "position_ids": positions})
    output = model({"token_ids": tokens, "padding_mask": padding})
    diff = float(np.max(np.abs(np.array(reference) - np.array(output))))
    assert diff == 0.0


def test_converter_writes_loadable_artifacts(tmp_path):
    backbone = build_reference()
    save_paz_models(backbone, tmp_path)
    config = load_config(tmp_path / "config.json")
    Gemma4DecoderStep(config).load_weights(
        str(tmp_path / "decoder_step.weights.h5"))
    Gemma4PerLayerEmbeddingStep(config).load_weights(
        str(tmp_path / "embedding_step.weights.h5"))
