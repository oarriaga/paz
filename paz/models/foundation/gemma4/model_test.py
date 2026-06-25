import jax.numpy as jp

from paz.models.foundation.gemma4.model import (
    build_text_backbone, build_text_backbone_args, compute_text_intermediates)


def build_test_inputs():
    token_ids = jp.array([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]], dtype=jp.int32)
    padding_mask = jp.array([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=jp.int32)
    return token_ids, padding_mask


def assert_close(left, right, tol=1e-6):
    diff = jp.max(jp.abs(left - right))
    assert float(diff) <= tol


def test_runtime_backbone_builds_successfully():
    config = build_text_backbone_args()
    model = build_text_backbone(config)
    token_ids, padding_mask = build_test_inputs()
    inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
    outputs = model(inputs)
    assert outputs.shape == (2, 5, config.hidden_dim)


def test_runtime_backbone_save_and_load(tmp_path):
    config = build_text_backbone_args()
    model = build_text_backbone(config)
    token_ids, padding_mask = build_test_inputs()
    inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
    output = model(inputs)
    path = tmp_path / "gemma4_text_backbone.weights.h5"
    model.save_weights(str(path))
    loaded = build_text_backbone(config, weights_path=path)
    assert_close(output, loaded(inputs))


def test_runtime_backbone_exposes_intermediates():
    config = build_text_backbone_args()
    model = build_text_backbone(config)
    token_ids, padding_mask = build_test_inputs()
    data = compute_text_intermediates(model, token_ids, padding_mask)
    assert data.embedding_output.shape == (2, 5, config.hidden_dim)
    assert len(data.block_outputs) == config.num_layers
    assert data.final_output.shape == (2, 5, config.hidden_dim)


def test_runtime_backbone_supports_per_layer_inputs():
    config = build_text_backbone_args(hidden_size_per_layer_input=2)
    model = build_text_backbone(config)
    token_ids, padding_mask = build_test_inputs()
    inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
    outputs = model(inputs)
    assert outputs.shape == (2, 5, config.hidden_dim)


def test_runtime_backbone_runs_global_partial_rope():
    config = build_text_backbone_args(
        num_layers=6, sliding_window_pattern=3, head_dim=16,
        global_head_dim=16, global_rope_partial_rotary_factor=0.25)
    model = build_text_backbone(config)
    token_ids, padding_mask = build_test_inputs()
    inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
    outputs = model(inputs)
    assert outputs.shape == (2, 5, config.hidden_dim)
