import numpy as np

from .model import build_text_backbone, build_text_backbone_args
from .reference import build_reference_text_backbone
from .reference import collect_reference_weight_shapes
from .reference import collect_runtime_weight_shapes
from .reference import compare_runtime_and_reference_intermediates
from .reference import copy_text_backbone_weights
from .reference import export_text_backbone_weights


def build_test_inputs():
    token_ids = np.array([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]], dtype="int32")
    padding_mask = np.array([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype="int32")
    return token_ids, padding_mask


def build_models(copy_weights=False):
    config = build_text_backbone_args()
    runtime_model = build_text_backbone(config)
    reference_model = build_reference_text_backbone(config)
    if copy_weights:
        copy_text_backbone_weights(runtime_model, reference_model)
    return config, runtime_model, reference_model


def assert_close(left, right):
    np.testing.assert_allclose(np.array(left), np.array(right), 1e-6, 1e-6)


def test_runtime_backbone_builds_successfully():
    config = build_text_backbone_args()
    model = build_text_backbone(config)
    token_ids, padding_mask = build_test_inputs()
    outputs = model({"token_ids": token_ids, "padding_mask": padding_mask})
    assert outputs.shape == (2, 5, config.hidden_dim)


def test_weight_shape_parity_matches_reference():
    _config, runtime_model, reference_model = build_models()
    runtime_shapes = collect_runtime_weight_shapes(runtime_model)
    reference_shapes = collect_reference_weight_shapes(reference_model)
    assert sorted(runtime_shapes) == sorted(reference_shapes)


def test_tensor_shape_parity_matches_reference():
    _config, runtime_model, reference_model = build_models()
    token_ids, padding_mask = build_test_inputs()
    runtime, reference = compare_runtime_and_reference_intermediates(
        runtime_model, reference_model, token_ids, padding_mask
    )
    assert runtime.embedding_output.shape == reference.embedding_output.shape
    assert runtime.final_output.shape == reference.final_output.shape
    for runtime_block, reference_block in zip(
        runtime.block_outputs, reference.block_outputs
    ):
        assert runtime_block.shape == reference_block.shape


def test_individual_block_forward_parity_matches_reference():
    _config, runtime_model, reference_model = build_models(copy_weights=True)
    token_ids, padding_mask = build_test_inputs()
    runtime, reference = compare_runtime_and_reference_intermediates(
        runtime_model, reference_model, token_ids, padding_mask
    )
    for runtime_block, reference_block in zip(
        runtime.block_outputs, reference.block_outputs
    ):
        assert_close(runtime_block, reference_block)


def test_intermediate_feature_parity_matches_reference():
    _config, runtime_model, reference_model = build_models(copy_weights=True)
    token_ids, padding_mask = build_test_inputs()
    runtime, reference = compare_runtime_and_reference_intermediates(
        runtime_model, reference_model, token_ids, padding_mask
    )
    assert_close(runtime.embedding_output, reference.embedding_output)
    assert_close(runtime.final_output, reference.final_output)


def test_full_output_parity_matches_reference_via_saved_weights(tmp_path):
    config = build_text_backbone_args()
    weights_path = tmp_path / "gemma4_text_backbone.weights.h5"
    _, reference_model = export_text_backbone_weights(config, weights_path)
    runtime_model = build_text_backbone(config, weights_path=weights_path)
    token_ids, padding_mask = build_test_inputs()
    runtime_output = runtime_model(
        {"token_ids": token_ids, "padding_mask": padding_mask}
    )
    reference_output = reference_model(
        {"token_ids": token_ids, "padding_mask": padding_mask}
    )
    assert_close(runtime_output, reference_output)
