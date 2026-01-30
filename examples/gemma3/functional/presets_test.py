from pathlib import Path

import numpy as np
from keras import ops

from examples.gemma3.functional.keras_hub_utils import ensure_keras_hub
from examples.gemma3.functional.presets import load_gemma3_preset_config
from examples.gemma3.functional.weights import build_dummy_inputs
from examples.gemma3.functional.weights import load_gemma3_backbone_from_preset

ensure_keras_hub()

from keras_hub.src.models.gemma3 import gemma3_backbone
from keras_hub.src.models.gemma3 import gemma3_vision_encoder


def _build_text_backbone():
    return gemma3_backbone.Gemma3Backbone(
        vocabulary_size=32,
        image_size=8,
        num_layers=1,
        num_query_heads=1,
        num_key_value_heads=1,
        hidden_dim=4,
        intermediate_dim=8,
        head_dim=4,
        query_head_dim_normalize=True,
        use_query_key_norm=True,
        use_post_ffw_norm=False,
        use_post_attention_norm=False,
        attention_logit_soft_cap=None,
        final_logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=8,
        vision_encoder=None,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
        dtype="float32",
        name="hub_text",
    )


def _build_vision_backbone():
    vision_encoder = gemma3_vision_encoder.Gemma3VisionEncoder(
        image_size=8,
        patch_size=4,
        num_heads=1,
        hidden_dim=4,
        num_layers=1,
        intermediate_dim=8,
        output_dim=4,
        pool_size=2,
        layer_norm_epsilon=1e-6,
        dtype="float32",
        name="hub_vision",
    )

    return gemma3_backbone.Gemma3Backbone(
        vocabulary_size=32,
        image_size=8,
        num_layers=1,
        num_query_heads=1,
        num_key_value_heads=1,
        hidden_dim=4,
        intermediate_dim=8,
        head_dim=4,
        query_head_dim_normalize=True,
        use_query_key_norm=True,
        use_post_ffw_norm=False,
        use_post_attention_norm=False,
        attention_logit_soft_cap=None,
        final_logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=8,
        vision_encoder=vision_encoder,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
        dtype="float32",
        name="hub_vision_backbone",
    )


def test_load_preset_config_from_local_text(tmp_path):
    preset_dir = Path(tmp_path) / "text_preset"
    backbone = _build_text_backbone()
    backbone.save_to_preset(str(preset_dir))

    preset_config = load_gemma3_preset_config(str(preset_dir))
    backbone_config = preset_config["backbone_config"]
    assert backbone_config["hidden_dim"] == backbone.hidden_dim
    assert backbone_config["num_layers"] == backbone.num_layers
    assert preset_config["vision_config"] is None


def test_load_preset_config_from_local_vision(tmp_path):
    preset_dir = Path(tmp_path) / "vision_preset"
    backbone = _build_vision_backbone()
    backbone.save_to_preset(str(preset_dir))

    preset_config = load_gemma3_preset_config(str(preset_dir))
    backbone_config = preset_config["backbone_config"]
    vision_config = preset_config["vision_config"]
    assert backbone_config["hidden_dim"] == backbone.hidden_dim
    assert vision_config is not None
    assert vision_config["output_dim"] == 4


def test_load_weights_from_local_text_preset(tmp_path):
    preset_dir = Path(tmp_path) / "text_weights"
    backbone = _build_text_backbone()
    backbone.save_to_preset(str(preset_dir))

    result = load_gemma3_backbone_from_preset(
        preset=str(preset_dir),
        dtype="float32",
        load_weights=True,
        sequence_length=6,
        batch_size=1,
        return_hub_model=True,
    )

    inputs = build_dummy_inputs(
        result["backbone_config"], batch_size=1, sequence_length=6
    )
    hub_output = result["hub_model"](inputs)
    token_ids = inputs["token_ids"]
    padding_mask = inputs["padding_mask"]
    apply_backbone = result["apply_backbone"]
    args = token_ids, padding_mask, None, None, None, False
    clean_output = apply_backbone(*args)
    clean_np = ops.convert_to_numpy(clean_output)
    hub_np = ops.convert_to_numpy(hub_output)
    np.testing.assert_allclose(clean_np, hub_np, rtol=1e-5, atol=1e-5)


def test_load_weights_from_local_vision_preset(tmp_path):
    preset_dir = Path(tmp_path) / "vision_weights"
    backbone = _build_vision_backbone()
    backbone.save_to_preset(str(preset_dir))

    result = load_gemma3_backbone_from_preset(
        preset=str(preset_dir),
        dtype="float32",
        load_weights=True,
        sequence_length=4,
        batch_size=1,
        num_images=1,
        return_hub_model=True,
    )

    inputs = build_dummy_inputs(
        result["backbone_config"],
        vision_config=result["vision_config"],
        batch_size=1,
        num_images=1,
        sequence_length=4,
    )
    hub_output = result["hub_model"](inputs)
    token_ids = inputs["token_ids"]
    padding_mask = inputs["padding_mask"]
    images = inputs["images"]
    vision_ids = inputs["vision_indices"]
    vision_mask = inputs["vision_mask"]
    apply_backbone = result["apply_backbone"]
    tokens = token_ids
    padding = padding_mask
    images_data = images
    ids = vision_ids
    mask = vision_mask
    args = tokens, padding, images_data, ids, mask, False
    clean_output = apply_backbone(*args)
    clean_np = ops.convert_to_numpy(clean_output)
    hub_np = ops.convert_to_numpy(hub_output)
    np.testing.assert_allclose(clean_np, hub_np, rtol=1e-5, atol=1e-5)
