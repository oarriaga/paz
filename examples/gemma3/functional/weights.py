from collections import namedtuple

import numpy as np
from keras import ops

from examples.gemma3.functional.keras_hub_utils import ensure_keras_hub
from examples.gemma3.functional.model import apply_gemma3_backbone
from examples.gemma3.functional.model import build_backbone_layers
from examples.gemma3.functional.presets import load_gemma3_preset_config
from examples.gemma3.functional.vision import compute_num_vision_tokens_per_image


Gemma3BackboneLoadResult = namedtuple(
    "Gemma3BackboneLoadResult",
    ["backbone_config", "vision_config", "layers", "hub_model"],
)


def _copy_rms_norm_weights(clean_layer, hub_layer):
    hub_weights = hub_layer.get_weights()
    clean_layer.set_weights([hub_weights[0] + 1.0])


def _copy_attention_weights(clean_layers, hub_attention):
    clean_layers.query_dense.set_weights(hub_attention.query_dense.get_weights())
    clean_layers.key_dense.set_weights(hub_attention.key_dense.get_weights())
    clean_layers.value_dense.set_weights(hub_attention.value_dense.get_weights())
    clean_layers.output_dense.set_weights(hub_attention.output_dense.get_weights())
    if hub_attention.use_query_key_norm:
        _copy_rms_norm_weights(clean_layers.query_norm, hub_attention.query_norm)
        _copy_rms_norm_weights(clean_layers.key_norm, hub_attention.key_norm)


def _copy_decoder_block_weights(clean_block, hub_block):
    _copy_rms_norm_weights(
        clean_block.layers.pre_attention_norm, hub_block.pre_attention_norm
    )
    if hub_block.use_post_attention_norm:
        _copy_rms_norm_weights(
            clean_block.layers.post_attention_norm,
            hub_block.post_attention_norm,
        )
    _copy_attention_weights(clean_block.layers.attention_layers, hub_block.attention)
    _copy_rms_norm_weights(clean_block.layers.pre_ffw_norm, hub_block.pre_ffw_norm)
    if hub_block.use_post_ffw_norm:
        _copy_rms_norm_weights(
            clean_block.layers.post_ffw_norm,
            hub_block.post_ffw_norm,
        )
    clean_block.layers.gating_ffw.set_weights(hub_block.gating_ffw.get_weights())
    clean_block.layers.gating_ffw_2.set_weights(
        hub_block.gating_ffw_2.get_weights()
    )
    clean_block.layers.ffw_linear.set_weights(hub_block.ffw_linear.get_weights())


def copy_vision_encoder_weights(clean_layers, hub_vision_encoder):
    hub_block = hub_vision_encoder.get_layer("image_encoder")
    hub_embedding = hub_block.vision_embeddings
    clean_layers.embedding_layers.patch_embedding.set_weights(
        hub_embedding.patch_embedding.get_weights()
    )
    clean_layers.embedding_layers.position_embedding.set_weights(
        hub_embedding.position_embedding.get_weights()
    )
    for clean_layer, hub_layer in zip(
        clean_layers.encoder_layers, hub_block.resblocks
    ):
        clean_layer.attention_layers.query_proj.set_weights(
            hub_layer.attn.query_proj.get_weights()
        )
        clean_layer.attention_layers.key_proj.set_weights(
            hub_layer.attn.key_proj.get_weights()
        )
        clean_layer.attention_layers.value_proj.set_weights(
            hub_layer.attn.value_proj.get_weights()
        )
        clean_layer.attention_layers.out_proj.set_weights(
            hub_layer.attn.out_proj.get_weights()
        )
        clean_layer.layer_norm_1.set_weights(hub_layer.layer_norm_1.get_weights())
        clean_layer.layer_norm_2.set_weights(hub_layer.layer_norm_2.get_weights())
        clean_layer.mlp_dense_1.set_weights(hub_layer.mlp_dense_1.get_weights())
        clean_layer.mlp_dense_2.set_weights(hub_layer.mlp_dense_2.get_weights())

    clean_layers.encoder_layer_norm.set_weights(
        hub_block.encoder_layer_norm.get_weights()
    )

    hub_output_layer = hub_vision_encoder.get_layer("vision_output_encoder")
    _copy_rms_norm_weights(
        clean_layers.output_layers.soft_embedding_norm,
        hub_output_layer.vision_soft_embedding_norm,
    )
    clean_layers.output_layers.input_projection.set_weights(
        hub_output_layer.vision_input_projection.get_weights()
    )


def copy_backbone_weights(clean_layers, hub_backbone):
    clean_layers.token_embedding.embedding.set_weights(
        hub_backbone.token_embedding.get_weights()
    )
    for clean_block, hub_block in zip(
        clean_layers.decoder_blocks, hub_backbone.transformer_layers
    ):
        _copy_decoder_block_weights(clean_block, hub_block)
    _copy_rms_norm_weights(clean_layers.final_norm, hub_backbone.layer_norm)
    if clean_layers.vision_layers is not None and hub_backbone.vision_encoder:
        copy_vision_encoder_weights(
            clean_layers.vision_layers, hub_backbone.vision_encoder
        )


def build_dummy_inputs(
    backbone_config,
    vision_config=None,
    batch_size=1,
    num_images=1,
    sequence_length=None,
    seed=0,
):
    rng = np.random.default_rng(seed)
    if vision_config is None:
        if sequence_length is None:
            sequence_length = 8
    else:
        num_vision_tokens_per_image = compute_num_vision_tokens_per_image(
            vision_config
        )
        total_vision_tokens = num_vision_tokens_per_image * num_images
        if sequence_length is None:
            sequence_length = total_vision_tokens + 1
        elif sequence_length <= total_vision_tokens:
            sequence_length = total_vision_tokens + 1

    token_ids = rng.integers(
        0,
        backbone_config.vocabulary_size,
        size=(batch_size, sequence_length),
        dtype=np.int32,
    )
    padding_mask = np.ones((batch_size, sequence_length), dtype=np.int32)
    inputs = {
        "token_ids": ops.convert_to_tensor(token_ids),
        "padding_mask": ops.convert_to_tensor(padding_mask),
    }

    if vision_config is None:
        return inputs

    images = rng.standard_normal(
        (
            batch_size,
            num_images,
            vision_config.image_size,
            vision_config.image_size,
            3,
        )
    ).astype("float32")
    num_vision_tokens_per_image = compute_num_vision_tokens_per_image(vision_config)
    total_vision_tokens = num_vision_tokens_per_image * num_images
    vision_indices = np.arange(1, total_vision_tokens + 1, dtype=np.int32)
    vision_indices = np.tile(vision_indices, (batch_size, 1))

    vision_mask = np.zeros((batch_size, sequence_length), dtype=np.int32)
    for batch_index in range(batch_size):
        vision_mask[batch_index, vision_indices[batch_index]] = 1

    inputs.update(
        {
            "images": ops.convert_to_tensor(images),
            "vision_indices": ops.convert_to_tensor(vision_indices),
            "vision_mask": ops.convert_to_tensor(vision_mask),
        }
    )
    return inputs


def load_gemma3_backbone_from_preset(
    preset,
    dtype=None,
    load_weights=True,
    sequence_length=None,
    num_images=1,
    batch_size=1,
    return_hub_model=False,
):
    preset_config = load_gemma3_preset_config(preset)
    backbone_config = preset_config.backbone_config
    vision_config = preset_config.vision_config
    layers = build_backbone_layers(
        backbone_config, vision_config=vision_config, dtype=dtype
    )
    inputs = build_dummy_inputs(
        backbone_config,
        vision_config=vision_config,
        batch_size=batch_size,
        num_images=num_images,
        sequence_length=sequence_length,
    )
    apply_gemma3_backbone(
        layers,
        backbone_config,
        inputs["token_ids"],
        inputs["padding_mask"],
        images=inputs.get("images"),
        vision_indices=inputs.get("vision_indices"),
        vision_mask=inputs.get("vision_mask"),
        training=False,
    )

    hub_model = None
    if load_weights:
        ensure_keras_hub()
        from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone

        hub_model = Gemma3Backbone.from_preset(preset, dtype=dtype)
        copy_backbone_weights(layers, hub_model)

    if return_hub_model:
        return Gemma3BackboneLoadResult(
            backbone_config, vision_config, layers, hub_model
        )
    return Gemma3BackboneLoadResult(backbone_config, vision_config, layers, None)


def verify_gemma3_preset_parity(
    preset,
    dtype=None,
    sequence_length=None,
    num_images=1,
    batch_size=1,
    rtol=1e-5,
    atol=1e-5,
):
    result = load_gemma3_backbone_from_preset(
        preset,
        dtype=dtype,
        load_weights=True,
        sequence_length=sequence_length,
        num_images=num_images,
        batch_size=batch_size,
        return_hub_model=True,
    )
    if result.hub_model is None:
        raise ValueError("Hub model weights were not loaded.")

    inputs = build_dummy_inputs(
        result.backbone_config,
        vision_config=result.vision_config,
        batch_size=batch_size,
        num_images=num_images,
        sequence_length=sequence_length,
    )
    hub_output = result.hub_model(inputs)
    clean_output = apply_gemma3_backbone(
        result.layers,
        result.backbone_config,
        inputs["token_ids"],
        inputs["padding_mask"],
        images=inputs.get("images"),
        vision_indices=inputs.get("vision_indices"),
        vision_mask=inputs.get("vision_mask"),
        training=False,
    )
    difference = ops.abs(clean_output - hub_output)
    max_diff = float(ops.convert_to_numpy(ops.max(difference)))
    threshold = atol + rtol * ops.abs(hub_output)
    passed = bool(ops.convert_to_numpy(ops.all(difference <= threshold)))
    return {
        "max_abs_diff": max_diff,
        "passed": passed,
    }
