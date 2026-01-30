import numpy as np
from keras import ops

from examples.gemma3.functional.keras_hub_utils import ensure_keras_hub
from examples.gemma3.functional.gemma3 import build_gemma3_backbone
from examples.gemma3.functional.gemma3 import build_vision_encoder
from examples.gemma3.functional import gemma3 as g3
from examples.gemma3.functional.presets import load_gemma3_preset_config


def _copy_rms_norm_weights(clean_layer, hub_layer):
    hub_weights = hub_layer.get_weights()
    clean_layer.set_weights([hub_weights[0] + 1.0])


def _copy_attention_weights(clean_layers, hub_attention):
    query_dense = clean_layers[0]
    key_dense = clean_layers[1]
    value_dense = clean_layers[2]
    output_dense = clean_layers[3]
    query_norm = clean_layers[4]
    key_norm = clean_layers[5]
    query_dense.set_weights(hub_attention.query_dense.get_weights())
    key_dense.set_weights(hub_attention.key_dense.get_weights())
    value_dense.set_weights(hub_attention.value_dense.get_weights())
    output_dense.set_weights(hub_attention.output_dense.get_weights())
    if hub_attention.use_query_key_norm:
        _copy_rms_norm_weights(query_norm, hub_attention.query_norm)
        _copy_rms_norm_weights(key_norm, hub_attention.key_norm)


def _copy_decoder_block_weights(clean_block, hub_block):
    pre_norm = clean_block[0]
    post_norm = clean_block[1]
    attention_layers = clean_block[2]
    pre_ffw_norm = clean_block[4]
    post_ffw_norm = clean_block[5]
    gating_ffw = clean_block[6]
    gating_ffw_2 = clean_block[7]
    ffw_linear = clean_block[8]
    _copy_rms_norm_weights(pre_norm, hub_block.pre_attention_norm)
    if hub_block.use_post_attention_norm:
        _copy_rms_norm_weights(post_norm, hub_block.post_attention_norm)
    _copy_attention_weights(attention_layers, hub_block.attention)
    _copy_rms_norm_weights(pre_ffw_norm, hub_block.pre_ffw_norm)
    if hub_block.use_post_ffw_norm:
        _copy_rms_norm_weights(post_ffw_norm, hub_block.post_ffw_norm)
    gating_ffw.set_weights(hub_block.gating_ffw.get_weights())
    gating_ffw_2.set_weights(hub_block.gating_ffw_2.get_weights())
    ffw_linear.set_weights(hub_block.ffw_linear.get_weights())


def copy_vision_encoder_weights(clean_layers, hub_vision_encoder):
    embedding_layers = clean_layers[0]
    encoder_layers = clean_layers[1]
    encoder_norm = clean_layers[2]
    output_layers = clean_layers[4]
    patch = embedding_layers[0]
    position = embedding_layers[1]

    hub_block = hub_vision_encoder.get_layer("image_encoder")
    hub_embedding = hub_block.vision_embeddings
    patch.set_weights(hub_embedding.patch_embedding.get_weights())
    position.set_weights(hub_embedding.position_embedding.get_weights())

    hub_layers = hub_block.resblocks
    for clean_layer, hub_layer in zip(encoder_layers, hub_layers):
        attention_layers = clean_layer[0]
        query_proj = attention_layers[0]
        key_proj = attention_layers[1]
        value_proj = attention_layers[2]
        out_proj = attention_layers[3]
        query_proj.set_weights(hub_layer.attn.query_proj.get_weights())
        key_proj.set_weights(hub_layer.attn.key_proj.get_weights())
        value_proj.set_weights(hub_layer.attn.value_proj.get_weights())
        out_proj.set_weights(hub_layer.attn.out_proj.get_weights())
        layer_norm_1 = clean_layer[1]
        layer_norm_2 = clean_layer[2]
        mlp_dense_1 = clean_layer[3]
        mlp_dense_2 = clean_layer[4]
        layer_norm_1.set_weights(hub_layer.layer_norm_1.get_weights())
        layer_norm_2.set_weights(hub_layer.layer_norm_2.get_weights())
        mlp_dense_1.set_weights(hub_layer.mlp_dense_1.get_weights())
        mlp_dense_2.set_weights(hub_layer.mlp_dense_2.get_weights())

    encoder_norm.set_weights(hub_block.encoder_layer_norm.get_weights())

    hub_output = hub_vision_encoder.get_layer("vision_output_encoder")
    soft_norm = output_layers[0]
    input_proj = output_layers[1]
    _copy_rms_norm_weights(soft_norm, hub_output.vision_soft_embedding_norm)
    input_proj.set_weights(hub_output.vision_input_projection.get_weights())


def copy_backbone_weights(clean_layers, hub_backbone):
    token_embedding = clean_layers[0]
    decoder_blocks = clean_layers[1]
    final_norm = clean_layers[2]
    vision_layers = clean_layers[3]
    token_embedding.set_weights(hub_backbone.token_embedding.get_weights())
    hub_blocks = hub_backbone.transformer_layers
    for block, hub_block in zip(decoder_blocks, hub_blocks):
        block_layers = block[1]
        _copy_decoder_block_weights(block_layers, hub_block)
    _copy_rms_norm_weights(final_norm, hub_backbone.layer_norm)
    if vision_layers is not None and hub_backbone.vision_encoder:
        copy_vision_encoder_weights(vision_layers, hub_backbone.vision_encoder)


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
        size = vision_config["image_size"]
        patch = vision_config["patch_size"]
        pool = vision_config["pool_size"]
        num_tokens = g3.compute_num_vision_tokens_per_image(size, patch, pool)
        total_tokens = num_tokens * num_images
        if sequence_length is None:
            sequence_length = total_tokens + 1
        elif sequence_length <= total_tokens:
            sequence_length = total_tokens + 1

    vocab_size = backbone_config["vocabulary_size"]
    ids_shape = (batch_size, sequence_length)
    token_ids = rng.integers(0, vocab_size, size=ids_shape, dtype=np.int32)
    padding_mask = np.ones((batch_size, sequence_length), dtype=np.int32)
    inputs = {}
    inputs["token_ids"] = ops.convert_to_tensor(token_ids)
    inputs["padding_mask"] = ops.convert_to_tensor(padding_mask)

    if vision_config is None:
        return inputs

    image_size = vision_config["image_size"]
    image_shape = (batch_size, num_images, image_size, image_size, 3)
    images = rng.standard_normal(image_shape).astype("float32")
    size = vision_config["image_size"]
    patch = vision_config["patch_size"]
    pool = vision_config["pool_size"]
    num_tokens = g3.compute_num_vision_tokens_per_image(size, patch, pool)
    total_tokens = num_tokens * num_images
    vision_indices = np.arange(1, total_tokens + 1, dtype=np.int32)
    vision_indices = np.tile(vision_indices, (batch_size, 1))

    vision_mask = np.zeros((batch_size, sequence_length), dtype=np.int32)
    for batch_index in range(batch_size):
        vision_mask[batch_index, vision_indices[batch_index]] = 1

    inputs["images"] = ops.convert_to_tensor(images)
    inputs["vision_indices"] = ops.convert_to_tensor(vision_indices)
    inputs["vision_mask"] = ops.convert_to_tensor(vision_mask)
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
    backbone_config = preset_config["backbone_config"]
    vision_config = preset_config["vision_config"]

    vision_apply = None
    vision_layers = None
    vision_tokens = None
    if vision_config is not None:
        vision_apply, vision_layers = build_vision_encoder(
            image_size=vision_config["image_size"],
            patch_size=vision_config["patch_size"],
            num_heads=vision_config["num_heads"],
            hidden_dim=vision_config["hidden_dim"],
            num_layers=vision_config["num_layers"],
            intermediate_dim=vision_config["intermediate_dim"],
            output_dim=vision_config["output_dim"],
            pool_size=vision_config["pool_size"],
            layer_norm_epsilon=vision_config["layer_norm_epsilon"],
            dropout=vision_config["dropout"],
            dtype=dtype,
            name_prefix="vision",
        )
        size = vision_config["image_size"]
        patch = vision_config["patch_size"]
        pool = vision_config["pool_size"]
        compute_tokens = g3.compute_num_vision_tokens_per_image
        vision_tokens = compute_tokens(size, patch, pool)

    apply_backbone, layers = build_gemma3_backbone(
        vocabulary_size=backbone_config["vocabulary_size"],
        image_size=backbone_config["image_size"],
        num_layers=backbone_config["num_layers"],
        num_query_heads=backbone_config["num_query_heads"],
        num_key_value_heads=backbone_config["num_key_value_heads"],
        hidden_dim=backbone_config["hidden_dim"],
        intermediate_dim=backbone_config["intermediate_dim"],
        head_dim=backbone_config["head_dim"],
        query_head_dim_normalize=backbone_config["query_head_dim_normalize"],
        use_query_key_norm=backbone_config["use_query_key_norm"],
        use_post_ffw_norm=backbone_config["use_post_ffw_norm"],
        use_post_attention_norm=backbone_config["use_post_attention_norm"],
        attention_logit_soft_cap=backbone_config["attention_logit_soft_cap"],
        use_sliding_window_attention=backbone_config["use_sliding_window_attention"],
        sliding_window_size=backbone_config["sliding_window_size"],
        local_rope_scaling_factor=backbone_config["local_rope_scaling_factor"],
        global_rope_scaling_factor=backbone_config["global_rope_scaling_factor"],
        use_bidirectional_attention=backbone_config["use_bidirectional_attention"],
        layer_norm_epsilon=backbone_config["layer_norm_epsilon"],
        dropout=backbone_config["dropout"],
        dtype=dtype,
        name_prefix="gemma3",
        vision_apply=vision_apply,
        vision_layers=vision_layers,
        vision_num_tokens=vision_tokens,
    )

    inputs = build_dummy_inputs(
        backbone_config,
        vision_config=vision_config,
        batch_size=batch_size,
        num_images=num_images,
        sequence_length=sequence_length,
    )
    token_ids = inputs["token_ids"]
    padding_mask = inputs["padding_mask"]
    images = inputs.get("images")
    vision_ids = inputs.get("vision_indices")
    vision_mask = inputs.get("vision_mask")
    tokens = token_ids
    padding = padding_mask
    images_data = images
    ids = vision_ids
    mask = vision_mask
    apply_backbone(tokens, padding, images_data, ids, mask, False)

    hub_model = None
    if load_weights:
        ensure_keras_hub()
        from keras_hub.src.models.gemma3 import gemma3_backbone

        hub_class = gemma3_backbone.Gemma3Backbone
        hub_model = hub_class.from_preset(preset, dtype=dtype)
        copy_backbone_weights(layers, hub_model)

    result = {}
    result["backbone_config"] = backbone_config
    result["vision_config"] = vision_config
    result["apply_backbone"] = apply_backbone
    result["layers"] = layers
    result["hub_model"] = hub_model if return_hub_model else None
    return result


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
        preset=preset,
        dtype=dtype,
        load_weights=True,
        sequence_length=sequence_length,
        num_images=num_images,
        batch_size=batch_size,
        return_hub_model=True,
    )
    hub_model = result["hub_model"]
    if hub_model is None:
        raise ValueError("Hub model weights were not loaded.")

    inputs = build_dummy_inputs(
        result["backbone_config"],
        vision_config=result["vision_config"],
        batch_size=batch_size,
        num_images=num_images,
        sequence_length=sequence_length,
    )
    hub_output = hub_model(inputs)
    token_ids = inputs["token_ids"]
    padding_mask = inputs["padding_mask"]
    images = inputs.get("images")
    vision_ids = inputs.get("vision_indices")
    vision_mask = inputs.get("vision_mask")
    apply_backbone = result["apply_backbone"]
    args = token_ids, padding_mask, images, vision_ids, vision_mask, False
    clean_output = apply_backbone(*args)
    difference = ops.abs(clean_output - hub_output)
    max_diff = float(ops.convert_to_numpy(ops.max(difference)))
    threshold = atol + rtol * ops.abs(hub_output)
    passed = bool(ops.convert_to_numpy(ops.all(difference <= threshold)))
    result = {}
    result["max_abs_diff"] = max_diff
    result["passed"] = passed
    return result
