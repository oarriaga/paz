from collections import namedtuple

import keras
from keras import ops

from examples.gemma3.functional.core import (
    apply_reversible_embedding,
    apply_reversible_projection,
    build_reversible_embedding,
    build_rms_norm,
)
from examples.gemma3.functional.decoder import (
    apply_decoder_block,
    build_decoder_block_config,
    build_decoder_block_layers,
)
from examples.gemma3.functional.interleave import interleave_embeddings
from examples.gemma3.functional.vision import (
    apply_vision_encoder,
    build_vision_encoder_layers,
    compute_num_vision_tokens_per_image,
)


BackboneConfig = namedtuple(
    "BackboneConfig",
    [
        "vocabulary_size",
        "image_size",
        "num_layers",
        "num_query_heads",
        "num_key_value_heads",
        "hidden_dim",
        "intermediate_dim",
        "head_dim",
        "query_head_dim_normalize",
        "use_query_key_norm",
        "use_post_ffw_norm",
        "use_post_attention_norm",
        "attention_logit_soft_cap",
        "final_logit_soft_cap",
        "use_sliding_window_attention",
        "sliding_window_size",
        "local_rope_scaling_factor",
        "global_rope_scaling_factor",
        "use_bidirectional_attention",
        "layer_norm_epsilon",
        "dropout",
    ],
)

DecoderBlock = namedtuple("DecoderBlock", ["config", "layers"])

BackboneLayers = namedtuple(
    "BackboneLayers",
    [
        "token_embedding",
        "decoder_blocks",
        "final_norm",
        "vision_layers",
        "vision_config",
        "num_vision_tokens_per_image",
        "text_only_model",
    ],
)


def build_backbone_config(
    vocabulary_size,
    image_size,
    num_layers,
    num_query_heads,
    num_key_value_heads,
    hidden_dim,
    intermediate_dim,
    head_dim,
    query_head_dim_normalize=True,
    use_query_key_norm=True,
    use_post_ffw_norm=False,
    use_post_attention_norm=False,
    attention_logit_soft_cap=None,
    final_logit_soft_cap=None,
    use_sliding_window_attention=False,
    sliding_window_size=1024,
    local_rope_scaling_factor=1.0,
    global_rope_scaling_factor=1.0,
    use_bidirectional_attention=False,
    layer_norm_epsilon=1e-6,
    dropout=0.0,
):
    return BackboneConfig(
        vocabulary_size=vocabulary_size,
        image_size=image_size,
        num_layers=num_layers,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        head_dim=head_dim,
        query_head_dim_normalize=query_head_dim_normalize,
        use_query_key_norm=use_query_key_norm,
        use_post_ffw_norm=use_post_ffw_norm,
        use_post_attention_norm=use_post_attention_norm,
        attention_logit_soft_cap=attention_logit_soft_cap,
        final_logit_soft_cap=final_logit_soft_cap,
        use_sliding_window_attention=use_sliding_window_attention,
        sliding_window_size=sliding_window_size,
        local_rope_scaling_factor=local_rope_scaling_factor,
        global_rope_scaling_factor=global_rope_scaling_factor,
        use_bidirectional_attention=use_bidirectional_attention,
        layer_norm_epsilon=layer_norm_epsilon,
        dropout=dropout,
    )


def _build_decoder_blocks(config, dtype=None, name_prefix="gemma3"):
    decoder_blocks = []
    for layer_index in range(config.num_layers):
        sliding_window = config.use_sliding_window_attention and (
            layer_index % 6 < 5
        )
        rope_wavelength = 10_000.0 if sliding_window else 1_000_000.0
        rope_scaling_factor = (
            config.local_rope_scaling_factor
            if sliding_window
            else config.global_rope_scaling_factor
        )
        block_config = build_decoder_block_config(
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.intermediate_dim,
            head_dim=config.head_dim,
            num_query_heads=config.num_query_heads,
            num_key_value_heads=config.num_key_value_heads,
            query_head_dim_normalize=config.query_head_dim_normalize,
            use_query_key_norm=config.use_query_key_norm,
            use_post_ffw_norm=config.use_post_ffw_norm,
            use_post_attention_norm=config.use_post_attention_norm,
            gate_dim_reduction=1,
            logit_soft_cap=config.attention_logit_soft_cap,
            use_sliding_window_attention=sliding_window,
            sliding_window_size=config.sliding_window_size,
            layer_norm_epsilon=config.layer_norm_epsilon,
            rope_wavelength=rope_wavelength,
            rope_scaling_factor=rope_scaling_factor,
            use_bidirectional_attention=config.use_bidirectional_attention,
            dropout=config.dropout,
        )
        block_layers = build_decoder_block_layers(
            block_config,
            dtype=dtype,
            name_prefix="{}_decoder_block_{}".format(name_prefix, layer_index),
        )
        decoder_blocks.append(DecoderBlock(block_config, block_layers))
    return decoder_blocks


def build_backbone_layers(
    config,
    vision_config=None,
    dtype=None,
    name_prefix="gemma3",
):
    token_embedding = build_reversible_embedding(
        vocabulary_size=config.vocabulary_size,
        hidden_dim=config.hidden_dim,
        logit_soft_cap=config.final_logit_soft_cap,
        dtype=dtype,
        name="{}_token_embedding".format(name_prefix),
    )

    vision_layers = None
    num_vision_tokens_per_image = None
    if vision_config is not None:
        vision_layers = build_vision_encoder_layers(
            vision_config, dtype=dtype, name_prefix="{}_vision".format(name_prefix)
        )
        num_vision_tokens_per_image = compute_num_vision_tokens_per_image(
            vision_config
        )

    decoder_blocks = _build_decoder_blocks(
        config, dtype=dtype, name_prefix=name_prefix
    )
    final_norm = build_rms_norm(
        "{}_final_norm".format(name_prefix),
        epsilon=config.layer_norm_epsilon,
        dtype=dtype,
    )
    text_only_model = vision_layers is None

    return BackboneLayers(
        token_embedding=token_embedding,
        decoder_blocks=decoder_blocks,
        final_norm=final_norm,
        vision_layers=vision_layers,
        vision_config=vision_config,
        num_vision_tokens_per_image=num_vision_tokens_per_image,
        text_only_model=text_only_model,
    )


def apply_gemma3_backbone(
    layers,
    config,
    token_ids,
    padding_mask,
    images=None,
    vision_indices=None,
    vision_mask=None,
    training=False,
):
    text_embeddings = apply_reversible_embedding(layers.token_embedding, token_ids)
    text_embeddings = text_embeddings * ops.cast(
        ops.sqrt(config.hidden_dim), text_embeddings.dtype
    )

    if layers.text_only_model:
        hidden = text_embeddings
    else:
        image_embeddings = apply_vision_encoder(
            layers.vision_layers,
            layers.vision_config,
            images,
            training=training,
        )
        hidden = interleave_embeddings(
            image_embeddings,
            text_embeddings,
            vision_indices,
            layers.num_vision_tokens_per_image,
        )

    for block in layers.decoder_blocks:
        hidden = apply_decoder_block(
            block.layers,
            block.config,
            hidden,
            padding_mask=padding_mask,
            vision_mask=vision_mask if not layers.text_only_model else None,
            training=training,
        )
    return layers.final_norm(hidden)


def build_gemma3_backbone_model(
    config,
    vision_config=None,
    dtype=None,
    name="gemma3_backbone",
    sequence_length=None,
    num_images=None,
    batch_size=None,
):
    if vision_config is None:
        token_shape = (sequence_length,) if sequence_length is not None else (None,)
        token_id_input = keras.Input(
            shape=token_shape,
            dtype="int32",
            name="token_ids",
            batch_size=batch_size,
        )
        padding_mask_input = keras.Input(
            shape=token_shape,
            dtype="int32",
            name="padding_mask",
            batch_size=batch_size,
        )
        layers = build_backbone_layers(
            config, vision_config=None, dtype=dtype, name_prefix=name
        )
        outputs = apply_gemma3_backbone(
            layers,
            config,
            token_id_input,
            padding_mask_input,
        )
        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
        }
    else:
        image_batch_shape = (
            num_images if num_images is not None else None,
            config.image_size,
            config.image_size,
            3,
        )
        token_shape = (sequence_length,) if sequence_length is not None else (None,)
        num_vision_tokens = None
        if num_images is not None:
            num_vision_tokens = (
                num_images * compute_num_vision_tokens_per_image(vision_config)
            )
        vision_indices_shape = (
            (num_vision_tokens,)
            if num_vision_tokens is not None
            else (None,)
        )
        image_input = keras.Input(
            shape=image_batch_shape,
            name="images",
            batch_size=batch_size,
        )
        vision_indices_input = keras.Input(
            shape=vision_indices_shape,
            dtype="int32",
            name="vision_indices",
            batch_size=batch_size,
        )
        vision_mask_input = keras.Input(
            shape=token_shape,
            dtype="int32",
            name="vision_mask",
            batch_size=batch_size,
        )
        token_id_input = keras.Input(
            shape=token_shape,
            dtype="int32",
            name="token_ids",
            batch_size=batch_size,
        )
        padding_mask_input = keras.Input(
            shape=token_shape,
            dtype="int32",
            name="padding_mask",
            batch_size=batch_size,
        )
        layers = build_backbone_layers(
            config, vision_config=vision_config, dtype=dtype, name_prefix=name
        )
        outputs = apply_gemma3_backbone(
            layers,
            config,
            token_id_input,
            padding_mask_input,
            images=image_input,
            vision_indices=vision_indices_input,
            vision_mask=vision_mask_input,
        )
        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
            "images": image_input,
            "vision_indices": vision_indices_input,
            "vision_mask": vision_mask_input,
        }

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model, layers


def build_gemma3_causal_lm_model(
    backbone_layers,
    backbone_config,
    name="gemma3_causal_lm",
    sequence_length=None,
    num_images=None,
    batch_size=None,
):
    if backbone_layers.text_only_model:
        token_shape = (sequence_length,) if sequence_length is not None else (None,)
        token_id_input = keras.Input(
            shape=token_shape,
            dtype="int32",
            name="token_ids",
            batch_size=batch_size,
        )
        padding_mask_input = keras.Input(
            shape=token_shape,
            dtype="int32",
            name="padding_mask",
            batch_size=batch_size,
        )
        hidden = apply_gemma3_backbone(
            backbone_layers,
            backbone_config,
            token_id_input,
            padding_mask_input,
        )
        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
        }
    else:
        image_batch_shape = (
            num_images if num_images is not None else None,
            backbone_config.image_size,
            backbone_config.image_size,
            3,
        )
        token_shape = (sequence_length,) if sequence_length is not None else (None,)
        num_vision_tokens = None
        if num_images is not None:
            num_vision_tokens = (
                num_images * backbone_layers.num_vision_tokens_per_image
            )
        vision_indices_shape = (
            (num_vision_tokens,)
            if num_vision_tokens is not None
            else (None,)
        )
        image_input = keras.Input(
            shape=image_batch_shape,
            name="images",
            batch_size=batch_size,
        )
        vision_indices_input = keras.Input(
            shape=vision_indices_shape,
            dtype="int32",
            name="vision_indices",
            batch_size=batch_size,
        )
        vision_mask_input = keras.Input(
            shape=token_shape,
            dtype="int32",
            name="vision_mask",
            batch_size=batch_size,
        )
        token_id_input = keras.Input(
            shape=token_shape,
            dtype="int32",
            name="token_ids",
            batch_size=batch_size,
        )
        padding_mask_input = keras.Input(
            shape=token_shape,
            dtype="int32",
            name="padding_mask",
            batch_size=batch_size,
        )
        hidden = apply_gemma3_backbone(
            backbone_layers,
            backbone_config,
            token_id_input,
            padding_mask_input,
            images=image_input,
            vision_indices=vision_indices_input,
            vision_mask=vision_mask_input,
        )
        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
            "images": image_input,
            "vision_indices": vision_indices_input,
            "vision_mask": vision_mask_input,
        }

    logits = apply_reversible_projection(backbone_layers.token_embedding, hidden)
    return keras.Model(inputs=inputs, outputs=logits, name=name)


def call_with_cache(
    backbone_layers,
    backbone_config,
    token_ids,
    cache,
    cache_update_index,
    img_embeddings=None,
    vision_mask=None,
    padding_mask=None,
    vision_indices=None,
    cache_update_mask=None,
):
    text_embeddings = apply_reversible_embedding(backbone_layers.token_embedding, token_ids)
    text_embeddings = text_embeddings * ops.cast(
        ops.sqrt(backbone_config.hidden_dim), text_embeddings.dtype
    )

    if img_embeddings is not None:
        hidden = interleave_embeddings(
            img_embeddings,
            text_embeddings,
            vision_indices,
            backbone_layers.num_vision_tokens_per_image,
        )
    else:
        hidden = text_embeddings

    caches = []
    for block_index, block in enumerate(backbone_layers.decoder_blocks):
        current_cache = cache[:, block_index, ...]
        hidden, next_cache = apply_decoder_block(
            block.layers,
            block.config,
            hidden,
            padding_mask=padding_mask,
            vision_mask=vision_mask,
            cache=current_cache,
            cache_update_index=cache_update_index,
            cache_update_mask=cache_update_mask,
        )
        caches.append(next_cache)

    cache = ops.stack(caches, axis=1)
    hidden_states = backbone_layers.final_norm(hidden)
    logits = apply_reversible_projection(backbone_layers.token_embedding, hidden_states)
    return logits, hidden_states, cache


def build_cache(
    backbone_layers,
    backbone_config,
    token_ids,
    img_embeddings=None,
    vision_mask=None,
    padding_mask=None,
    vision_indices=None,
):
    batch_size = ops.shape(token_ids)[0]
    max_length = ops.shape(token_ids)[1]
    num_layers = len(backbone_layers.decoder_blocks)
    num_heads = backbone_config.num_key_value_heads
    head_dim = backbone_config.head_dim
    cache_shape = [batch_size, num_layers, 2, max_length, num_heads, head_dim]
    cache = ops.zeros(cache_shape, dtype=backbone_layers.final_norm.compute_dtype)
    logits, hidden_states, cache = call_with_cache(
        backbone_layers,
        backbone_config,
        token_ids,
        cache=cache,
        cache_update_index=0,
        img_embeddings=img_embeddings,
        vision_mask=vision_mask,
        padding_mask=padding_mask,
        vision_indices=vision_indices,
        cache_update_mask=None,
    )
    return hidden_states, cache
