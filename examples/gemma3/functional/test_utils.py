import numpy as np

from examples.gemma3.functional.keras_hub_utils import ensure_keras_hub

ensure_keras_hub()


def copy_rms_norm_weights(clean_layer, hub_layer):
    hub_weights = hub_layer.get_weights()
    clean_layer.set_weights([hub_weights[0] + 1.0])


def copy_attention_weights(clean_layers, hub_attention):
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
        copy_rms_norm_weights(query_norm, hub_attention.query_norm)
        copy_rms_norm_weights(key_norm, hub_attention.key_norm)


def copy_decoder_block_weights(clean_block, hub_block):
    pre_norm = clean_block[0]
    post_norm = clean_block[1]
    attention_layers = clean_block[2]
    pre_ffw_norm = clean_block[4]
    post_ffw_norm = clean_block[5]
    gating_ffw = clean_block[6]
    gating_ffw_2 = clean_block[7]
    ffw_linear = clean_block[8]
    copy_rms_norm_weights(pre_norm, hub_block.pre_attention_norm)
    if hub_block.use_post_attention_norm:
        copy_rms_norm_weights(post_norm, hub_block.post_attention_norm)
    copy_attention_weights(attention_layers, hub_block.attention)
    copy_rms_norm_weights(pre_ffw_norm, hub_block.pre_ffw_norm)
    if hub_block.use_post_ffw_norm:
        copy_rms_norm_weights(post_ffw_norm, hub_block.post_ffw_norm)
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
    copy_rms_norm_weights(soft_norm, hub_output.vision_soft_embedding_norm)
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
        copy_decoder_block_weights(block_layers, hub_block)
    copy_rms_norm_weights(final_norm, hub_backbone.layer_norm)
    if vision_layers is not None and hub_backbone.vision_encoder:
        copy_vision_encoder_weights(vision_layers, hub_backbone.vision_encoder)


def collect_backbone_weights(clean_layers):
    weights = []
    token_embedding = clean_layers[0]
    decoder_blocks = clean_layers[1]
    final_norm = clean_layers[2]
    vision_layers = clean_layers[3]
    weights.extend(token_embedding.weights)
    for block in decoder_blocks:
        block_layers = block[1]
        weights.extend(block_layers[0].weights)
        if block_layers[1] is not None:
            weights.extend(block_layers[1].weights)
        attention_layers = block_layers[2]
        weights.extend(attention_layers[0].weights)
        weights.extend(attention_layers[1].weights)
        weights.extend(attention_layers[2].weights)
        weights.extend(attention_layers[3].weights)
        if attention_layers[4] is not None:
            weights.extend(attention_layers[4].weights)
        if attention_layers[5] is not None:
            weights.extend(attention_layers[5].weights)
        weights.extend(block_layers[4].weights)
        if block_layers[5] is not None:
            weights.extend(block_layers[5].weights)
        weights.extend(block_layers[6].weights)
        weights.extend(block_layers[7].weights)
        weights.extend(block_layers[8].weights)
    weights.extend(final_norm.weights)

    if vision_layers is not None:
        embedding_layers = vision_layers[0]
        encoder_layers = vision_layers[1]
        encoder_norm = vision_layers[2]
        output_layers = vision_layers[4]
        patch = embedding_layers[0]
        position = embedding_layers[1]
        weights.extend(patch.weights)
        weights.extend(position.weights)
        for encoder_layer in encoder_layers:
            attention_layers = encoder_layer[0]
            weights.extend(attention_layers[0].weights)
            weights.extend(attention_layers[1].weights)
            weights.extend(attention_layers[2].weights)
            weights.extend(attention_layers[3].weights)
            weights.extend(encoder_layer[1].weights)
            weights.extend(encoder_layer[2].weights)
            weights.extend(encoder_layer[3].weights)
            weights.extend(encoder_layer[4].weights)
        weights.extend(encoder_norm.weights)
        weights.extend(output_layers[0].weights)
        weights.extend(output_layers[1].weights)

    return weights


def count_params(weights):
    return int(sum(np.prod(weight.shape) for weight in weights))
