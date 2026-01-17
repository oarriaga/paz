import sys
from pathlib import Path

import numpy as np
import keras
from keras import ops

from examples.gemma3.functional.core import apply_reversible_projection
from examples.gemma3.functional.model import (
    apply_gemma3_backbone,
    build_backbone_config,
    build_backbone_layers,
)
from examples.gemma3.functional.vision import build_vision_encoder_config

KERAS_HUB_ROOT = Path(__file__).resolve().parents[1] / "keras-hub"
if not hasattr(keras.layers, "ReversibleEmbedding"):
    class ReversibleEmbedding(keras.layers.Layer):
        def __init__(
            self,
            input_dim,
            output_dim,
            tie_weights=True,
            embeddings_initializer="uniform",
            logit_soft_cap=None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.tie_weights = tie_weights
            self.logit_soft_cap = logit_soft_cap
            self.embedding = keras.layers.Embedding(
                input_dim=input_dim,
                output_dim=output_dim,
                embeddings_initializer=embeddings_initializer,
                dtype=self.dtype_policy,
                name="embedding",
            )

        def build(self, input_shape):
            self.embedding.build(input_shape)
            self.built = True

        def call(self, inputs, reverse=False):
            if not reverse:
                return self.embedding(inputs)
            kernel = self.embedding.embeddings
            logits = ops.matmul(inputs, ops.transpose(kernel))
            if self.logit_soft_cap is None:
                return logits
            logits = ops.divide(logits, self.logit_soft_cap)
            logits = ops.multiply(ops.tanh(logits), self.logit_soft_cap)
            return logits

    keras.layers.ReversibleEmbedding = ReversibleEmbedding
sys.path.insert(0, str(KERAS_HUB_ROOT))

from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_causal_lm import Gemma3CausalLM
from keras_hub.src.models.gemma3.gemma3_vision_encoder import Gemma3VisionEncoder


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


def _copy_vision_encoder_weights(clean_layers, hub_model):
    hub_block = hub_model.get_layer("image_encoder")
    hub_embedding = hub_block.vision_embeddings
    clean_layers.vision_layers.embedding_layers.patch_embedding.set_weights(
        hub_embedding.patch_embedding.get_weights()
    )
    clean_layers.vision_layers.embedding_layers.position_embedding.set_weights(
        hub_embedding.position_embedding.get_weights()
    )
    for clean_layer, hub_layer in zip(
        clean_layers.vision_layers.encoder_layers, hub_block.resblocks
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

    clean_layers.vision_layers.encoder_layer_norm.set_weights(
        hub_block.encoder_layer_norm.get_weights()
    )

    hub_output_layer = hub_model.get_layer("vision_output_encoder")
    _copy_rms_norm_weights(
        clean_layers.vision_layers.output_layers.soft_embedding_norm,
        hub_output_layer.vision_soft_embedding_norm,
    )
    clean_layers.vision_layers.output_layers.input_projection.set_weights(
        hub_output_layer.vision_input_projection.get_weights()
    )


def _copy_backbone_weights(clean_layers, hub_backbone):
    clean_layers.token_embedding.embedding.set_weights(
        hub_backbone.token_embedding.get_weights()
    )
    for clean_block, hub_block in zip(
        clean_layers.decoder_blocks, hub_backbone.transformer_layers
    ):
        _copy_decoder_block_weights(clean_block, hub_block)
    _copy_rms_norm_weights(clean_layers.final_norm, hub_backbone.layer_norm)


def _collect_backbone_weights(clean_layers):
    weights = []
    weights.extend(clean_layers.token_embedding.embedding.weights)
    for block in clean_layers.decoder_blocks:
        weights.extend(block.layers.pre_attention_norm.weights)
        if block.layers.post_attention_norm is not None:
            weights.extend(block.layers.post_attention_norm.weights)
        attention_layers = block.layers.attention_layers
        weights.extend(attention_layers.query_dense.weights)
        weights.extend(attention_layers.key_dense.weights)
        weights.extend(attention_layers.value_dense.weights)
        weights.extend(attention_layers.output_dense.weights)
        if attention_layers.query_norm is not None:
            weights.extend(attention_layers.query_norm.weights)
        if attention_layers.key_norm is not None:
            weights.extend(attention_layers.key_norm.weights)
        weights.extend(block.layers.pre_ffw_norm.weights)
        if block.layers.post_ffw_norm is not None:
            weights.extend(block.layers.post_ffw_norm.weights)
        weights.extend(block.layers.gating_ffw.weights)
        weights.extend(block.layers.gating_ffw_2.weights)
        weights.extend(block.layers.ffw_linear.weights)
    weights.extend(clean_layers.final_norm.weights)

    if clean_layers.vision_layers is not None:
        weights.extend(
            clean_layers.vision_layers.embedding_layers.patch_embedding.weights
        )
        weights.extend(
            clean_layers.vision_layers.embedding_layers.position_embedding.weights
        )
        for encoder_layer in clean_layers.vision_layers.encoder_layers:
            attention_layers = encoder_layer.attention_layers
            weights.extend(attention_layers.query_proj.weights)
            weights.extend(attention_layers.key_proj.weights)
            weights.extend(attention_layers.value_proj.weights)
            weights.extend(attention_layers.out_proj.weights)
            weights.extend(encoder_layer.layer_norm_1.weights)
            weights.extend(encoder_layer.layer_norm_2.weights)
            weights.extend(encoder_layer.mlp_dense_1.weights)
            weights.extend(encoder_layer.mlp_dense_2.weights)
        weights.extend(clean_layers.vision_layers.encoder_layer_norm.weights)
        weights.extend(clean_layers.vision_layers.output_layers.soft_embedding_norm.weights)
        weights.extend(clean_layers.vision_layers.output_layers.input_projection.weights)

    return weights


def _count_params(weights):
    return int(sum(np.prod(weight.shape) for weight in weights))


def _build_text_only_setup():
    config = build_backbone_config(
        vocabulary_size=32,
        image_size=16,
        num_layers=2,
        num_query_heads=2,
        num_key_value_heads=1,
        hidden_dim=8,
        intermediate_dim=16,
        head_dim=4,
        query_head_dim_normalize=True,
        use_query_key_norm=True,
        use_post_ffw_norm=False,
        use_post_attention_norm=False,
        attention_logit_soft_cap=None,
        final_logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=8,
        local_rope_scaling_factor=1.0,
        global_rope_scaling_factor=1.0,
        use_bidirectional_attention=False,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
    )
    hub_model = Gemma3Backbone(
        vocabulary_size=32,
        image_size=16,
        num_layers=2,
        num_query_heads=2,
        num_key_value_heads=1,
        hidden_dim=8,
        intermediate_dim=16,
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
    clean_layers = build_backbone_layers(
        config, vision_config=None, dtype="float32", name_prefix="clean_text"
    )

    token_ids = ops.convert_to_tensor([[1, 2, 3], [4, 5, 0]], dtype="int32")
    padding_mask = ops.convert_to_tensor([[1, 1, 1], [1, 1, 0]], dtype="int32")
    inputs = {"token_ids": token_ids, "padding_mask": padding_mask}

    hub_model(inputs)
    apply_gemma3_backbone(
        clean_layers,
        config,
        token_ids,
        padding_mask,
        training=False,
    )
    return config, hub_model, clean_layers, inputs


def _build_vision_setup():
    vision_config = build_vision_encoder_config(
        image_size=16,
        patch_size=4,
        num_heads=2,
        hidden_dim=8,
        num_layers=2,
        intermediate_dim=16,
        output_dim=8,
        pool_size=2,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
    )
    config = build_backbone_config(
        vocabulary_size=32,
        image_size=16,
        num_layers=2,
        num_query_heads=2,
        num_key_value_heads=1,
        hidden_dim=8,
        intermediate_dim=16,
        head_dim=4,
        query_head_dim_normalize=True,
        use_query_key_norm=True,
        use_post_ffw_norm=False,
        use_post_attention_norm=False,
        attention_logit_soft_cap=None,
        final_logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=8,
        local_rope_scaling_factor=1.0,
        global_rope_scaling_factor=1.0,
        use_bidirectional_attention=False,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
    )
    hub_vision_encoder = Gemma3VisionEncoder(
        image_size=16,
        patch_size=4,
        num_heads=2,
        hidden_dim=8,
        num_layers=2,
        intermediate_dim=16,
        output_dim=8,
        pool_size=2,
        layer_norm_epsilon=1e-6,
        dtype="float32",
        name="hub_vision",
    )
    hub_model = Gemma3Backbone(
        vocabulary_size=32,
        image_size=16,
        num_layers=2,
        num_query_heads=2,
        num_key_value_heads=1,
        hidden_dim=8,
        intermediate_dim=16,
        head_dim=4,
        query_head_dim_normalize=True,
        use_query_key_norm=True,
        use_post_ffw_norm=False,
        use_post_attention_norm=False,
        attention_logit_soft_cap=None,
        final_logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=8,
        vision_encoder=hub_vision_encoder,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
        dtype="float32",
        name="hub_vision_backbone",
    )
    clean_layers = build_backbone_layers(
        config, vision_config=vision_config, dtype="float32", name_prefix="clean_vision"
    )

    token_ids = ops.convert_to_tensor([[1, 2, 3, 4, 5]], dtype="int32")
    padding_mask = ops.convert_to_tensor([[1, 1, 1, 1, 1]], dtype="int32")
    images = ops.convert_to_tensor(
        np.random.default_rng(0)
        .standard_normal((1, 1, 16, 16, 3))
        .astype("float32")
    )
    vision_indices = ops.convert_to_tensor([[1, 2, 3, 4]], dtype="int32")
    vision_mask = ops.convert_to_tensor([[0, 1, 1, 1, 1]], dtype="int32")

    inputs = {
        "token_ids": token_ids,
        "padding_mask": padding_mask,
        "images": images,
        "vision_indices": vision_indices,
        "vision_mask": vision_mask,
    }

    hub_model(inputs)
    apply_gemma3_backbone(
        clean_layers,
        config,
        token_ids,
        padding_mask,
        images=images,
        vision_indices=vision_indices,
        vision_mask=vision_mask,
        training=False,
    )
    return config, hub_model, clean_layers, inputs


def test_backbone_text_only_param_count_matches_hub():
    _, hub_model, clean_layers, _ = _build_text_only_setup()
    hub_count = hub_model.count_params()
    clean_count = _count_params(_collect_backbone_weights(clean_layers))
    assert hub_count == clean_count


def test_backbone_text_only_output_matches_hub():
    config, hub_model, clean_layers, inputs = _build_text_only_setup()
    _copy_backbone_weights(clean_layers, hub_model)
    hub_output = hub_model(inputs)
    clean_output = apply_gemma3_backbone(
        clean_layers,
        config,
        inputs["token_ids"],
        inputs["padding_mask"],
        training=False,
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(clean_output),
        ops.convert_to_numpy(hub_output),
        rtol=1e-5,
        atol=1e-5,
    )


def test_backbone_vision_param_count_matches_hub():
    _, hub_model, clean_layers, _ = _build_vision_setup()
    hub_count = hub_model.count_params()
    clean_count = _count_params(_collect_backbone_weights(clean_layers))
    assert hub_count == clean_count


def test_backbone_vision_output_matches_hub():
    config, hub_model, clean_layers, inputs = _build_vision_setup()
    _copy_backbone_weights(clean_layers, hub_model)
    _copy_vision_encoder_weights(clean_layers, hub_model.vision_encoder)
    hub_output = hub_model(inputs)
    clean_output = apply_gemma3_backbone(
        clean_layers,
        config,
        inputs["token_ids"],
        inputs["padding_mask"],
        images=inputs["images"],
        vision_indices=inputs["vision_indices"],
        vision_mask=inputs["vision_mask"],
        training=False,
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(clean_output),
        ops.convert_to_numpy(hub_output),
        rtol=1e-5,
        atol=1e-5,
    )


def test_causal_lm_logits_match_hub():
    config, hub_backbone, clean_layers, inputs = _build_text_only_setup()
    _copy_backbone_weights(clean_layers, hub_backbone)
    hub_model = Gemma3CausalLM(preprocessor=None, backbone=hub_backbone)
    hub_model(inputs)

    clean_hidden = apply_gemma3_backbone(
        clean_layers,
        config,
        inputs["token_ids"],
        inputs["padding_mask"],
        training=False,
    )
    clean_logits = apply_reversible_projection(
        clean_layers.token_embedding, clean_hidden
    )
    hub_output = hub_model(inputs)
    np.testing.assert_allclose(
        ops.convert_to_numpy(clean_logits),
        ops.convert_to_numpy(hub_output),
        rtol=1e-5,
        atol=1e-5,
    )

