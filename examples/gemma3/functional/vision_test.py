import sys
from pathlib import Path

import numpy as np
import keras
from keras import ops

from examples.gemma3.functional.vision import (
    apply_vision_encoder,
    build_vision_encoder_config,
    build_vision_encoder_layers,
)

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

from keras_hub.src.models.gemma3.gemma3_vision_encoder import Gemma3VisionEncoder


def _copy_rms_norm_weights(clean_layer, hub_layer):
    hub_weights = hub_layer.get_weights()
    clean_layer.set_weights([hub_weights[0] + 1.0])


def _copy_attention_weights(clean_layers, hub_attention):
    clean_layers.query_proj.set_weights(hub_attention.query_proj.get_weights())
    clean_layers.key_proj.set_weights(hub_attention.key_proj.get_weights())
    clean_layers.value_proj.set_weights(hub_attention.value_proj.get_weights())
    clean_layers.out_proj.set_weights(hub_attention.out_proj.get_weights())


def _copy_encoder_layer_weights(clean_layer, hub_layer):
    _copy_attention_weights(clean_layer.attention_layers, hub_layer.attn)
    clean_layer.layer_norm_1.set_weights(hub_layer.layer_norm_1.get_weights())
    clean_layer.layer_norm_2.set_weights(hub_layer.layer_norm_2.get_weights())
    clean_layer.mlp_dense_1.set_weights(hub_layer.mlp_dense_1.get_weights())
    clean_layer.mlp_dense_2.set_weights(hub_layer.mlp_dense_2.get_weights())


def test_vision_encoder_output_matches_hub():
    rng = np.random.default_rng(3)
    config = build_vision_encoder_config(
        image_size=16,
        patch_size=4,
        num_heads=2,
        hidden_dim=8,
        num_layers=2,
        intermediate_dim=16,
        output_dim=12,
        pool_size=2,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
    )

    images = ops.convert_to_tensor(
        rng.standard_normal((2, 1, 16, 16, 3)).astype("float32")
    )

    hub_model = Gemma3VisionEncoder(
        image_size=16,
        patch_size=4,
        num_heads=2,
        hidden_dim=8,
        num_layers=2,
        intermediate_dim=16,
        output_dim=12,
        pool_size=2,
        layer_norm_epsilon=1e-6,
        dtype="float32",
        name="hub_vision",
    )
    hub_model(images)

    layers = build_vision_encoder_layers(config, dtype="float32", name_prefix="clean")
    apply_vision_encoder(layers, config, images, training=False)

    hub_block = hub_model.get_layer("image_encoder")
    hub_embedding = hub_block.vision_embeddings
    layers.embedding_layers.patch_embedding.set_weights(
        hub_embedding.patch_embedding.get_weights()
    )
    layers.embedding_layers.position_embedding.set_weights(
        hub_embedding.position_embedding.get_weights()
    )

    for clean_layer, hub_layer in zip(
        layers.encoder_layers, hub_block.resblocks
    ):
        _copy_encoder_layer_weights(clean_layer, hub_layer)

    layers.encoder_layer_norm.set_weights(
        hub_block.encoder_layer_norm.get_weights()
    )

    hub_output_layer = hub_model.get_layer("vision_output_encoder")
    _copy_rms_norm_weights(
        layers.output_layers.soft_embedding_norm,
        hub_output_layer.vision_soft_embedding_norm,
    )
    layers.output_layers.input_projection.set_weights(
        hub_output_layer.vision_input_projection.get_weights()
    )

    hub_output = hub_model(images)
    clean_output = apply_vision_encoder(layers, config, images, training=False)
    np.testing.assert_allclose(
        ops.convert_to_numpy(clean_output),
        ops.convert_to_numpy(hub_output),
        rtol=1e-5,
        atol=1e-5,
    )

