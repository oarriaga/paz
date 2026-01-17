import sys
from pathlib import Path

import numpy as np
import keras
from keras import ops

from examples.gemma3.functional.interleave import interleave_embeddings

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

from keras_hub.src.models.gemma3.gemma3_interleave_embeddings import (
    Gemma3InterleaveEmbeddings,
)


def test_interleave_embeddings_matches_hub():
    rng = np.random.default_rng(4)
    batch_size = 2
    sequence_length = 6
    embedding_dim = 4
    num_vision_tokens_per_image = 2

    text_embeddings = ops.convert_to_tensor(
        rng.standard_normal((batch_size, sequence_length, embedding_dim)).astype(
            "float32"
        )
    )
    image_embeddings = ops.convert_to_tensor(
        rng.standard_normal((batch_size, num_vision_tokens_per_image, embedding_dim)).astype(
            "float32"
        )
    )
    vision_indices = ops.convert_to_tensor(
        [[2, 3], [1, 2]], dtype="int32"
    )

    hub_layer = Gemma3InterleaveEmbeddings(num_vision_tokens_per_image)
    hub_output = hub_layer(image_embeddings, text_embeddings, vision_indices)
    clean_output = interleave_embeddings(
        image_embeddings,
        text_embeddings,
        vision_indices,
        num_vision_tokens_per_image,
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(clean_output),
        ops.convert_to_numpy(hub_output),
        rtol=1e-6,
        atol=1e-6,
    )

