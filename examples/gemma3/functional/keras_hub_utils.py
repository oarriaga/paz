from pathlib import Path
import sys

import keras
from keras import ops


def ensure_reversible_embedding():
    if hasattr(keras.layers, "ReversibleEmbedding"):
        return

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


def ensure_keras_hub():
    ensure_reversible_embedding()
    gemma3_root = Path(__file__).resolve().parents[1]
    keras_hub_root = gemma3_root / "keras-hub"
    if keras_hub_root.exists() and str(keras_hub_root) not in sys.path:
        sys.path.insert(0, str(keras_hub_root))
    import keras_hub

    return keras_hub

