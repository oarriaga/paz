import os
from pathlib import Path
import sys


def ensure_jax_backend():
    backend = os.environ.get("KERAS_BACKEND")
    if backend is None:
        os.environ["KERAS_BACKEND"] = "jax"
        return
    if backend != "jax":
        message = "KERAS_BACKEND must be 'jax' for Gemma3 tests, got '{}'."
        message = message.format(backend)
        raise ValueError(message)


def ensure_reversible_embedding():
    import keras
    from keras import ops

    if hasattr(keras.layers, "ReversibleEmbedding"):
        return

    class ReversibleEmbedding(keras.layers.Layer):
        def __init__(self, input_dim, output_dim, **kwargs):
            tie_weights = kwargs.pop("tie_weights", True)
            init = kwargs.pop("embeddings_initializer", "uniform")
            logit_soft_cap = kwargs.pop("logit_soft_cap", None)
            super().__init__(**kwargs)
            self.tie_weights = tie_weights
            self.logit_soft_cap = logit_soft_cap
            embed_kwargs = {}
            embed_kwargs["input_dim"] = input_dim
            embed_kwargs["output_dim"] = output_dim
            embed_kwargs["embeddings_initializer"] = init
            embed_kwargs["dtype"] = self.dtype_policy
            embed_kwargs["name"] = "embedding"
            self.embedding = keras.layers.Embedding(**embed_kwargs)

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
    ensure_jax_backend()
    ensure_reversible_embedding()
    import keras

    if keras.config.backend() != "jax":
        message = "Keras backend must be 'jax', got '{}'."
        message = message.format(keras.config.backend())
        raise ValueError(message)
    gemma3_root = Path(__file__).resolve().parents[1]
    keras_hub_root = gemma3_root / "keras-hub"
    if not keras_hub_root.exists():
        message = "Expected local keras-hub checkout at '{}'."
        message = message.format(keras_hub_root)
        raise FileNotFoundError(message)
    if str(keras_hub_root) not in sys.path:
        sys.path.insert(0, str(keras_hub_root))
    import keras_hub

    return keras_hub
