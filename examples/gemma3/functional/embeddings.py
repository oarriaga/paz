from keras import ops
from keras.initializers import VarianceScaling
from keras.layers import Embedding

from examples.gemma3.functional.core import apply_tanh_soft_cap


def build_reversible_embedding(vocabulary_size, hidden_dim, dtype=None, name="token_embedding"):
    initializer = VarianceScaling(
        scale=1.0, mode="fan_in", distribution="untruncated_normal"
    )
    return Embedding(
        vocabulary_size,
        hidden_dim,
        embeddings_initializer=initializer,
        dtype=dtype,
        name=name,
    )


def apply_reversible_projection(embedding, hidden_states, logit_soft_cap=None):
    kernel = embedding.embeddings
    logits = ops.matmul(hidden_states, ops.transpose(kernel))
    return apply_tanh_soft_cap(logits, logit_soft_cap)


def _apply_token_embedding(token_embedding, hidden_dim, token_ids):
    text = token_embedding(token_ids)
    scale = ops.cast(ops.sqrt(hidden_dim), text.dtype)
    return text * scale
