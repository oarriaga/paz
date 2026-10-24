"""Bidirectional Llama-3 encoder used by LLM2Vec, as a functional Keras model.

Defaults are the Llama-3-8B-Instruct architecture: pre-norm decoder
blocks with grouped-query attention, rotary positions and SwiGLU, but
without the causal mask. The model maps token ids ``(1, T)`` to final
hidden states ``(1, T, hidden)``; pooling happens in ``encoding.py``.
Weights are bfloat16 like the reference, softmax runs in float32.
"""
from keras import Model, ops
from keras.layers import Dense, EinsumDense, Embedding, Input, Reshape

from paz.layers import RMSNormalization
from paz.models.transformers.embeddings import rotary


def build(vocab_size=128256, hidden_dim=4096, num_layers=32, num_heads=32,
          num_kv_heads=8, head_dim=128, ff_dim=14336,
          rope_wavelength=500000.0, epsilon=1e-5, dtype="bfloat16",
          name="llm2vec_llama3"):
    token_ids = Input((None,), dtype="int32", name="token_ids")
    embed = Embedding(vocab_size, hidden_dim, dtype=dtype,
                      name="token_embedding")
    x = embed(token_ids)
    args = (num_heads, num_kv_heads, head_dim, ff_dim, rope_wavelength,
            epsilon, dtype)
    for layer in range(num_layers):
        x = decoder_block(x, *args, f"layer_{layer}")
    hidden = build_norm(epsilon, dtype, "final_norm")(x)
    return Model(token_ids, hidden, name=name)


def decoder_block(x, num_heads, num_kv_heads, head_dim, ff_dim, wavelength,
                  epsilon, dtype, name):
    norm = build_norm(epsilon, dtype, f"{name}_attention_norm")
    args = (num_heads, num_kv_heads, head_dim, wavelength, dtype)
    x = x + attend(norm(x), *args, f"{name}_attention")
    norm = build_norm(epsilon, dtype, f"{name}_feedforward_norm")
    return x + feed_forward(norm(x), ff_dim, dtype, f"{name}_feedforward")


def build_norm(epsilon, dtype, name):
    return RMSNormalization(epsilon, dtype=dtype, name=name)


def attend(x, num_heads, num_kv_heads, head_dim, wavelength, dtype, name):
    query = project_heads(x, num_heads, head_dim, dtype, f"{name}_query")
    key = project_heads(x, num_kv_heads, head_dim, dtype, f"{name}_key")
    value = project_heads(x, num_kv_heads, head_dim, dtype, f"{name}_value")
    query = rotary.apply(query, wavelength, 1.0, head_dim)
    key = rotary.apply(key, wavelength, 1.0, head_dim)
    context = attend_groups(query, key, value, num_kv_heads)
    output = EinsumDense("btnh,nhd->btd", (None, x.shape[-1]), dtype=dtype,
                         name=f"{name}_output")
    return output(context)


def project_heads(x, num_heads, head_dim, dtype, name):
    layer = EinsumDense("btd,dnh->btnh", (None, num_heads, head_dim),
                        dtype=dtype, name=name)
    return layer(x)


def attend_groups(query, key, value, num_kv_heads):
    """Grouped-query attention over every key, no causal mask."""
    head_dim = query.shape[-1]
    groups = query.shape[-2] // num_kv_heads
    query = Reshape((-1, num_kv_heads, groups, head_dim))(query)
    scores = ops.einsum("btkgh,bskh->bkgts", query, key) * head_dim ** -0.5
    probabilities = ops.softmax(ops.cast(scores, "float32"), axis=-1)
    probabilities = ops.cast(probabilities, value.dtype)
    context = ops.einsum("bkgts,bskh->btkgh", probabilities, value)
    return Reshape((-1, num_kv_heads * groups, head_dim))(context)


def feed_forward(x, ff_dim, dtype, name):
    gate = dense(ff_dim, dtype, f"{name}_gate")(x)
    up = dense(ff_dim, dtype, f"{name}_up")(x)
    return dense(x.shape[-1], dtype, f"{name}_down")(ops.silu(gate) * up)


def dense(units, dtype, name):
    return Dense(units, use_bias=False, dtype=dtype, name=name)
