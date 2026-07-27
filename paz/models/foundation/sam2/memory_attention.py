"""SAM 2 memory attention: condition frame features on the memory bank.

Four layers of RoPE self-attention over the current frame followed by RoPE
cross-attention to the memory (spatial memories plus object pointers), then an
MLP. Rotary tables are passed in as cos/sin inputs so the graph stays static;
object-pointer positions get an identity rotation instead of a runtime slice.
Each memory token also carries a one-hot row selecting its frame's slot in the
learned ``maskmem_tpos_enc`` table, which is added to its position; pointer
tokens get an all-zero row because their position is already temporal.

The memory arrives padded to a fixed length with a keep-mask, so the tracker
compiles this graph once instead of once per bank size. Masked keys are dropped
before the softmax, which makes the result identical to passing the shorter
memory.
"""
import numpy as np
from keras import Input, Model, ops
from keras.layers import Dense, LayerNormalization, Reshape

from paz.models.transformers.attention import compute_masked_attention
from paz.models.transformers.attention import expand_mask_for_heads
from paz.models.foundation.sam2.configuration import MEMORY_DIM
from paz.models.foundation.sam2.configuration import NUM_MEMORIES

MODEL_DIM = 256
NUM_HEADS = 1
NUM_LAYERS = 4
ROPE_DIM = MODEL_DIM // 2


def build(name="sam2_memory_attention"):
    curr = Input((None, MODEL_DIM), name="curr")
    curr_pos = Input((None, MODEL_DIM), name="curr_pos")
    memory = Input((None, MEMORY_DIM), name="memory")
    memory_pos = Input((None, MEMORY_DIM), name="memory_pos")
    memory_time = Input((None, NUM_MEMORIES), name="memory_time")
    memory_mask = Input((None,), name="memory_mask")
    curr_cos = Input((None, ROPE_DIM), name="curr_cos")
    curr_sin = Input((None, ROPE_DIM), name="curr_sin")
    memory_cos = Input((None, ROPE_DIM), name="memory_cos")
    memory_sin = Input((None, ROPE_DIM), name="memory_sin")
    tokens = ops.add(curr, 0.1 * curr_pos)
    positions = ops.add(memory_pos, temporal_encoding(memory_time))
    keep = expand_mask_for_heads(memory_mask)
    rope = curr_cos, curr_sin, memory_cos, memory_sin
    for index in range(NUM_LAYERS):
        tokens = apply_layer(tokens, memory, positions, keep, rope, index)
    tokens = normalize(tokens, "mematt_norm")
    tensors = (curr, curr_pos, memory, memory_pos, memory_time, memory_mask)
    tables = (curr_cos, curr_sin, memory_cos, memory_sin)
    return Model(tensors + tables, tokens, name=name)


def temporal_encoding(memory_time):
    kwargs = dict(use_bias=False, name="maskmem_tpos_enc")
    return Dense(MEMORY_DIM, **kwargs)(memory_time)


def apply_layer(tokens, memory, memory_pos, keep, rope, index):
    curr_cos, curr_sin, memory_cos, memory_sin = rope
    name = f"mematt_{index}"
    normed = normalize(tokens, f"{name}_norm1")
    query = (normed, curr_cos, curr_sin)
    key = (normed, curr_cos, curr_sin)
    attended = attention(query, key, normed, None, f"{name}_self")
    tokens = ops.add(tokens, attended)
    normed = normalize(tokens, f"{name}_norm2")
    query = (normed, curr_cos, curr_sin)
    key = (ops.add(memory, memory_pos), memory_cos, memory_sin)
    attended = attention(query, key, memory, keep, f"{name}_cross")
    tokens = ops.add(tokens, attended)
    normed = normalize(tokens, f"{name}_norm3")
    forwarded = feedforward(normed, name)
    return ops.add(tokens, forwarded)


def attention(query, key, value, keep, name):
    query_tokens, query_cos, query_sin = query
    key_tokens, key_cos, key_sin = key
    q = Dense(MODEL_DIM, name=f"{name}_q")(query_tokens)
    k = Dense(MODEL_DIM, name=f"{name}_k")(key_tokens)
    v = Dense(MODEL_DIM, name=f"{name}_v")(value)
    q = apply_rotary(q, query_cos, query_sin)
    k = apply_rotary(k, key_cos, key_sin)
    heads = to_head(q), to_head(k), to_head(v)
    context = compute_masked_attention(*heads, keep)
    return Dense(MODEL_DIM, name=f"{name}_out")(from_head(context))


def to_head(x):
    return ops.expand_dims(x, axis=1)


def from_head(context):
    return ops.squeeze(context, axis=1)


def apply_rotary(x, cos, sin):
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    stacked = ops.stack([rotated_even, rotated_odd], axis=-1)
    return Reshape((-1, MODEL_DIM))(stacked)


def feedforward(x, name):
    hidden = Dense(2048, activation="relu", name=f"{name}_mlp1")(x)
    return Dense(MODEL_DIM, name=f"{name}_mlp2")(hidden)


def normalize(x, name):
    return LayerNormalization(epsilon=1e-5, name=name)(x)


def rotary_tables(end_x, end_y, dim=MODEL_DIM, theta=10000.0):
    positions = np.arange(end_x * end_y)
    columns = (positions % end_x).astype(np.float32)
    rows = (positions // end_x).astype(np.float32)
    frequencies = 1.0 / (theta ** (np.arange(0, dim, 4)[:dim // 4] / dim))
    angles_x = np.outer(columns, frequencies)
    angles_y = np.outer(rows, frequencies)
    angles = np.concatenate([angles_x, angles_y], axis=-1).astype(np.float32)
    return np.cos(angles), np.sin(angles)


def identity_tables(count):
    cos = np.ones((count, ROPE_DIM), np.float32)
    sin = np.zeros((count, ROPE_DIM), np.float32)
    return cos, sin
