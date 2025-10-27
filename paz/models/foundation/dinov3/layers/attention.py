import keras
from keras import ops
from keras.layers import Dense, Dropout
from paz.models.foundation.dinov3.utils.utils import cat_keep_shapes, uncat_with_shapes


def rope_rotate_half(x):
    """Rotates half the hidden dimensions."""
    x1, x2 = keras.ops.split(x, 2, axis=-1)
    return keras.ops.concatenate([-x2, x1], axis=-1)


def rope_apply(x, sin, cos):
    """Applies rotary position embedding."""
    return (x * cos) + (rope_rotate_half(x) * sin)


class SelfAttention(keras.Layer):
    """Keras implementation of DINOv3's SelfAttention layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = Dense(dim * 3, use_bias=qkv_bias, name="qkv")
        self.attn_drop_layer = Dropout(attn_drop)
        self.proj = Dense(dim, use_bias=proj_bias, name="proj")
        self.proj_drop_layer = Dropout(proj_drop)

    def apply_rope(self, q, k, rope):
        sin, cos = rope

        prefix = ops.shape(q)[-2] - ops.shape(sin)[-2]
        if prefix < 0:
            raise ValueError("Input sequence is shorter than the RoPE sequence.")

        q_prefix, q_main = q[:, :, :prefix, :], q[:, :, prefix:, :]
        k_prefix, k_main = k[:, :, :prefix, :], k[:, :, prefix:, :]

        q_rotated = rope_apply(q_main, sin, cos)
        k_rotated = rope_apply(k_main, sin, cos)

        q = ops.concatenate((q_prefix, q_rotated), axis=-2)
        k = ops.concatenate((k_prefix, k_rotated), axis=-2)

        return q, k

    def compute_attention(self, qkv, rope=None, training=None):
        B, N, _ = ops.shape(qkv)
        qkv_r = ops.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        q, k, v = ops.unstack(qkv_r, axis=2)
        q = ops.transpose(q, axes=[0, 2, 1, 3])
        k = ops.transpose(k, axes=[0, 2, 1, 3])
        v = ops.transpose(v, axes=[0, 2, 1, 3])

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        attn = ops.matmul(q, ops.transpose(k, axes=[0, 1, 3, 2]))
        attn = attn * self.scale
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop_layer(attn, training=training)

        x = ops.matmul(attn, v)
        x = ops.transpose(x, axes=[0, 2, 1, 3])

        return ops.reshape(x, (B, N, self.dim))

    def forward_tensor(self, x, rope=None, training=None):
        """Processes a single tensor input."""
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, rope=rope, training=training)
        x = self.proj(attn_v)
        return self.proj_drop_layer(x, training=training)

    def forward_list(self, x_list, rope_list=None, training=None):
        """Processes a list of tensors."""
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)

        if rope_list is None:
            rope_list = [None] * len(qkv_list)

        att_out = [
            self.compute_attention(qkv, rope=rope, training=training)
            for qkv, rope in zip(qkv_list, rope_list)
        ]

        x_flat_out, shapes_out, num_tokens_out = cat_keep_shapes(att_out)
        x_flat_out = self.proj(x_flat_out)
        x_flat_out = self.proj_drop_layer(x_flat_out, training=training)

        return uncat_with_shapes(x_flat_out, shapes_out, num_tokens_out)

    def call(self, x, rope=None, training=None):
        """Handles both a single tensor and a list of tensors."""
        if isinstance(x, list):
            return self.forward_list(x, rope_list=rope, training=training)
        else:
            return self.forward_tensor(x, rope=rope, training=training)


class CausalSelfAttention(keras.Layer):
    """Keras implementation of DINOv3's CausalSelfAttention layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = Dense(dim * 3, use_bias=qkv_bias, name="qkv")
        self.attn_drop_layer = Dropout(attn_drop)
        self.proj = Dense(dim, use_bias=proj_bias, name="proj")
        self.proj_drop_layer = Dropout(proj_drop)

    def call(self, x, is_causal=True, training=None):
        B, N, C = ops.shape(x)
        qkv = self.qkv(x)
        qkv_r = ops.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        q, k, v = ops.unstack(qkv_r, axis=2)
        q = ops.transpose(q, axes=[0, 2, 1, 3])
        k = ops.transpose(k, axes=[0, 2, 1, 3])
        v = ops.transpose(v, axes=[0, 2, 1, 3])

        attn = ops.matmul(q, ops.transpose(k, axes=[0, 1, 3, 2]))
        attn = attn * self.scale

        if is_causal:
            seq_len = ops.shape(q)[2]
            mask = ops.triu(ops.ones((seq_len, seq_len)), k=1)
            attn += mask * -1e9

        attn = ops.softmax(attn, axis=-1)

        attn = self.attn_drop_layer(attn, training=training)
        x = ops.matmul(attn, v)

        x = ops.transpose(x, axes=[0, 2, 1, 3])
        x = ops.reshape(x, (B, N, C))
        x = self.proj(x)
        return self.proj_drop_layer(x, training=training)
