import keras
from keras import ops, initializers
from keras.layers import Dense, Dropout, Layer


class Attention(Layer):
    def __init__(
        self,
        dimension,
        num_heads=8,
        qkv_bias=False,
        projection_bias=True,
        attention_drop_rate=0.0,
        projection_drop_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dimension = dimension
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.projection_bias = projection_bias
        self.attention_drop_rate_val = attention_drop_rate
        self.projection_drop_rate_val = projection_drop_rate
        self.head_dim = self.dimension // self.num_heads
        self.scale = self.head_dim**-0.5

        initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        self.qkv = Dense(
            self.dimension * 3, use_bias=self.qkv_bias, kernel_initializer=initializer, name="qkv"
        )
        self.proj = Dense(
            self.dimension, use_bias=self.projection_bias, kernel_initializer=initializer, name="proj"
        )
        self.attention_drop = Dropout(self.attention_drop_rate_val)
        self.projection_drop = Dropout(self.projection_drop_rate_val)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, attn_bias=None, training=None):
        B, N, C = ops.shape(x)
        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, axes=(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = ops.matmul(q, ops.transpose(k, axes=[0, 1, 3, 2])) * self.scale

        if attn_bias is not None:
            attn += attn_bias

        attn = ops.softmax(attn, axis=-1)
        attn = self.attention_drop(attn, training=training)

        x = ops.transpose((attn @ v), axes=(0, 2, 1, 3))
        x = ops.reshape(x, (B, N, C))

        x = self.proj(x)
        x = self.projection_drop(x, training=training)
        return x


class MemEffAttention(Attention):
    def call(self, x, attn_bias=None, training=None):
        return super().call(x, attn_bias=attn_bias, training=training)
