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
        self.attention_drop_rate = attention_drop_rate
        self.projection_drop_rate = projection_drop_rate
        self.head_dimension = self.dimension // self.num_heads
        self.scale = self.head_dimension**-0.5

        initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        kwargs = {"kernel_initializer": initializer}
        self.qkv = Dense(self.dimension * 3, use_bias=self.qkv_bias, name="qkv", **kwargs)
        self.proj = Dense(self.dimension, use_bias=self.projection_bias, name="proj", **kwargs)
        self.attention_drop = Dropout(self.attention_drop_rate)
        self.projection_drop = Dropout(self.projection_drop_rate)

    def call(self, x, attention_bias=None, training=None):
        batch_size, num_tokens, channels = ops.shape(x)

        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (batch_size, num_tokens, 3, self.num_heads, self.head_dimension))
        qkv = ops.transpose(qkv, axes=(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = ops.matmul(q, ops.transpose(k, axes=[0, 1, 3, 2])) * self.scale

        if attention_bias is not None:
            attention = attention + attention_bias

        attention = ops.softmax(attention, axis=-1)
        attention = self.attention_drop(attention, training=training)

        x = ops.transpose((attention @ v), axes=(0, 2, 1, 3))
        x = ops.reshape(x, (batch_size, num_tokens, channels))

        x = self.proj(x)
        x = self.projection_drop(x, training=training)
        return x


class MemEffAttention(Attention):
    def call(self, x, attention_bias=None, training=None):
        return super().call(x, attention_bias=attention_bias, training=training)
