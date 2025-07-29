import keras
from keras import ops, initializers
from keras.layers import Dense, Dropout, Layer


class Attention(Layer):
    def __init__(
        self,
        dimension,
        number_of_heads=8,
        use_query_key_value_bias=False,
        use_projection_bias=True,
        attention_drop_rate=0.0,
        projection_drop_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dimension = dimension
        self.number_of_heads = number_of_heads
        self.use_query_key_value_bias = use_query_key_value_bias
        self.use_projection_bias = use_projection_bias
        self.attention_drop_rate = attention_drop_rate
        self.projection_drop_rate = projection_drop_rate
        self.head_dimension = self.dimension // self.number_of_heads
        self.scale = self.head_dimension**-0.5

        initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        dense_kwargs = {"kernel_initializer": initializer}
        self.predict_query_key_value = Dense(
            self.dimension * 3,
            use_bias=self.use_query_key_value_bias,
            name="qkv",
            **dense_kwargs,
        )
        self.projection_layer = Dense(
            self.dimension,
            use_bias=self.use_projection_bias,
            name="proj",
            **dense_kwargs,
        )
        self.attention_drop = Dropout(self.attention_drop_rate)
        self.projection_drop = Dropout(self.projection_drop_rate)

    def build(self, input_shape):
        self.predict_query_key_value.build(input_shape)
        self.projection_layer.build((input_shape[0], input_shape[1], self.dimension))
        self.built = True

    def call(self, x, attention_bias=None, training=None):
        batch_size, number_of_tokens, channels = ops.shape(x)

        query_key_value = self.predict_query_key_value(x)
        query_key_value = ops.reshape(
            query_key_value,
            (
                batch_size,
                number_of_tokens,
                3,
                self.number_of_heads,
                self.head_dimension,
            ),
        )
        query_key_value = ops.transpose(query_key_value, axes=(2, 0, 3, 1, 4))
        query, key, value = query_key_value[0], query_key_value[1], query_key_value[2]

        attention = (
            ops.matmul(query, ops.transpose(key, axes=[0, 1, 3, 2])) * self.scale
        )

        if attention_bias is not None:
            attention = attention + attention_bias

        attention = ops.softmax(attention, axis=-1)
        attention = self.attention_drop(attention, training=training)

        x = ops.transpose((attention @ value), axes=(0, 2, 1, 3))
        x = ops.reshape(x, (batch_size, number_of_tokens, channels))

        x = self.projection_layer(x)
        x = self.projection_drop(x, training=training)
        return x
