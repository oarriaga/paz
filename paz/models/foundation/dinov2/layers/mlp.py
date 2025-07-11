import keras


class MLP(keras.layers.Layer):
    def __init__(
        self,
        input_features,
        hidden_features=None,
        output_features=None,
        activation_layer=None,
        drop=0.0,
        bias=True,
    ):
        super().__init__()
        output_features = output_features if output_features is not None else input_features
        hidden_features = hidden_features if hidden_features is not None else input_features
        initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        self.fully_connected_layer_1 = keras.layers.Dense(
            units=hidden_features, use_bias=bias, name="fully_connected_layer_1", kernel_initializer=initializer
        )
        self.activation = activation_layer if activation_layer is not None else keras.layers.Activation("gelu")
        self.fully_connected_layer_2 = keras.layers.Dense(
            units=output_features, use_bias=bias, name="fully_connected_layer_2", kernel_initializer=initializer
        )
        self.drop_layer = keras.layers.Dropout(rate=drop, name="drop")

    def call(self, x, training=None):
        x = self.fully_connected_layer_1(x)
        x = self.activation(x)
        x = self.drop_layer(x, training=training)
        x = self.fully_connected_layer_2(x)
        x = self.drop_layer(x, training=training)
        return x
