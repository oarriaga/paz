import keras


class MLP(keras.layers.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=None,
        drop=0.0,
        bias=True,
    ):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features
        initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        self.fc1 = keras.layers.Dense(
            units=hidden_features, use_bias=bias, name="fc1", kernel_initializer=initializer
        )
        self.act = act_layer if act_layer is not None else keras.layers.Activation("gelu")
        self.fc2 = keras.layers.Dense(
            units=out_features, use_bias=bias, name="fc2", kernel_initializer=initializer
        )
        self.drop_layer = keras.layers.Dropout(rate=drop, name="drop")

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop_layer(x, training=training)
        x = self.fc2(x)
        x = self.drop_layer(x, training=training)
        return x
