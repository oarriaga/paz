import keras


class MLP(keras.Model):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer_cls=keras.layers.Activation,
        drop=0.0,
        bias=True,
    ):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features

        self.fc1 = keras.layers.Dense(units=hidden_features, use_bias=bias, name="fc1")
        self.act = act_layer_cls(activation=keras.activations.gelu, name="act")
        self.fc2 = keras.layers.Dense(units=out_features, use_bias=bias, name="fc2")
        self.drop_layer = keras.layers.Dropout(rate=drop, name="drop")

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop_layer(x)
        x = self.fc2(x)
        x = self.drop_layer(x)
        return x
