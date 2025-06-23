import keras
from keras import layers, ops


class SwiGLUFFN(keras.layers.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias=True,
        act_layer=None,
        drop=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.bias = bias

        self.effective_out_features = out_features if out_features is not None else in_features
        self.effective_hidden_features = hidden_features if hidden_features is not None else in_features

        self.w12 = keras.layers.Dense(units=2 * self.effective_hidden_features, use_bias=bias, name="w12")
        self.w3 = keras.layers.Dense(units=self.effective_out_features, use_bias=bias, name="w3")

        self.act = act_layer if act_layer is not None else keras.layers.Activation("silu")

    def call(self, x, training=None):
        x12 = self.w12(x)
        x1, x2 = ops.split(x12, 2, axis=-1)
        hidden = self.act(x1) * x2
        return self.w3(hidden)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_features": self.in_features,
                "hidden_features": self.hidden_features,
                "out_features": self.out_features,
                "bias": self.bias,
            }
        )
        return config


class SwiGLUFFNFused(keras.layers.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias=True,
        act_layer=None,
        drop=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.bias = bias

        self.effective_out_features = out_features if out_features is not None else in_features
        effective_hidden_features = hidden_features if hidden_features is not None else in_features
        self.effective_hidden_features = (int(effective_hidden_features * 2 / 3) + 7) // 8 * 8

        self.w12 = keras.layers.Dense(units=2 * self.effective_hidden_features, use_bias=bias, name="w12")
        self.w3 = keras.layers.Dense(units=self.effective_out_features, use_bias=bias, name="w3")
        self.act = act_layer if act_layer is not None else keras.layers.Activation("silu")

    def call(self, x, training=None):
        x12 = self.w12(x)
        x1, x2 = ops.split(x12, 2, axis=-1)
        hidden = self.act(x1) * x2
        return self.w3(hidden)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_features": self.in_features,
                "hidden_features": self.hidden_features,
                "out_features": self.out_features,
                "bias": self.bias,
            }
        )
        return config


class SwiGLUFFNAligned(keras.layers.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias=True,
        align_to=8,
        act_layer=None,
        drop=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.bias = bias
        self.align_to = align_to

        self.effective_out_features = out_features if out_features is not None else in_features
        effective_hidden_features = hidden_features if hidden_features is not None else in_features

        d = int(effective_hidden_features * 2 / 3)
        self.swiglu_hidden_features = d + (-d % align_to)

        self.w1 = layers.Dense(self.swiglu_hidden_features, use_bias=bias, name="w1")
        self.w2 = layers.Dense(self.swiglu_hidden_features, use_bias=bias, name="w2")
        self.w3 = layers.Dense(self.effective_out_features, use_bias=bias, name="w3")
        self.act = act_layer if act_layer is not None else keras.layers.Activation("silu")


    def call(self, x, training=None):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        return self.w3(hidden)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_features": self.in_features,
                "hidden_features": self.hidden_features,
                "out_features": self.out_features,
                "bias": self.bias,
                "align_to": self.align_to,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
