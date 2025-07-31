import keras
from keras import layers, ops


class SwiGLUFFN(keras.layers.Layer):
    """
    Standard SwiGLU Feed-Forward Network.
    This Class uses a single dense layer and splits the output.
    """

    def __init__(
        self,
        input_features,
        hidden_features=None,
        output_features=None,
        use_bias=True,
        activation_layer=None,
        drop_rate=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.use_bias = use_bias

        self.effective_output_features = (
            output_features if output_features is not None else input_features
        )
        self.effective_hidden_features = (
            hidden_features if hidden_features is not None else input_features
        )

        self.fused_gate_and_value_projection = keras.layers.Dense(
            units=2 * self.effective_hidden_features,
            use_bias=self.use_bias,
            name="fused_gate_and_value_projection",
        )
        self.output_projection = keras.layers.Dense(
            units=self.effective_output_features,
            use_bias=self.use_bias,
            name="output_projection",
        )

        self.activation_layer = (
            activation_layer
            if activation_layer is not None
            else keras.layers.Activation("silu")
        )

    def build(self, input_shape):
        self.fused_gate_and_value_projection.build(input_shape)
        output_proj_input_shape = (
            input_shape[0],
            input_shape[1],
            self.effective_hidden_features,
        )
        self.output_projection.build(output_proj_input_shape)
        self.built = True

    def call(self, x, training=None):
        gate_and_value = self.fused_gate_and_value_projection(x)
        value, gate = ops.split(gate_and_value, 2, axis=-1)
        hidden = self.activation_layer(value) * gate
        return self.output_projection(hidden)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_features": self.input_features,
                "hidden_features": self.hidden_features,
                "output_features": self.output_features,
                "use_bias": self.use_bias,
            }
        )
        return config


class SwiGLUFFNFused(keras.layers.Layer):
    """
    Fused SwiGLU Feed-Forward Network.
    Similar to the standard Class but with specific hidden feature sizing.
    """

    def __init__(
        self,
        input_features,
        hidden_features=None,
        output_features=None,
        use_bias=True,
        activation_layer=None,
        drop_rate=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.use_bias = use_bias
        self.activation_layer = keras.layers.Activation("silu")

        self.effective_output_features = (
            output_features if output_features is not None else input_features
        )
        effective_hidden_features = (
            hidden_features if hidden_features is not None else input_features
        )
        self.effective_hidden_features = (
            (int(effective_hidden_features * 2 / 3) + 7) // 8 * 8
        )

        self.fused_gate_and_value_projection = keras.layers.Dense(
            units=2 * self.effective_hidden_features,
            use_bias=use_bias,
            name="fused_gate_and_value_projection",
        )
        self.output_projection = keras.layers.Dense(
            units=self.effective_output_features,
            use_bias=use_bias,
            name="output_projection",
        )
        self.drop_layer = keras.layers.Dropout(rate=drop_rate)

        self.activation_layer = keras.layers.Activation("silu")

    def build(self, input_shape):
        self.fused_gate_and_value_projection.build(input_shape)
        output_proj_input_shape = (
            input_shape[0],
            input_shape[1],
            self.effective_hidden_features,
        )
        self.output_projection.build(output_proj_input_shape)
        self.built = True

    def call(self, x, training=None):
        gate_and_value = self.fused_gate_and_value_projection(x)
        value, gate = ops.split(gate_and_value, 2, axis=-1)
        hidden = self.activation_layer(value) * gate

        output = self.output_projection(hidden)
        return self.drop_layer(output, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_features": self.input_features,
                "hidden_features": self.hidden_features,
                "output_features": self.output_features,
                "use_bias": self.use_bias,
            }
        )
        return config


class SwiGLUFFNAligned(keras.layers.Layer):
    """
    Aligned SwiGLU Feed-Forward Network.
    This Class uses two separate dense layers for the initial projection.
    """

    def __init__(
        self,
        input_features,
        hidden_features=None,
        output_features=None,
        use_bias=True,
        align_to=8,
        activation_layer=None,
        drop_rate=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.use_bias = use_bias
        self.align_to = align_to

        self.effective_output_features = (
            output_features if output_features is not None else input_features
        )
        effective_hidden_features = (
            hidden_features if hidden_features is not None else input_features
        )

        d = int(effective_hidden_features * 2 / 3)
        self.swiglu_hidden_features = d + (-d % align_to)

        self.value_projection = layers.Dense(
            self.swiglu_hidden_features, use_bias=use_bias, name="value_projection"
        )
        self.gate_projection = layers.Dense(
            self.swiglu_hidden_features, use_bias=use_bias, name="gate_projection"
        )
        self.output_projection = layers.Dense(
            self.effective_output_features, use_bias=use_bias, name="output_projection"
        )
        self.activation_layer = (
            activation_layer
            if activation_layer is not None
            else keras.layers.Activation("silu")
        )

    def call(self, x, training=None):
        value = self.value_projection(x)
        gate = self.gate_projection(x)
        hidden = self.activation_layer(value) * gate
        return self.output_projection(hidden)

    def build(self, input_shape):
        self.value_projection.build(input_shape)
        self.gate_projection.build(input_shape)
        output_proj_input_shape = (
            input_shape[0],
            input_shape[1],
            self.swiglu_hidden_features,
        )
        self.output_projection.build(output_proj_input_shape)
        self.built = True

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_features": self.input_features,
                "hidden_features": self.hidden_features,
                "output_features": self.output_features,
                "use_bias": self.use_bias,
                "align_to": self.align_to,
            }
        )
        return config
