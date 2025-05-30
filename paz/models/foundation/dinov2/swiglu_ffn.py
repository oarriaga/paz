import keras
import keras.ops as ops
from typing import Optional, Callable


class SwiGLUFFN(keras.layers.Layer):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable] = None,
        drop: float = 0.0,
        bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.effective_out_features = out_features if out_features is not None else in_features
        self.effective_hidden_features = hidden_features if hidden_features is not None else in_features

        self.act_layer_param = act_layer
        self.drop_param = drop
        self.bias_param = bias

        self.w12 = keras.layers.Dense(units=2 * self.effective_hidden_features, use_bias=bias, name="w12")
        self.w3 = keras.layers.Dense(units=self.effective_out_features, use_bias=bias, name="w3")

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        x12 = self.w12(x)
        x1, x2 = ops.split(x12, num_or_size_splits=2, axis=-1)

        hidden = ops.silu(x1) * x2
        return self.w3(hidden)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_features": self.in_features,
                "hidden_features": self.effective_hidden_features,
                "out_features": self.effective_out_features,
                "bias": self.bias_param,
                "drop": self.drop_param,
            }
        )
        return config
