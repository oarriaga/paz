import keras
from keras import layers
from typing import List, Optional
from paz.models.foundation.dinov3.utils import cat_keep_shapes, uncat_with_shapes


class ListForwardMixin:
    """A mixin to handle forward passes on lists of tensors."""

    def call_list(
        self, x_list: List[keras.KerasTensor], training: bool = False
    ) -> List[keras.KerasTensor]:
        """
        Processes a list of tensors by concatenating, running the forward pass,
        and then splitting them back to their original shapes.
        """
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat_out = self(x_flat, training=training)
        return uncat_with_shapes(x_flat_out, shapes, num_tokens)


class Mlp(layers.Layer, ListForwardMixin):
    def __init__(
        self,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: str = "gelu",
        drop: float = 0.0,
        bias: bool = True,
        approximate_gelu: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_features_config = hidden_features
        self.out_features_config = out_features
        if act_layer == "gelu":
            self.act = lambda x: keras.activations.gelu(x, approximate=approximate_gelu)
        else:
            self.act = keras.activations.get(act_layer)
        self.drop_rate = drop
        self.use_bias = bias

    def build(self, input_shape: tuple):
        in_features = input_shape[-1]
        hidden_features = self.hidden_features_config or in_features
        out_features = self.out_features_config or in_features

        self.fc1 = keras.layers.Dense(
            hidden_features, use_bias=self.use_bias, name="fc1"
        )
        self.fc2 = layers.Dense(out_features, use_bias=self.use_bias, name="fc2")
        self.drop = layers.Dropout(self.drop_rate)
        super().build(input_shape)

    def _forward_tensor(
        self, x: keras.KerasTensor, training: bool = False
    ) -> keras.KerasTensor:
        """Processes a single tensor input."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x

    def call(self, x: keras.KerasTensor, training: bool = False) -> keras.KerasTensor:
        """Handles both a single tensor and a list of tensors."""
        if isinstance(x, list):
            return self.call_list(x, training=training)
        else:
            return self._forward_tensor(x, training=training)


class SwiGLUFFN(layers.Layer, ListForwardMixin):
    """Keras implementation of the SwiGLUFFN module."""

    def __init__(
        self,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
        align_to: int = 8,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.hidden_features_config = hidden_features
        self.out_features_config = out_features
        self.use_bias = bias
        self.align_to = align_to

    def build(self, input_shape: tuple):
        in_features = input_shape[-1]
        out_features = self.out_features_config or in_features
        hidden_features = self.hidden_features_config or in_features

        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = (
            (d + self.align_to - 1) // self.align_to * self.align_to
        )

        self.w1 = layers.Dense(
            swiglu_hidden_features, use_bias=self.use_bias, name="w1"
        )
        self.w2 = layers.Dense(
            swiglu_hidden_features, use_bias=self.use_bias, name="w2"
        )
        self.w3 = layers.Dense(out_features, use_bias=self.use_bias, name="w3")

        super().build(input_shape)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = keras.activations.silu(x1) * x2
        return self.w3(hidden)
