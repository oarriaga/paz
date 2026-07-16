"""Small serializable layers that own SAM 2 weights not covered by Keras.

``ChannelBias`` adds a learned per-channel vector, used for ``no_mem_embed``
and for the mask decoder's dense no-mask embedding.
"""
import keras
from keras import ops
from keras.layers import Layer


@keras.saving.register_keras_serializable(package="paz")
class BroadcastTokens(Layer):
    def __init__(self, count, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.count = count
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.tokens = self.add_weight(
            name="tokens", shape=(self.count, self.hidden_size),
            initializer="zeros")

    def call(self, reference):
        batch = ops.shape(reference)[0]
        return ops.broadcast_to(
            self.tokens[None], (batch, self.count, self.hidden_size))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.count, self.hidden_size)

    def get_config(self):
        arguments = dict(count=self.count, hidden_size=self.hidden_size)
        return {**super().get_config(), **arguments}


@keras.saving.register_keras_serializable(package="paz")
class ChannelBias(Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="bias", shape=(self.channels,), initializer="zeros")

    def call(self, x):
        return x + self.bias

    def get_config(self):
        return {**super().get_config(), "channels": self.channels}
