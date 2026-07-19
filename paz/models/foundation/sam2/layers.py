"""Small serializable layers that own SAM 2 weights not covered by Keras.

``ChannelBias`` adds a learned per-channel vector, used for ``no_mem_embed``
and the dense no-mask embedding. ``BroadcastTokens`` broadcasts a learned
token table across the batch of a reference tensor.
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
        shape = (self.count, self.hidden_size)
        kwargs = dict(name="tokens", shape=shape, initializer="zeros")
        self.tokens = self.add_weight(**kwargs)

    def call(self, reference):
        batch = ops.shape(reference)[0]
        shape = (batch, self.count, self.hidden_size)
        return ops.broadcast_to(self.tokens[None], shape)

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
        shape = (self.channels,)
        kwargs = dict(name="bias", shape=shape, initializer="zeros")
        self.bias = self.add_weight(**kwargs)

    def call(self, x):
        return x + self.bias

    def get_config(self):
        return {**super().get_config(), "channels": self.channels}


@keras.saving.register_keras_serializable(package="paz")
class ChannelScale(Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        shape = (self.channels,)
        kwargs = dict(name="scale", shape=shape, initializer="ones")
        self.scale = self.add_weight(**kwargs)

    def call(self, x):
        return x * self.scale

    def get_config(self):
        return {**super().get_config(), "channels": self.channels}
