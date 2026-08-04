"""Learnable tokens as a serializable weight, without Lambda layers.

Holds a ``(count, hidden_size)`` weight and broadcasts it to the batch of a
reference tensor. Used for class, register, position, and camera tokens so the
functional models stay serializable.
"""
import keras
from keras import ops
from keras.layers import Layer


@keras.saving.register_keras_serializable(package="paz")
class LearnableTokens(Layer):
    def __init__(self, count, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.count = count
        self.hidden_size = hidden_size

    def build(self, input_shape):
        shape = (self.count, self.hidden_size)
        arguments = dict(name="tokens", shape=shape, initializer="zeros")
        self.tokens = self.add_weight(**arguments)

    def call(self, reference):
        return ops.zeros_like(reference[:, :self.count, :]) + self.tokens

    def get_config(self):
        config = super().get_config()
        config.update(count=self.count, hidden_size=self.hidden_size)
        return config
