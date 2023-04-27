import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class AnchorsLayer(Layer):
    def __init__(self, anchors, name="anchors", **kwargs):
        super().__init__(name=name, **kwargs)
        self.anchors = tf.Variable(anchors, trainable=False)

    def call(self, input_image):
        return self.anchors
