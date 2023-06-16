import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class AnchorsLayer(Layer):
    """Computes a set of anchor boxes as a variable function in tensorflow
     and provide them as output when the anchor layer class is called.

    # Arguments:
        anchors: matched to the shape of the input image [N, (x1, y1, x2, y2)]

    # Returns:
        anchors: [N, (x1, y1, x2, y2)]

    """
    def __init__(self, anchors, name="anchors", **kwargs):
        super().__init__(name=name, **kwargs)
        self.anchors = tf.Variable(anchors, trainable=False)

    def call(self, input_image):
        return self.anchors
