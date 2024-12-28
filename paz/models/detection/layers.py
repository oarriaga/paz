import keras.backend as K
from keras.layers import Layer
from tensorflow.keras.initializers import Constant


class Conv2DNormalization(Layer):
    """Normalization layer as described in ParseNet paper.

    # Arguments
        scale: Float determining how much to scale the features.
        axis: Integer specifying axis of image channels.

    # Returns
        Feature map tensor normalized with an L2 norm and then scaled.

    # References
        - [ParseNet: Looking Wider to
            See Better](https://arxiv.org/abs/1506.04579)
    """
    def __init__(self, scale, axis=3, **kwargs):
        self.scale = scale
        self.axis = axis
        super(Conv2DNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma', shape=(input_shape[self.axis],),
            initializer=Constant(self.scale), trainable=True)

    def output_shape(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        return self.gamma * K.l2_normalize(x, self.axis)
