import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant


class Conv2DNormalization(Layer):
    """Normalization layer as described in ParseNet paper.
    # Arguments
        scale: Float determining how much to scale the features.
        axis: Integer specifying axis of image channels.

    # Returns
        Feature map tensor normalized with an L2 norm and then scaled.

    # References
        http://cs.unc.edu/~wliu/papers/parsenet.pdf
    """
    def __init__(self, scale, axis=3, **kwargs):
        self.scale = scale
        self.axis = axis
        super(Conv2DNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        print('input_shape', input_shape)
        self.gamma = self.add_weight(
            name='gamma', shape=(input_shape[self.axis]),
            initializer=Constant(self.scale), trainable=True)
        super(Conv2DNormalization, self).build(input_shape)

    def call(self, x, mask=None):
        return self.gamma * K.l2_normalize(x, self.axis)
