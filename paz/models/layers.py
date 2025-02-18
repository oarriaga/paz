from keras.layers import Layer
from keras.initializers import Constant
import keras.ops as K
import keras.backend as B
from keras import activations


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
            name="gamma",
            shape=(input_shape[self.axis],),
            initializer=Constant(self.scale),
            trainable=True,
        )

    def output_shape(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        l2_norm = K.sqrt(
            K.sum(K.square(x), axis=self.axis, keepdims=True) + B.epsilon()
        )
        normalized_x = x / l2_norm
        return self.gamma * normalized_x


class ReduceMean(Layer):
    """Wraps tensorflow's `reduce_mean` function into a keras layer.

    # Arguments
        axes: List of integers. Axes along which mean is to be calculated.
        keepdims: Bool, whether to presere the dimension or not.
    """

    def __init__(self, axes=[1, 2], keepdims=True):
        self.axes = axes
        self.keepdims = keepdims
        super(ReduceMean, self).__init__()

    def call(self, x):
        return K.mean(x, axis=self.axes, keepdims=self.keepdims)


class Sigmoid(Layer):
    """Wraps tensorflow's `sigmoid` function into a keras layer."""

    def __init__(self):
        super(Sigmoid, self).__init__()

    def call(self, x):
        return activations.sigmoid(x)


class Add(Layer):
    """Wraps tensorflow's `add` function into a keras layer."""

    def __init__(self):
        super(Add, self).__init__()

    def call(self, x, y):
        return K.add(x, y)
