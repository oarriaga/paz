import keras


class ReduceMean(keras.layers.Layer):
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
        return keras.ops.mean(x, axis=self.axes, keepdims=self.keepdims)
