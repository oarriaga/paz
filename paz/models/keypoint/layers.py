from keras.layers import Layer
from keras import ops


class SubtractScalar(Layer):
    """Subtracts scalar value to tensor.

    # Arguments
        constant: Float. Value to be subtracted to all tensor values.
    """

    def __init__(self, constant, **kwargs):
        self.constant = constant
        super(SubtractScalar, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SubtractScalar, self).build(input_shape)

    def call(self, x):
        return x - self.constant

    def compute_output_shape(self, input_shape):
        return input_shape


class ExpectedDepth(Layer):
    """Calculates the expected depth along ''axes''.
    This layer takes two inputs. First input is a depth estimation tensor.
    Second input is a probability map of the keypoints.
    It multiplies both values and calculates the expected depth.

    # Arguments
        axes: List of integers. Axes for which the expected value
            will be calculated.
    """

    def __init__(self, axes=[2, 3], **kwargs):
        self.axes = axes
        super(ExpectedDepth, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_keypoints = input_shape[0][1]
        super(ExpectedDepth, self).build(input_shape)

    def call(self, x):
        z_volume, uv_volume = x
        z = ops.sum(z_volume * uv_volume, axis=self.axes)
        z = ops.expand_dims(z, -1)
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.num_keypoints, 1)


class ExpectedValue2D(Layer):
    """Calculates the expected value along ''axes''.

    # Arguments
        axes: List of integers. Axes for which the expected value
            will be calculated.
    """

    def __init__(self, axes=[2, 3], **kwargs):
        self.axes = axes
        super(ExpectedValue2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_keypoints = input_shape[1]
        self.feature_map_size = input_shape[2]
        super(ExpectedValue2D, self).build(input_shape)

    def call(self, x):
        range_x, range_y = self.meshgrid(self.feature_map_size)
        expected_x = ops.sum(x * range_x, axis=self.axes)
        expected_y = ops.sum(x * range_y, axis=self.axes)
        keypoints_stack = ops.stack([expected_x, expected_y], -1)
        keypoints = ops.reshape(keypoints_stack, [-1, self.num_keypoints, 2])
        return keypoints

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_keypoints, 2)

    def meshgrid(self, feature_map_size):
        """Returns a meshgrid ranging from [-1, 1] in x, y axes."""
        r = ops.arange(0.5, feature_map_size, 1) / (feature_map_size / 2) - 1
        range_x, range_y = ops.meshgrid(r, -r)
        return ops.cast(range_x, "float32"), ops.cast(range_y, "float32")
