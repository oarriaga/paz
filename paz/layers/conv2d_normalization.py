from keras import ops
import keras


@keras.saving.register_keras_serializable()
class Conv2DNormalization(keras.layers.Layer):
    """Normalization layer as described in ParseNet paper.

    # Arguments
        scale: Float determining how much to scale the features.
        axis: Integer specifying axis of image channels.

    # Returns
        Feature map tensor normalized with an L2 norm and then scaled.

    # References
        - [ParseNet: Looking Wider to See Better](
        https://arxiv.org/abs/1506.04579)
    """

    def __init__(self, scale, axis=3, **kwargs):
        self.scale = scale
        self.axis = axis
        super(Conv2DNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(input_shape[self.axis],),
            initializer=keras.initializers.Constant(self.scale),
            trainable=True,
        )

    def output_shape(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        norm = ops.norm(x, ord=2, axis=self.axis, keepdims=True)
        return self.gamma * (x / norm)

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale, "axis": self.axis})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
