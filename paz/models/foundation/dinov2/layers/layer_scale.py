import keras
import keras.ops as ops


class LayerScale(keras.layers.Layer):
    def __init__(self, dimension, init_values=1e-5, inplace=False, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.dimension = dimension
        self.init_values = init_values
        self.inplace = inplace

        self.gamma = self.add_weight(
            name="gamma",
            shape=(self.dimension,),
            initializer=keras.initializers.Constant(self.init_values),
            trainable=True,
        )

    def call(self, x):
        return ops.multiply(x, self.gamma)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dimension": self.dimension,
                "init_values": self.init_values,
                "inplace": self.inplace,
            }
        )
        return config
