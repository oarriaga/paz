import keras
from keras import ops


@keras.saving.register_keras_serializable()
class RMSNorm(keras.Layer):
    def __init__(self, epsilon: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[-1]
        self.weight = self.add_weight(
            shape=(dim,),
            initializer="ones",
            trainable=True,
            name="weight",
        )
        super().build(input_shape)

    def call(self, x):
        x_dtype = x.dtype
        x_f32 = ops.cast(x, "float32")

        variance = ops.mean(ops.square(x_f32), axis=-1, keepdims=True)
        norm_x = x_f32 * ops.rsqrt(variance + self.epsilon)

        norm_x = ops.cast(norm_x, x_dtype)
        return norm_x * self.weight

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config
