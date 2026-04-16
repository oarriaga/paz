import keras
from keras import ops
from keras.layers import Layer


@keras.saving.register_keras_serializable(package="gemma4")
class Gemma4RMSNormalization(Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def get_config(self):
        config = super().get_config()
        config["epsilon"] = self.epsilon
        return config

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True,
        )
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        x = ops.cast(x, "float32")
        scale = ops.cast(self.scale, "float32")
        variance = ops.mean(
            ops.square(x), axis=-1, keepdims=True)
        output = x * ops.power(variance + self.epsilon, -0.5)
        output = output * scale
        return ops.cast(output, self.compute_dtype)


@keras.saving.register_keras_serializable(package="gemma4")
class Gemma4VNorm(Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def get_config(self):
        config = super().get_config()
        config["epsilon"] = self.epsilon
        return config

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        x = ops.cast(x, "float32")
        variance = ops.mean(
            ops.square(x), axis=-1, keepdims=True)
        output = x * ops.power(variance + self.epsilon, -0.5)
        return ops.cast(output, self.compute_dtype)


@keras.saving.register_keras_serializable(package="gemma4")
class ScalarMultiply(Layer):
    def get_config(self):
        return super().get_config()

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(),
            initializer="ones",
            trainable=False,
        )
        self.built = True

    def call(self, x):
        return x * ops.cast(self.scale, x.dtype)


def build_rms_norm(epsilon, dtype, name):
    return Gemma4RMSNormalization(
        epsilon=epsilon, dtype=dtype, name=name)


def build_v_norm(epsilon, dtype, name):
    return Gemma4VNorm(epsilon=epsilon, dtype=dtype, name=name)


def build_scalar_multiply(name):
    return ScalarMultiply(name=name)
