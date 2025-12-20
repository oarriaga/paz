import keras
from keras import layers
import keras.ops as ops


@keras.saving.register_keras_serializable()
class DepthwiseConvBlock(layers.Layer):
    def __init__(self, dim, layer_scale_init_value=0, **kwargs):  # Fixed default to 0
        super().__init__(**kwargs)
        self.dim = dim
        self.layer_scale_init_value = layer_scale_init_value

        # 1. Depthwise Conv
        self.dwconv = layers.Conv2D(
            filters=dim,
            kernel_size=3,
            padding="same",
            groups=dim,
            data_format="channels_first",
            name="dwconv",
        )

        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")
        self.pwconv1 = layers.Dense(dim, name="pwconv1")
        self.act = layers.Activation("gelu", name="act")

        self.gamma = None
        if layer_scale_init_value > 0:
            self.gamma = self.add_weight(
                name="gamma",
                shape=(dim,),
                initializer=keras.initializers.Constant(layer_scale_init_value),
                trainable=True,
            )

    def call(self, x):
        input_tensor = x

        x = self.dwconv(x)

        x = ops.transpose(x, axes=(0, 2, 3, 1))

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)

        if self.gamma is not None:
            x = x * self.gamma

        x = ops.transpose(x, axes=(0, 3, 1, 2))

        return input_tensor + x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "layer_scale_init_value": self.layer_scale_init_value,
            }
        )
        return config
