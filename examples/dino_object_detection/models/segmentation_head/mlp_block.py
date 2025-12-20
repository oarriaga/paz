import keras
from keras import layers, ops


class MLPBlock(keras.layers.Layer):
    def __init__(self, dim, layer_scale_init_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.layer_scale_init_value = layer_scale_init_value

        # 1. LayerNorm (PyTorch default eps is 1e-5)
        self.norm_in = layers.LayerNormalization(epsilon=1e-5)

        # 2. MLP Layers
        self.fc1 = layers.Dense(dim * 4)
        self.act = layers.Activation("gelu")
        self.fc2 = layers.Dense(dim)

        # 3. Gamma parameter (LayerScale)
        self.gamma = None

    def build(self, input_shape):
        if self.layer_scale_init_value > 0:
            self.gamma = self.add_weight(
                name="gamma",
                shape=(self.dim,),
                initializer=keras.initializers.Constant(self.layer_scale_init_value),
                trainable=True,
            )
        super().build(input_shape)

    def call(self, x):
        input_tensor = x

        # Pre-Norm
        x = self.norm_in(x)

        # MLP Block
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        # Layer Scale
        if self.gamma is not None:
            x = x * self.gamma

        # Residual Connection
        return x + input_tensor
