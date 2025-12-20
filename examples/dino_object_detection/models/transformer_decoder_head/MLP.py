import keras
from keras import layers


class MLP(keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers

        h = [hidden_dim] * (num_layers - 1)
        specs = list(zip([input_dim] + h, h + [output_dim]))

        self.mlp_layers = keras.Sequential()

        for in_d, out_d in specs:
            layer = layers.Dense(units=out_d, use_bias=True)
            layer.build((None, in_d))
            self.mlp_layers.add(layer)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        for i, layer in enumerate(self.mlp_layers.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = keras.activations.relu(x)
        return x
