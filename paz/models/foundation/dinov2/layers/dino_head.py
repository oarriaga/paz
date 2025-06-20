import keras
from keras import layers, initializers
import keras.ops as ops


class WeightNormDense(layers.Layer):
    def __init__(
        self, units, use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.v = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            trainable=True,
            name="v",
        )

        self.g = self.add_weight(shape=(self.units,), initializer="ones", trainable=True, name="g")

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                trainable=True,
                name="bias",
            )
        else:
            self.bias = None

        self.built = True

    def call(self, inputs):
        v_norm = ops.sqrt(ops.sum(ops.square(self.v), axis=0, keepdims=True))
        v_normalized = self.v / ops.maximum(v_norm, 1e-12)

        kernel = self.g * v_normalized

        output = ops.matmul(inputs, kernel)
        if self.use_bias:
            output = ops.add(output, self.bias)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)


class DINOHead(layers.Layer):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias
        )

        self.last_layer = WeightNormDense(
            out_dim,
            use_bias=False,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
        )

    def call(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == "float16" else 1e-12

        x = ops.divide(x, ops.maximum(ops.sqrt(ops.sum(ops.square(x), axis=-1, keepdims=True)), eps))

        x = self.last_layer(x)

        return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return layers.Dense(
            bottleneck_dim,
            use_bias=bias,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            bias_initializer="zeros",
        )
    else:
        mlp_layers = []

        mlp_layers.append(
            layers.Dense(
                hidden_dim,
                use_bias=bias,
                kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
                bias_initializer="zeros",
            )
        )
        if use_bn:
            mlp_layers.append(layers.BatchNormalization())
        mlp_layers.append(layers.Activation("gelu"))

        for _ in range(nlayers - 2):
            mlp_layers.append(
                layers.Dense(
                    hidden_dim,
                    use_bias=bias,
                    kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
                    bias_initializer="zeros",
                )
            )
            if use_bn:
                mlp_layers.append(layers.BatchNormalization())
            mlp_layers.append(layers.Activation("gelu"))

        mlp_layers.append(
            layers.Dense(
                bottleneck_dim,
                use_bias=bias,
                kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
                bias_initializer="zeros",
            )
        )

        return keras.Sequential(mlp_layers)
