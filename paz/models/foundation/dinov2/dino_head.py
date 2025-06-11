import keras
from keras import layers, initializers, constraints
import keras.ops as ops


class WeightNormConstraint(constraints.Constraint):
    """Constraint that implements weight normalization similar to PyTorch's weight_norm."""

    def __call__(self, w):
        # Normalize along all axes except the last (output) dimension
        norm = ops.sqrt(ops.sum(ops.square(w), axis=0, keepdims=True))
        return ops.divide(w, ops.add(norm, 1e-12))


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

        # Apply weight normalization constraint to match PyTorch's weight_norm
        self.last_layer = layers.Dense(
            out_dim,
            use_bias=False,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            kernel_constraint=WeightNormConstraint(),
        )

        # Store the g parameter (magnitude) separately like PyTorch
        self.weight_g = self.add_weight(name="weight_g", shape=(out_dim,), initializer="ones", trainable=True)

    def call(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == "float16" else 1e-12

        # L2 normalization (equivalent to nn.functional.normalize)
        norm = ops.sqrt(ops.sum(ops.square(x), axis=-1, keepdims=True))
        x = ops.divide(x, ops.add(norm, eps))

        # Apply last layer with weight normalization
        x = self.last_layer(x)

        # Apply the magnitude scaling (g parameter)
        x = x * self.weight_g

        return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    """Build MLP layers for DINOHead in Keras with proper weight initialization."""
    if nlayers == 1:
        return layers.Dense(
            bottleneck_dim,
            use_bias=bias,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            bias_initializer="zeros",
        )
    else:
        mlp_layers = []

        # First layer
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

        # Hidden layers
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

        # Final layer
        mlp_layers.append(
            layers.Dense(
                bottleneck_dim,
                use_bias=bias,
                kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
                bias_initializer="zeros",
            )
        )

        return keras.Sequential(mlp_layers)
