import keras
from keras import ops


class SeparableConv4D(keras.layers.Layer):
    """Approximates a 4D convolution using two sequential 3D convolutions."""

    def __init__(
        self,
        out_planes,
        stride=(1, 1, 1),
        ksize=3,
        do_padding=True,
        use_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.out_planes = out_planes
        self.stride = stride
        self.ksize = ksize
        self.do_padding = do_padding
        self.use_bias = use_bias

    def build(self, input_shape):
        # The number of input channels is inferred from the input shape.
        # Assumes data_format is 'channels_first'.
        self.in_planes = input_shape[1]
        self.is_proj = self.in_planes != self.out_planes

        # Projection layer to match output channels if necessary.
        # This uses Conv2D as in the original implementation.
        if self.is_proj:
            self.proj = keras.Sequential(
                [
                    keras.layers.Conv2D(
                        filters=self.out_planes,
                        kernel_size=1,
                        use_bias=self.use_bias,
                        padding="valid",
                        data_format="channels_first",
                        name="proj_conv2d",
                    ),
                    keras.layers.BatchNormalization(axis=1, name="proj_bn"),
                ]
            )

        # Padding values for the two 3D convolutions
        padding1_val = (
            (
                (0, 0),
                (self.ksize // 2, self.ksize // 2),
                (self.ksize // 2, self.ksize // 2),
            )
            if self.do_padding
            else ((0, 0), (0, 0), (0, 0))
        )
        padding2_val = (
            (
                (self.ksize // 2, self.ksize // 2),
                (self.ksize // 2, self.ksize // 2),
                (0, 0),
            )
            if self.do_padding
            else ((0, 0), (0, 0), (0, 0))
        )

        # First convolution block (operates on the last two spatial dimensions)
        self.conv1 = keras.Sequential(
            [
                keras.layers.ZeroPadding3D(
                    padding=padding1_val, data_format="channels_first"
                ),
                keras.layers.Conv3D(
                    filters=self.in_planes,
                    kernel_size=(1, self.ksize, self.ksize),
                    strides=self.stride,
                    use_bias=self.use_bias,
                    padding="valid",  # Padding is handled by ZeroPadding3D
                    data_format="channels_first",
                ),
                keras.layers.BatchNormalization(axis=1),
            ]
        )

        # Second convolution block (operates on the first two spatial dimensions)
        self.conv2 = keras.Sequential(
            [
                keras.layers.ZeroPadding3D(
                    padding=padding2_val, data_format="channels_first"
                ),
                keras.layers.Conv3D(
                    filters=self.in_planes,
                    kernel_size=(self.ksize, self.ksize, 1),
                    strides=self.stride,
                    use_bias=self.use_bias,
                    padding="valid",  # Padding is handled by ZeroPadding3D
                    data_format="channels_first",
                ),
                keras.layers.BatchNormalization(axis=1),
            ]
        )

        self.relu = keras.layers.ReLU()

        # Ensure the build method of the parent class is called.
        super().build(input_shape)

    def call(self, inputs):
        """
        Defines the forward pass of the layer.

        Args:
            inputs (tensor): The input tensor of shape (b, c, u, v, h, w).

        Returns:
            A tensor of shape (b, out_planes, u, v, h, w).
        """
        # Get original dimensions. The final reshape relies on these,
        # which is a behavior inherited from the source PyTorch code.
        b, c, u, v, h, w = ops.shape(inputs)

        # Reshape for the second convolution (operates on u, v dims)
        # Input: (b, c, u, v, h, w) -> Reshaped: (b, c, u, v, h*w)
        x = ops.reshape(inputs, (b, c, u, v, h * w))
        x = self.conv2(x)
        x = self.relu(x)

        # After conv2, get the new u and v dimensions
        _, c_out, u_p, v_p, _ = ops.shape(x)

        # Reshape for the first convolution (operates on h, w dims)
        # We use the original h and w, as conv2 did not operate on them.
        x = ops.reshape(x, (b, c_out, u_p * v_p, h, w))
        x = self.conv1(x)

        # Projection if the number of channels needs to be changed
        if self.is_proj:
            # Reshape for the 2D convolution in the projection block
            _, c_out, _, h_p, w_p = ops.shape(x)
            x = ops.reshape(x, (b, c_out, -1, w_p))
            x = self.proj(x)

        # Final reshape back to a 6D tensor.
        # This uses the original spatial dimensions (u, v, h, w), faithfully
        # translating the original PyTorch code's behavior.
        output = ops.reshape(x, (b, self.out_planes, u, v, h, w))

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_planes": self.out_planes,
                "stride": self.stride,
                "ksize": self.ksize,
                "do_padding": self.do_padding,
                "use_bias": self.use_bias,
            }
        )
        return config
