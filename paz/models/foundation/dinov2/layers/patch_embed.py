import keras


def make_2tuple(x):
    if isinstance(x, (list, tuple)):
        assert len(x) == 2
        return tuple(x)
    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(keras.layers.Layer):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)
    Args:
        img_size: Image size.
        patch_size: Patch token size.
        input_channels: Number of input image channels.
        embedding_dimension: Number of linear projection output channels.
        normalization_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        input_channels=3,
        embedding_dimension=768,
        normalization_layer=None,
        flatten_embedding=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )
        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.number_of_patches = patch_grid_size[0] * patch_grid_size[1]
        self.input_channels = input_channels
        self.embedding_dimension = embedding_dimension
        self.flatten_embedding = flatten_embedding

        self.projection_layer = keras.layers.Conv2D(
            filters=embedding_dimension,
            kernel_size=patch_HW,
            strides=patch_HW,
            padding="valid",
            name="proj",
        )
        self.normalize = (
            normalization_layer(embedding_dimension)
            if normalization_layer
            else keras.layers.Identity()
        )

    def build(self, input_shape):
        """Build the internal layers."""
        self.projection_layer.build(input_shape)
        if hasattr(self.normalize, "build") and not self.normalize.built:
            self.normalize.build(
                (input_shape[0], self.number_of_patches, self.embedding_dimension)
            )
        self.built = True

    def call(self, x):
        batch_size = keras.ops.shape(x)[0]
        H = keras.ops.shape(x)[1]
        W = keras.ops.shape(x)[2]

        patch_H, patch_W = self.patch_size

        H = H if isinstance(H, int) else H.numpy() if hasattr(H, "numpy") else H
        W = W if isinstance(W, int) else W.numpy() if hasattr(W, "numpy") else W

        assert (
            H % patch_H == 0
        ), f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert (
            W % patch_W == 0
        ), f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.projection_layer(x)

        H_new = keras.ops.shape(x)[1]
        W_new = keras.ops.shape(x)[2]

        x = keras.ops.reshape(x, (batch_size, H_new * W_new, self.embedding_dimension))

        x = self.normalize(x)

        if not self.flatten_embedding:
            x = keras.ops.reshape(
                x, (batch_size, H_new, W_new, self.embedding_dimension)
            )

        return x
