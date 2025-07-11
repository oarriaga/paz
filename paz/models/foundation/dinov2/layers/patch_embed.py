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
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        input_channels=3,
        embedding_dimension=768,
        norm_layer=None,
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
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.input_channels = input_channels
        self.embedding_dimension = embedding_dimension
        self.flatten_embedding = flatten_embedding

        self.proj = keras.layers.Conv2D(
            filters=embedding_dimension, kernel_size=patch_HW, strides=patch_HW, padding="valid", name="proj"
        )
        self.norm = norm_layer(embedding_dimension) if norm_layer else keras.layers.Identity()

    def call(self, x):
        batch_size = keras.ops.shape(x)[0]
        H = keras.ops.shape(x)[1]
        W = keras.ops.shape(x)[2]

        patch_H, patch_W = self.patch_size

        H = H if isinstance(H, int) else H.numpy() if hasattr(H, "numpy") else H
        W = W if isinstance(W, int) else W.numpy() if hasattr(W, "numpy") else W

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)

        H_new = keras.ops.shape(x)[1]
        W_new = keras.ops.shape(x)[2]

        x = keras.ops.reshape(x, (batch_size, H_new * W_new, self.embedding_dimension))

        x = self.norm(x)

        if not self.flatten_embedding:
            x = keras.ops.reshape(x, (batch_size, H_new, W_new, self.embedding_dimension))

        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embedding_dimension * self.input_channels * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embedding_dimension
        return flops
