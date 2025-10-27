from keras.layers import Conv2D, Identity
import keras


def make_2tuple(x):
    if isinstance(x, (list, tuple)):
        assert len(x) == 2, f"Expected tuple of length 2, but got {len(x)}"
        return tuple(x)
    assert isinstance(x, int), f"Expected int or tuple, but got {type(x)}"
    return (x, x)


class PatchEmbed(keras.layers.Layer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        norm_layer=None,
        flatten_embedding=True,
        use_bias=True,
        **kwargs,
    ):
        kwargs.pop("in_chans", None)
        super().__init__(**kwargs)
        self.img_size = make_2tuple(img_size)
        self.patch_size = make_2tuple(patch_size)
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.projection = Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            use_bias=use_bias,
            name="projection",
        )
        self.norm = norm_layer(name="norm") if norm_layer else Identity()

    def call(self, x):
        x = self.projection(x)
        B, H_new, W_new, C_new = keras.ops.shape(x)
        x = keras.ops.reshape(x, (B, H_new * W_new, C_new))
        x = self.norm(x)
        if not self.flatten_embedding:
            x = keras.ops.reshape(x, (B, H_new, W_new, C_new))
        return x
