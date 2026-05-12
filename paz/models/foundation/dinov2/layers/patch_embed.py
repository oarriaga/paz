import types
import keras
from keras import ops


def make_2tuple(x):
    if isinstance(x, (list, tuple)):
        assert len(x) == 2
        return tuple(x)
    assert isinstance(x, int)
    return (x, x)


def build_projection_layer(patch_HW, dimension):
    kw = {"filters": dimension, "kernel_size": patch_HW, "strides": patch_HW}
    kw.update({"padding": "valid", "name": "proj"})
    return keras.layers.Conv2D(**kw)


def build_normalize_layer(normalization_layer, dimension):
    if normalization_layer:
        return normalization_layer(dimension)
    return keras.layers.Identity()


def build_patch_embed_layers(patch_HW, dimension, normalization_layer):
    proj = build_projection_layer(patch_HW, dimension)
    norm = build_normalize_layer(normalization_layer, dimension)
    return proj, norm


def set_all_attributes(model, image_HW, patch_HW, nc, dim, flat, proj, norm):
    grid = image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1]
    model.patches_resolution = grid
    model.number_of_patches = grid[0] * grid[1]
    model.img_size = image_HW
    model.patch_size = patch_HW
    model.input_channels = nc
    model.embedding_dimension = dim
    model.flatten_embedding = flat
    model.projection_layer = proj
    model.normalize = norm


def validate_patch_dims(H, W, patch_size):
    patch_H, patch_W = patch_size
    H = H if isinstance(H, int) else H.numpy() if hasattr(H, "numpy") else H
    W = W if isinstance(W, int) else W.numpy() if hasattr(W, "numpy") else W
    assert H % patch_H == 0, f"H {H} not multiple of patch_H {patch_H}"
    assert W % patch_W == 0, f"W {W} not multiple of patch_W {patch_W}"


def flatten_projected(x, batch_size, dimension):
    H_new, W_new = ops.shape(x)[1], ops.shape(x)[2]
    return H_new, W_new, ops.reshape(x, (batch_size, H_new * W_new, dimension))


def apply_patch_embed(self, x, training=None, **_):
    batch_size = ops.shape(x)[0]
    validate_patch_dims(ops.shape(x)[1], ops.shape(x)[2], self.patch_size)
    args = self.projection_layer(x), batch_size, self.embedding_dimension
    H_new, W_new, x = flatten_projected(*args)
    x = self.normalize(x)
    if not self.flatten_embedding:
        shape = batch_size, H_new, W_new, self.embedding_dimension
        x = ops.reshape(x, shape)
    return x


def PatchEmbed(
    img_size=224,
    patch_size=16,
    input_channels=3,
    embedding_dimension=768,
    normalization_layer=None,
    flatten_embedding=True,
    **kwargs,
):
    image_HW = make_2tuple(img_size)
    patch_HW = make_2tuple(patch_size)
    dim, flat = embedding_dimension, flatten_embedding
    proj, norm = build_patch_embed_layers(patch_HW, dim, normalization_layer)
    x_in = keras.Input(shape=(None, None, input_channels))
    model = keras.Model(inputs=x_in, outputs=proj(x_in), **kwargs)
    args = model, image_HW, patch_HW, input_channels, dim, flat, proj, norm
    set_all_attributes(*args)
    model.call = types.MethodType(apply_patch_embed, model)
    return model
