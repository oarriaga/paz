"""Image preprocessing for the Gemma4 vision encoder.

Resizes square images to `image_size`, extracts `patch_size` patches into the
flattened `(num_patches, 3*patch**2)` layout the vision encoder expects, and
builds the matching `(x, y)` patch position ids. Pixel values are assumed to be
in [0, 1]; the patch embedder rescales them to [-1, 1].
"""
from keras import ops


def preprocess_images(images, config):
    side = config.image_size
    resized = ops.image.resize(
        images, (side, side), interpolation="bicubic", antialias=True)
    pixel_values = extract_patches(resized, config.patch_size)
    position_ids = build_patch_position_ids(side, config.patch_size)
    batch = ops.shape(images)[0]
    position_ids = ops.broadcast_to(
        position_ids, (batch,) + ops.shape(position_ids)[1:])
    return {"pixel_values": pixel_values, "pixel_position_ids": position_ids}


def extract_patches(images, patch_size):
    shape = ops.shape(images)
    batch, side = shape[0], shape[1]
    count = side // patch_size
    patched = ops.reshape(
        images, (batch, count, patch_size, count, patch_size, 3))
    patched = ops.transpose(patched, (0, 1, 3, 2, 4, 5))
    return ops.reshape(
        patched, (batch, count * count, 3 * patch_size * patch_size))


def build_patch_position_ids(side, patch_size):
    count = side // patch_size
    columns = ops.tile(ops.reshape(ops.arange(count), (1, count)), (count, 1))
    rows = ops.tile(ops.reshape(ops.arange(count), (count, 1)), (1, count))
    columns = ops.reshape(columns, (-1,))
    rows = ops.reshape(rows, (-1,))
    position_ids = ops.stack((columns, rows), axis=-1)
    return ops.expand_dims(ops.cast(position_ids, "int32"), axis=0)
