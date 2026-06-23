"""Image preprocessing for the Gemma4 vision encoder.

Resizes images preserving aspect ratio to roughly `max_patches` patches (each
side a multiple of patch_size*pool_size), extracts patches into the flattened
`(num_patches, 3*patch**2)` layout, builds the matching `(x, y)` position ids,
and pads to `max_patches` (pixel values with 0, positions with -1). Pixel values
are assumed to be in [0, 1]; the patch embedder rescales them to [-1, 1].

Run eagerly (concrete image shapes); not part of a functional graph.
"""
from keras import ops


def preprocess_images(images, config):
    resized = resize_to_aspect(images, config)
    pixel_values, position_ids = extract_patches(resized, config.patch_size)
    return pad_to_max_patches(pixel_values, position_ids, config)


def resize_to_aspect(images, config):
    side_mult = config.patch_size * config.pool_size
    target_pixels = config.max_patches * config.patch_size ** 2
    shape = ops.shape(images)
    height, width = shape[1], shape[2]
    total = ops.cast(height * width, "float32")
    factor = ops.sqrt(ops.cast(target_pixels, "float32") / total)
    target_height = round_to_multiple(factor * ops.cast(height, "float32"),
                                      side_mult)
    target_width = round_to_multiple(factor * ops.cast(width, "float32"),
                                     side_mult)
    return ops.image.resize(
        images, (target_height, target_width), interpolation="bicubic",
        antialias=True)


def round_to_multiple(value, multiple):
    rounded = ops.cast(ops.floor(value / multiple) * multiple, "int32")
    return ops.maximum(rounded, multiple)


def extract_patches(images, patch_size):
    shape = ops.shape(images)
    batch, height, width = shape[0], shape[1], shape[2]
    rows, columns = height // patch_size, width // patch_size
    patched = ops.reshape(
        images, (batch, rows, patch_size, columns, patch_size, 3))
    patched = ops.transpose(patched, (0, 1, 3, 2, 4, 5))
    pixel_values = ops.reshape(
        patched, (batch, rows * columns, 3 * patch_size * patch_size))
    position_ids = build_position_grid(rows, columns)
    position_ids = ops.broadcast_to(
        position_ids, (batch,) + tuple(ops.shape(position_ids)[1:]))
    return pixel_values, position_ids


def build_position_grid(rows, columns):
    grid_x = ops.tile(ops.reshape(ops.arange(columns), (1, columns)), (rows, 1))
    grid_y = ops.tile(ops.reshape(ops.arange(rows), (rows, 1)), (1, columns))
    grid = ops.stack(
        (ops.reshape(grid_x, (-1,)), ops.reshape(grid_y, (-1,))), axis=-1)
    return ops.expand_dims(ops.cast(grid, "int32"), axis=0)


def pad_to_max_patches(pixel_values, position_ids, config):
    shape = ops.shape(pixel_values)
    batch, count = shape[0], shape[1]
    pad = config.max_patches - count
    value_pad = ops.zeros((batch, pad, shape[2]), pixel_values.dtype)
    position_pad = -ops.ones((batch, pad, 2), dtype="int32")
    pixel_values = ops.concatenate([pixel_values, value_pad], axis=1)
    position_ids = ops.concatenate([position_ids, position_pad], axis=1)
    return {"pixel_values": pixel_values, "pixel_position_ids": position_ids}
