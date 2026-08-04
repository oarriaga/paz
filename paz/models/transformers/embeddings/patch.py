"""Patch-grid spatial positions for two-dimensional rotary embeddings."""
from keras import ops


def build_patch_positions(height, width):
    rows = ops.arange(height, dtype="int32")
    columns = ops.arange(width, dtype="int32")
    grid_rows = ops.repeat(rows, width)
    grid_columns = ops.tile(columns, (height,))
    return ops.stack([grid_rows, grid_columns], axis=-1)
