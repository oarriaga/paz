"""View routing between per-view (local) and cross-view (global) layouts.

Tokens carry an explicit view axis as ``(batch, views, tokens, channels)``.
Local blocks fold views into the batch so each view attends to itself; global
blocks merge views into one sequence so all views attend jointly.
"""
import numpy as np
from keras import ops


def fold_view_images(images):
    height, width, channels = images.shape[-3:]
    return ops.reshape(images, (-1, height, width, channels))


def fold_views_into_batch(tokens):
    channels = tokens.shape[-1]
    num_tokens = tokens.shape[-2]
    return ops.reshape(tokens, (-1, num_tokens, channels))


def restore_view_dimension(tokens, num_views):
    channels = tokens.shape[-1]
    num_tokens = tokens.shape[-2]
    return ops.reshape(tokens, (-1, num_views, num_tokens, channels))


def merge_views_into_sequence(tokens):
    channels = tokens.shape[-1]
    num_views = tokens.shape[-3]
    num_tokens = tokens.shape[-2]
    return ops.reshape(tokens, (-1, num_views * num_tokens, channels))


def split_sequence_into_views(tokens, num_views):
    channels = tokens.shape[-1]
    num_tokens = tokens.shape[-2] // num_views
    return ops.reshape(tokens, (-1, num_views, num_tokens, channels))


def build_local_positions(grid_height, grid_width):
    grid = build_grid_positions(grid_height, grid_width) + 1
    special = np.zeros((1, 2), "int32")
    positions = np.concatenate([special, grid], axis=0)
    return ops.array(positions[None])


def build_global_positions(num_views, grid_height, grid_width):
    num_patches = grid_height * grid_width
    special = np.zeros((1, 2), "int32")
    patches = np.ones((num_patches, 2), "int32")
    per_view = np.concatenate([special, patches], axis=0)
    positions = np.tile(per_view, (num_views, 1))
    return ops.array(positions[None])


def build_grid_positions(grid_height, grid_width):
    rows = np.repeat(np.arange(grid_height), grid_width)
    columns = np.tile(np.arange(grid_width), grid_height)
    return np.stack([rows, columns], axis=-1).astype("int32")
