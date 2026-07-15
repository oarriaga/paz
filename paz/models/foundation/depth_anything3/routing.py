"""View routing between per-view (local) and cross-view (global) layouts.

Tokens carry an explicit view axis as ``(batch, views, tokens, channels)``.
Local blocks fold views into the batch so each view attends to itself; global
blocks merge views into one sequence so all views attend jointly.
"""
import jax.numpy as jp
from keras import ops


def fold_view_images(images):
    batch, views, height, width, channels = images.shape
    return ops.reshape(images, (-1, height, width, channels))


def fold_views_into_batch(tokens):
    batch, views, num_tokens, channels = tokens.shape
    return ops.reshape(tokens, (-1, num_tokens, channels))


def restore_view_dimension(tokens, num_views):
    batch, num_tokens, channels = tokens.shape
    return ops.reshape(tokens, (-1, num_views, num_tokens, channels))


def merge_views_into_sequence(tokens):
    batch, num_views, num_tokens, channels = tokens.shape
    return ops.reshape(tokens, (-1, num_views * num_tokens, channels))


def split_sequence_into_views(tokens, num_views):
    batch, sequence, channels = tokens.shape
    num_tokens = sequence // num_views
    return ops.reshape(tokens, (-1, num_views, num_tokens, channels))


def build_local_positions(grid_height, grid_width):
    grid = build_grid_positions(grid_height, grid_width) + 1
    special = jp.zeros((1, 2), "int32")
    positions = jp.concatenate([special, grid], axis=0)
    return positions[None]


def build_global_positions(num_views, grid_height, grid_width):
    num_patches = grid_height * grid_width
    special = jp.zeros((1, 2), "int32")
    patches = jp.ones((num_patches, 2), "int32")
    per_view = jp.concatenate([special, patches], axis=0)
    positions = jp.tile(per_view, (num_views, 1))
    return positions[None]


def build_grid_positions(grid_height, grid_width):
    rows = jp.repeat(jp.arange(grid_height), grid_width)
    columns = jp.tile(jp.arange(grid_width), grid_height)
    return jp.stack([rows, columns], axis=-1).astype("int32")
