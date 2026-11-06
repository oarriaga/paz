import functools
import math

from keras import ops


def position_embedding_sine(mask, num_pos_feats=64, temperature=10000, normalize=False, scale=None, align_dim_orders=True):  # fmt: skip
    if scale is not None and normalize is False:
        raise ValueError("normalize should be True if scale is passed")
    if scale is None:
        scale = 2 * math.pi
    height = mask.shape[1]
    width = mask.shape[2]
    not_mask = ops.cast(ops.logical_not(mask), "float32")
    y_embed = ops.cumsum(not_mask, axis=1)
    x_embed = ops.cumsum(not_mask, axis=2)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale
    dim_t = ops.cast(ops.arange(num_pos_feats), "float32")
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = ops.expand_dims(x_embed, axis=-1) / dim_t
    pos_y = ops.expand_dims(y_embed, axis=-1) / dim_t
    pos_x = interleave_sin_cos(pos_x, height, width, num_pos_feats)
    pos_y = interleave_sin_cos(pos_y, height, width, num_pos_feats)
    pos = ops.concatenate([pos_y, pos_x], axis=3)
    if align_dim_orders:
        pos = ops.transpose(pos, (1, 2, 0, 3))
    return pos


def interleave_sin_cos(pos, height, width, num_pos_feats):
    sin = ops.sin(pos[:, :, :, 0::2])
    cos = ops.cos(pos[:, :, :, 1::2])
    stacked = ops.stack([sin, cos], axis=4)
    return ops.reshape(stacked, (-1, height, width, num_pos_feats))


def build_position_encoding(hidden_dim, position_embedding):
    if position_embedding not in ("v2", "sine"):
        raise ValueError(f"not supported {position_embedding}")
    keys = ("num_pos_feats", "temperature", "normalize", "scale")
    values = (hidden_dim // 2, 10000, True, 2 * math.pi)
    kwargs = dict(zip(keys, values))
    return functools.partial(position_embedding_sine, **kwargs)
