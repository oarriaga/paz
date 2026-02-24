import math

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="backbone")
class PositionEmbeddingSine(layers.Layer):
    """
    Sine/cosine positional encoding generalized to work on images.
    Identical logic to the PyTorch DETR/DINO PositionEmbeddingSine.
    """

    def __init__(
        self,
        num_pos_feats=64,
        temperature=10000,
        normalize=False,
        scale=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self._export = False

    def _compute_pos(self, mask, align_dim_orders=True):
        """Core position computation from a boolean mask (B, H, W)."""
        not_mask = ops.cast(ops.logical_not(mask), "float32")
        y_embed = ops.cumsum(not_mask, axis=1)
        x_embed = ops.cumsum(not_mask, axis=2)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = ops.cast(ops.arange(self.num_pos_feats), "float32")
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = ops.expand_dims(x_embed, axis=-1) / dim_t  # (B, H, W, D)
        pos_y = ops.expand_dims(y_embed, axis=-1) / dim_t

        # Interleave sin/cos: stack along new dim then flatten
        pos_x = ops.stack(
            [ops.sin(pos_x[:, :, :, 0::2]), ops.cos(pos_x[:, :, :, 1::2])],
            axis=4,
        )
        pos_x = ops.reshape(pos_x, ops.shape(pos_x)[:3] + (-1,))

        pos_y = ops.stack(
            [ops.sin(pos_y[:, :, :, 0::2]), ops.cos(pos_y[:, :, :, 1::2])],
            axis=4,
        )
        pos_y = ops.reshape(pos_y, ops.shape(pos_y)[:3] + (-1,))

        if align_dim_orders:
            # (B, H, W, C) → (H, W, B, C)
            pos = ops.concatenate([pos_y, pos_x], axis=3)
            pos = ops.transpose(pos, (1, 2, 0, 3))
        else:
            # (B, H, W, C)
            pos = ops.concatenate([pos_y, pos_x], axis=3)
            # pos = ops.transpose(pos, (0, 3, 1, 2))
        return pos

    def call(self, tensor_or_mask, mask=None, align_dim_orders=False):
        """
        Forward pass.

        Usage modes:
            1) call(nested_tensor_tuple, align_dim_orders=...) 
               where nested_tensor_tuple = (tensors, mask)
            2) call(mask, align_dim_orders=...)   — export mode
        """
        if mask is not None:
            # Called as call(tensors, mask)
            return self._compute_pos(mask, align_dim_orders=align_dim_orders)
        elif isinstance(tensor_or_mask, (tuple, list)) and len(tensor_or_mask) == 2:
            # Called as call((tensors, mask))
            _, mask = tensor_or_mask
            return self._compute_pos(mask, align_dim_orders=align_dim_orders)
        else:
            # Export mode: tensor_or_mask IS the mask
            return self._compute_pos(
                tensor_or_mask, align_dim_orders=align_dim_orders
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_pos_feats": self.num_pos_feats,
                "temperature": self.temperature,
                "normalize": self.normalize,
                "scale": self.scale,
            }
        )
        return config


def build_position_encoding(hidden_dim, position_embedding):
    """Factory for position encoding layers."""
    N_steps = hidden_dim // 2
    if position_embedding in ("v2", "sine"):
        position_embedding = PositionEmbeddingSine(
            N_steps, normalize=True, name="position_embedding_sine"
        )
    else:
        raise ValueError(f"not supported {position_embedding}")
    return position_embedding
