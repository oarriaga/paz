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
        """Compute sinusoidal position encodings from a boolean mask.

        Args:
            mask (Tensor): Boolean mask of shape (B, H, W).
            align_dim_orders (bool): If True, return (H, W, B, C);
                otherwise return (B, H, W, C).

        Returns:
            Tensor: Position encodings.
        """
        not_mask = ops.cast(ops.logical_not(mask), "float32")
        y_embed = ops.cumsum(not_mask, axis=1)
        x_embed = ops.cumsum(not_mask, axis=2)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = ops.cast(ops.arange(self.num_pos_feats), "float32")
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = ops.expand_dims(x_embed, axis=-1) / dim_t
        pos_y = ops.expand_dims(y_embed, axis=-1) / dim_t

        # Interleave sin/cos components
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
            pos = ops.concatenate([pos_y, pos_x], axis=3)
            pos = ops.transpose(pos, (1, 2, 0, 3))
        else:
            pos = ops.concatenate([pos_y, pos_x], axis=3)
        return pos

    def call(self, tensor_or_mask, mask=None, align_dim_orders=False):
        """Compute position encodings.

        Supports three calling conventions:
            1. call(tensors, mask=mask) - explicit tensors and mask.
            2. call((tensors, mask)) - tuple input.
            3. call(mask) - mask-only export mode.

        Args:
            tensor_or_mask: Either a mask tensor or a (tensors, mask) tuple.
            mask (Tensor): Optional boolean mask of shape (B, H, W).
            align_dim_orders (bool): Output dimension ordering.

        Returns:
            Tensor: Position encodings.
        """
        if mask is not None:
            return self._compute_pos(mask, align_dim_orders=align_dim_orders)
        elif isinstance(tensor_or_mask, (tuple, list)) and len(tensor_or_mask) == 2:
            _, mask = tensor_or_mask
            return self._compute_pos(mask, align_dim_orders=align_dim_orders)
        else:
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
    """Create a position encoding layer.

    Args:
        hidden_dim (int): Total hidden dimension (split across two axes).
        position_embedding (str): Encoding type, either 'sine' or 'v2'.

    Returns:
        PositionEmbeddingSine: Configured position encoding layer.

    Raises:
        ValueError: If position_embedding type is not supported.
    """
    N_steps = hidden_dim // 2
    if position_embedding in ("v2", "sine"):
        position_embedding = PositionEmbeddingSine(
            N_steps, normalize=True, name="position_embedding_sine"
        )
    else:
        raise ValueError(f"not supported {position_embedding}")
    return position_embedding
