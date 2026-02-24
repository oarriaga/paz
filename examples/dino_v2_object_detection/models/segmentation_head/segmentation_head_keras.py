import keras
from keras import ops
import numpy as np


def grid_sample(input_tensor, grid, align_corners=False):
    """
    Sample input_tensor at grid coordinates.
    Args:
        input_tensor: (N, C, H, W)
        grid: (N, H_out, W_out, 2) values in [-1, 1]
    Returns:
        output: (N, C, H_out, W_out)
    """
    # Get shapes
    N, C, H, W = ops.shape(input_tensor)
    _, H_out, W_out, _ = ops.shape(grid)

    # Extract x and y coordinates
    x = grid[..., 0]
    y = grid[..., 1]

    # Convert [-1, 1] to pixel coordinates
    if align_corners:
        x = ((x + 1) / 2) * (W - 1)
        y = ((y + 1) / 2) * (H - 1)
    else:
        x = ((x + 1) * W - 1) / 2
        y = ((y + 1) * H - 1) / 2

    # Get corner coordinates
    x0 = ops.floor(x)
    x1 = x0 + 1
    y0 = ops.floor(y)
    y1 = y0 + 1

    # Clip coordinates to be within image bounds
    x0_safe = ops.clip(x0, 0, W - 1)
    x1_safe = ops.clip(x1, 0, W - 1)
    y0_safe = ops.clip(y0, 0, H - 1)
    y1_safe = ops.clip(y1, 0, H - 1)

    # Cast to integer for indexing
    x0_int = ops.cast(x0_safe, "int32")
    x1_int = ops.cast(x1_safe, "int32")
    y0_int = ops.cast(y0_safe, "int32")
    y1_int = ops.cast(y1_safe, "int32")

    # Helper to gather values
    # input_tensor is (N, C, H, W) -> we need to gather at (n, c, y, x)
    # We'll transpose to (N, H, W, C) for easier gathering if needed, or just handle indices

    # Let's use batch-wise gather.
    # Current ops.take or ops.gather support might be tricky for 4D.
    # We can flatten spatial dims.

    # Transpose input to (N, H, W, C) for easier handling
    input_perm = ops.transpose(input_tensor, (0, 2, 3, 1))  # (N, H, W, C)

    # Calculate batch indices
    batch_range = ops.arange(N)
    batch_indices = ops.reshape(batch_range, (N, 1, 1))
    batch_indices = ops.broadcast_to(batch_indices, (N, H_out, W_out))

    # We need to gather from (N, H, W, C) using indices (b, y, x)
    # Since we want to gather along H and W, we can treat (N, H, W) as flat or use advanced indexing logic simulation

    # Implementing bilinear interpolation

    def get_pixel_values(y_idx, x_idx):
        # Stack indices: (N, H_out, W_out, 3) -> (b, y, x)
        # But ops.gather_nd usually expects indices to be strictly int.
        # Construct indices structure

        # Flatten input spatial: input (N, H*W, C)
        flat_input = ops.reshape(input_perm, (N, H * W, C))

        # Calculate flat spatial indices: y * W + x
        flat_indices = y_idx * W + x_idx  # (N, H_out, W_out)

        # Gather logic:
        # We need to pick for each batch b, the rows specified by flat_indices[b]
        # ops.take_along_axis is useful if we flatten batch?

        # Flatten batch: (N * H * W, C)
        super_flat_input = ops.reshape(input_perm, (-1, C))

        # Offset flat_indices by batch * H * W
        batch_offset = ops.arange(N) * (H * W)  # (N,)
        batch_offset = ops.reshape(batch_offset, (N, 1, 1))
        global_indices = flat_indices + batch_offset  # (N, H_out, W_out)

        global_indices_flat = ops.reshape(global_indices, (-1,))

        values = ops.take(
            super_flat_input, global_indices_flat, axis=0
        )  # (N*H_out*W_out, C)
        return ops.reshape(values, (N, H_out, W_out, C))

    Ia = get_pixel_values(y0_int, x0_int)
    Ib = get_pixel_values(y1_int, x0_int)
    Ic = get_pixel_values(y0_int, x1_int)
    Id = get_pixel_values(y1_int, x1_int)

    # Calculate weights
    # x, y are (N, H_out, W_out) float
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # Expand weights to (N, H_out, W_out, 1) to broadcast over C
    wa = ops.expand_dims(wa, -1)
    wb = ops.expand_dims(wb, -1)
    wc = ops.expand_dims(wc, -1)
    wd = ops.expand_dims(wd, -1)

    # Interpolate
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id  # (N, H_out, W_out, C)

    # Transpose back to (N, C, H_out, W_out)
    return ops.transpose(out, (0, 3, 1, 2))


def point_sample(input_tensor, point_coords, **kwargs):
    """
    Replication of torch.nn.functional.grid_sample wrapper for points.
    Args:
        input_tensor: (N, C, H, W)
        point_coords: (N, P, 2) or (N, Hgrid, Wgrid, 2) normalized in [0, 1]
    """
    add_dim = False
    if ops.ndim(point_coords) == 3:
        add_dim = True
        point_coords = ops.expand_dims(point_coords, 2)  # (N, P, 1, 2)

    # Convert [0, 1] to [-1, 1]
    grid = 2.0 * point_coords - 1.0

    # Note: PyTorch padding_mode='border' is equivalent to nearest edge replication (clamp).
    # Our grid_sample uses clamp on coordinates, essentially 'border' mode.
    # PyTorch default align_corners for grid_sample is False (in recent versions, but caller specifies it).
    # In segmentation_head.py check calls:
    # point_sample(..., align_corners=False) is called in get_uncertain_point_coords_with_randomness

    align_corners = kwargs.get("align_corners", False)

    output = grid_sample(input_tensor, grid, align_corners=align_corners)

    if add_dim:
        output = ops.squeeze(output, 3)  # (N, C, P)

    return output


def calculate_uncertainty(logits):
    """
    logits: (R, 1, ...)
    """
    # assert logits.shape[1] == 1
    # gt_class_logits = logits.clone()
    # return -(torch.abs(gt_class_logits))
    return -ops.abs(logits)


def get_uncertain_point_coords_with_randomness(
    coarse_logits,
    uncertainty_func,
    num_points,
    oversample_ratio=3,
    importance_sample_ratio=0.75,
):
    """
    Sample points based on uncertainty.
    """
    # oversample_ratio >= 1
    # importance_sample_ratio <= 1 and >= 0

    N = ops.shape(coarse_logits)[0]  # num_boxes
    num_sampled = int(num_points * oversample_ratio)

    # point_coords = torch.rand(num_boxes, num_sampled, 2)
    point_coords = keras.random.uniform((N, num_sampled, 2), minval=0.0, maxval=1.0)

    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)

    point_uncertainties = uncertainty_func(point_logits)

    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points

    # idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    # Keras/Jax top_k
    # point_uncertainties is (N, 1, P)
    uncertainties_sub = point_uncertainties[:, 0, :]  # (N, P)

    # top_k returns values, indices
    _, idx = ops.top_k(uncertainties_sub, k=num_uncertain_points)

    # Gather coordinates
    # point_coords is (N, num_sampled, 2)
    # idx is (N, num_uncertain_points)

    # We want to gather from point_coords using idx for each batch element.
    # Batched gather logic again.

    # Flatten point_coords to (N * num_sampled, 2)
    # Offset indices
    shift = ops.arange(N) * num_sampled
    shift = ops.expand_dims(shift, -1)  # (N, 1)

    idx_flat = ops.reshape(idx + ops.cast(shift, idx.dtype), (-1,))

    point_coords_flat = ops.reshape(point_coords, (-1, 2))

    selected_points = ops.take(point_coords_flat, idx_flat, axis=0)
    selected_points = ops.reshape(selected_points, (N, num_uncertain_points, 2))

    if num_random_points > 0:
        random_points = keras.random.uniform(
            (N, num_random_points, 2), minval=0.0, maxval=1.0
        )
        selected_points = ops.concatenate([selected_points, random_points], axis=1)

    return selected_points


class DepthwiseConvBlock(keras.Layer):
    """Simplified ConvNeXt block without the MLP subnet.

    Accepts and returns **NCHW** tensors, matching the PyTorch original.
    Internally permutes to NHWC for LayerNorm / Dense, then back.
    """

    def __init__(self, dim, layer_scale_init_value=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.layer_scale_init_value = layer_scale_init_value

        # dwconv: kernel=3, padding=1 (same), groups=dim (depthwise)
        # PyTorch nn.Conv2d operates on NCHW, so we use channels_first.
        self.dwconv = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            padding="same",
            data_format="channels_first",
            depth_multiplier=1,
            use_bias=True,
        )
        self.norm = keras.layers.LayerNormalization(
            epsilon=1e-6, axis=-1
        )  # operates on last axis (C) after NHWC permute
        self.pwconv1 = keras.layers.Dense(dim)  # Linear (dim -> dim)
        self.act = keras.layers.Activation("gelu")

        # Gamma
        if layer_scale_init_value > 0:
            self.gamma = self.add_weight(
                name="gamma",
                shape=(dim,),
                initializer=keras.initializers.Constant(layer_scale_init_value),
                trainable=True,
            )
        else:
            self.gamma = None

    def call(self, x):
        # x: (N, C, H, W) — NCHW
        input_tensor = x
        x = self.dwconv(x)  # NCHW -> NCHW
        x = ops.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = ops.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
        return x + input_tensor


class MLPBlock(keras.Layer):
    def __init__(self, dim, layer_scale_init_value=0, **kwargs):
        super().__init__(**kwargs)
        self.norm_in = keras.layers.LayerNormalization(
            epsilon=1e-5, axis=-1
        )  # PyTorch default eps=1e-5

        self.linear1 = keras.layers.Dense(dim * 4)
        self.act = keras.layers.Activation("gelu")
        self.linear2 = keras.layers.Dense(dim)

        if layer_scale_init_value > 0:
            self.gamma = self.add_weight(
                name="gamma",
                shape=(dim,),
                initializer=keras.initializers.Constant(layer_scale_init_value),
                trainable=True,
            )
        else:
            self.gamma = None

    def call(self, x):
        input_tensor = x
        x = self.norm_in(x)

        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)

        if self.gamma is not None:
            x = self.gamma * x

        return x + input_tensor


class SegmentationHead(keras.Layer):
    def __init__(
        self, in_dim, num_blocks, bottleneck_ratio=1, downsample_ratio=4, **kwargs
    ):
        super().__init__(**kwargs)
        self.downsample_ratio = downsample_ratio

        self.interaction_dim = (
            in_dim // bottleneck_ratio if bottleneck_ratio is not None else in_dim
        )

        self.blocks = [DepthwiseConvBlock(in_dim) for _ in range(num_blocks)]

        if bottleneck_ratio is None:
            self.spatial_features_proj = keras.layers.Identity()
        else:
            # Conv2d 1x1 — operates on NCHW like PyTorch
            self.spatial_features_proj = keras.layers.Conv2D(
                self.interaction_dim,
                kernel_size=1,
                data_format="channels_first",
                use_bias=True,
            )

        self.query_features_block = MLPBlock(in_dim)

        if bottleneck_ratio is None:
            self.query_features_proj = keras.layers.Identity()
        else:
            self.query_features_proj = keras.layers.Dense(self.interaction_dim)

        # Bias parameter (1,)
        # PyTorch: nn.Parameter(torch.zeros(1), requires_grad=True)
        self.bias = self.add_weight(
            name="bias", shape=(1,), initializer="zeros", trainable=True
        )

        self._export_mode = False

    def export(self):
        self._export_mode = True
        # Recursive export calls - in Keras we might just iterate children if they have export
        # But here blocks are in a list, so we manually call it.
        # DepthwiseConvBlock and MLPBlock in original code don't seem to have export() method shown,
        # but the code loops named_modules.
        # In the provided file, only SegmentationHead has export().
        # And it checks `if hasattr(m, "export")`.
        # DepthwiseConvBlock and MLPBlock do not have export method in the provided file.
        pass

    def call(
        self, spatial_features, query_features, image_size=None, skip_blocks=False
    ):
        """
        spatial_features: (B, C, H, W)  — NCHW, matching the PyTorch convention
        query_features: list of (B, N, C) tensors (one per decoder layer)
        image_size: tuple (H, W) or list [H, W]
        """
        # Handle export mode
        if self._export_mode:
            return self.call_export(
                spatial_features, query_features, image_size, skip_blocks
            )

        # spatial_features stays NCHW throughout (matching PyTorch)

        if image_size is not None:
            target_h = image_size[0] // self.downsample_ratio
            target_w = image_size[1] // self.downsample_ratio
            target_size = (target_h, target_w)

            # ops.image.resize expects NHWC — transpose around it
            sf = ops.transpose(spatial_features, (0, 2, 3, 1))
            sf = ops.image.resize(sf, target_size, interpolation="bilinear")
            spatial_features = ops.transpose(sf, (0, 3, 1, 2))

        mask_logits = []
        if not skip_blocks:
            for i in range(len(self.blocks)):
                if i >= len(query_features):
                    break

                block = self.blocks[i]
                qf = query_features[i]

                spatial_features = block(spatial_features)  # NCHW

                s_proj = self.spatial_features_proj(
                    spatial_features
                )  # NCHW (channels_first Conv2D)

                qf_out = self.query_features_block(qf)
                qf_out = self.query_features_proj(qf_out)  # (B, N, C_inter)

                logit = ops.einsum("bchw,bnc->bnhw", s_proj, qf_out)
                mask_logits.append(logit + self.bias)
        else:
            if len(query_features) != 1:
                raise ValueError(
                    "skip_blocks is only supported for length 1 query features"
                )

            qf = query_features[0]
            qf_out = self.query_features_block(qf)
            qf_out = self.query_features_proj(qf_out)

            logit = ops.einsum("bchw,bnc->bnhw", spatial_features, qf_out)
            mask_logits.append(logit + self.bias)

        return mask_logits

    def call_export(
        self, spatial_features, query_features, image_size, skip_blocks=False
    ):
        if len(query_features) != 1:
            raise ValueError(
                "at export time, segmentation head expects exactly one query feature"
            )

        # spatial_features is NCHW

        if image_size is not None:
            target_h = image_size[0] // self.downsample_ratio
            target_w = image_size[1] // self.downsample_ratio
            target_size = (target_h, target_w)

            sf = ops.transpose(spatial_features, (0, 2, 3, 1))
            sf = ops.image.resize(sf, target_size, interpolation="bilinear")
            spatial_features = ops.transpose(sf, (0, 3, 1, 2))

        if not skip_blocks:
            for block in self.blocks:
                spatial_features = block(spatial_features)

        spatial_features_proj = self.spatial_features_proj(spatial_features)

        qf = query_features[0]
        qf_out = self.query_features_block(qf)
        qf_out = self.query_features_proj(qf_out)

        logit = ops.einsum("bchw,bnc->bnhw", spatial_features_proj, qf_out)
        return [logit + self.bias]

    def sparse_call(
        self, spatial_features, query_features, image_size, skip_blocks=False
    ):
        # Emulate sparse_forward — returns list of dicts
        # spatial_features is NCHW

        if image_size is not None:
            target_h = image_size[0] // self.downsample_ratio
            target_w = image_size[1] // self.downsample_ratio
            target_size = (target_h, target_w)

            sf = ops.transpose(spatial_features, (0, 2, 3, 1))
            sf = ops.image.resize(sf, target_size, interpolation="bilinear")
            spatial_features = ops.transpose(sf, (0, 3, 1, 2))

        output_dicts = []

        if not skip_blocks:
            for i in range(len(self.blocks)):
                if i >= len(query_features):
                    break
                block = self.blocks[i]
                qf = query_features[i]

                spatial_features = block(spatial_features)
                s_proj = self.spatial_features_proj(spatial_features)

                qf_out = self.query_features_block(qf)
                qf_out = self.query_features_proj(qf_out)

                output_dicts.append(
                    {
                        "spatial_features": s_proj,
                        "query_features": qf_out,
                        "bias": self.bias,
                    }
                )
        else:
            if len(query_features) != 1:
                raise ValueError("skip_blocks...")
            qf = query_features[0]
            qf_out = self.query_features_block(qf)
            qf_out = self.query_features_proj(qf_out)

            # In sparse mode, we return the components for einsum
            # spatial_features is NCHW (matching PyTorch)
            output_dicts.append(
                {
                    "spatial_features": spatial_features,
                    "query_features": qf_out,
                    "bias": self.bias,
                }
            )

        return output_dicts
