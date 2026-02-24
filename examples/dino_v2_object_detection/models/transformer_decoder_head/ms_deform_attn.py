import math
import numpy as np

import keras
from keras import ops
from keras import layers


def grid_sample(input, grid, align_corners=False):
    """
    Keras 3 implementation of torch.nn.functional.grid_sample.

    Args:
        input: (N, C, H, W) or (N, H, W, C) - Input feature map. We assume (N, C, H, W) to match PyTorch
               channel-first convention in this context, or we can transpose.
               Use channel_first here because the porting usually involves weights from PyTorch.
               Actually, let's stick to the input signature.
               The caller in ms_deform_attn_core passes (B * n_heads, head_dim, H, W).
        grid: (N, H_out, W_out, 2) - Sampling grid, values in [-1, 1].

    Returns:
        output: (N, C, H_out, W_out)
    """
    # Assuming input is (N, C, H_in, W_in)
    N, C, H_in, W_in = input.shape
    _, H_out, W_out, _ = grid.shape

    # Extract x and y coordinates from grid
    x = grid[..., 0]
    y = grid[..., 1]

    # Compute pixel coordinates
    if align_corners:
        x = ((x + 1) / 2) * (W_in - 1)
        y = ((y + 1) / 2) * (H_in - 1)
    else:
        x = ((x + 1) * W_in - 1) / 2
        y = ((y + 1) * H_in - 1) / 2

    # Get corner pixel coordinates
    x0 = ops.floor(x)
    x1 = x0 + 1
    y0 = ops.floor(y)
    y1 = y0 + 1

    # Clip coordinates to be within image bounds for gathering
    # We clip to ensure we don't index out of bounds, but we will mask invalid samples later
    x0_c = ops.clip(x0, 0, W_in - 1)
    x1_c = ops.clip(x1, 0, W_in - 1)
    y0_c = ops.clip(y0, 0, H_in - 1)
    y1_c = ops.clip(y1, 0, H_in - 1)

    # Cast to integer for indexing
    x0_i = ops.cast(x0_c, "int32")
    x1_i = ops.cast(x1_c, "int32")
    y0_i = ops.cast(y0_c, "int32")
    y1_i = ops.cast(y1_c, "int32")

    # Helper function to gather values
    def gather_values(y_coords, x_coords):
        # Flatten batch and spatial dimensions for gather
        # input is (N, C, H_in, W_in)

        # Create batch indices
        batch_idx = ops.arange(N)
        batch_idx = ops.reshape(batch_idx, (N, 1, 1))
        batch_idx = ops.broadcast_to(batch_idx, (N, H_out, W_out))

        # Flat index: b * (H_in * W_in) + y * W_in + x
        flat_indices = batch_idx * (H_in * W_in) + y_coords * W_in + x_coords
        flat_indices = ops.reshape(flat_indices, (-1,))

        # Reshape input: (N, C, H_in, W_in) -> permute to (N, H_in, W_in, C)
        input_permuted = ops.transpose(input, (0, 2, 3, 1))
        input_flat = ops.reshape(input_permuted, (-1, C))

        # Gather
        values = ops.take(input_flat, flat_indices, axis=0)  # (NHW_out, C)

        # Reshape back to (N, H_out, W_out, C)
        values = ops.reshape(values, (N, H_out, W_out, C))

        # Transpose back to (N, C, H_out, W_out)
        values = ops.transpose(values, (0, 3, 1, 2))
        return values

    raw_Ia = gather_values(y0_i, x0_i)
    raw_Ib = gather_values(y0_i, x1_i)
    raw_Ic = gather_values(y1_i, x0_i)
    raw_Id = gather_values(y1_i, x1_i)

    # Calculate weights
    step_x = x - x0
    step_y = y - y0

    def expand_weights(w):
        return ops.expand_dims(w, axis=1)

    wa = expand_weights((1 - step_x) * (1 - step_y))
    wb = expand_weights(step_x * (1 - step_y))
    wc = expand_weights((1 - step_x) * step_y)
    wd = expand_weights(step_x * step_y)

    # Validity masks
    # Check if UNCLIPPED coordinates are within bounds [0, W-1] and [0, H-1]
    # x0, x1, y0, y1 are floats (or same type as grid)

    def get_valid_mask(x_coord, y_coord):
        # x_coord, y_coord are (N, H_out, W_out)
        vx = ops.logical_and(x_coord >= 0, x_coord <= W_in - 1)
        vy = ops.logical_and(y_coord >= 0, y_coord <= H_in - 1)
        valid = ops.logical_and(vx, vy)
        return ops.expand_dims(
            ops.cast(valid, input.dtype), axis=1
        )  # (N, 1, H_out, W_out)

    mask_a = get_valid_mask(x0, y0)
    mask_b = get_valid_mask(x1, y0)
    mask_c = get_valid_mask(x0, y1)
    mask_d = get_valid_mask(x1, y1)

    Ia = raw_Ia * mask_a
    Ib = raw_Ib * mask_b
    Ic = raw_Ic * mask_c
    Id = raw_Id * mask_d

    interpolated = wa * Ia + wb * Ib + wc * Ic + wd * Id

    # No global masking needed as we masked individual corners
    output = interpolated
    return output


def ms_deform_attn_core(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """
    Core function for Multi-Scale Deformable Attention.
    Args:
        value: (N, Len_in, n_heads, C_head) - transformed values
        value_spatial_shapes: (n_levels, 2) - [(H_0, W_0), ..., (H_{L-1}, W_{L-1})]
        sampling_locations: (N, Len_q, n_heads, n_levels, n_points, 2)
        attention_weights: (N, Len_q, n_heads, n_levels, n_points)

    Returns:
        output: (N, Len_q, n_heads * C_head) -> Actually usually (N, Len_q, C)
    """
    N, Len_in, n_heads, C_head = value.shape
    N, Len_q, n_heads, n_levels, n_points, _ = sampling_locations.shape

    # Split value into list of per-level values
    # In PyTorch: value.split([H*W for ...], dim=1)
    # We need to manually slice because split with size list might not be direct in generic ops

    # Calculate split sections
    split_sizes = [int(h * w) for h, w in value_spatial_shapes]

    # Check total length
    assert sum(split_sizes) == Len_in

    # value_list = ops.split(value, split_sizes, axis=1) # Keras split takes num_or_size_splits?
    # Keras ops.split usually takes num_splits (int) or indices.
    # Let's use slicing manually for robustness
    value_list = []
    current_idx = 0
    for size in split_sizes:
        value_list.append(value[:, current_idx : current_idx + size, :, :])
        current_idx += size

    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    for lid_, (H, W) in enumerate(value_spatial_shapes):
        # value_l_: (N, H*W, n_heads, C_head) -> (N, H, W, n_heads, C_head)
        # But we want (B*n_heads, C_head, H, W) for our grid_sample to match PyTorch logic?
        # Or (B, n_heads, C_head, H, W)?

        # PyTorch logic:
        # value_l_ = value_list[lid_].view(B * n_heads, head_dim, H, W)
        # sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # grid_sample(..., align_corners=False)

        # Let's adopt (N * n_heads, C_head, H, W) for values
        # value_list[lid_] has shape (N, H*W, n_heads, C_head)

        val = value_list[lid_]  # (N, H*W, n_heads, C_head)
        val = ops.transpose(val, (0, 2, 3, 1))  # (N, n_heads, C_head, H*W)
        val = ops.reshape(
            val, (N * n_heads, C_head, int(H), int(W))
        )  # (N*n_heads, C_head, H, W)

        # Sampling grid:
        # sampling_locations: (N, Len_q, n_heads, n_levels, n_points, 2)
        # select level: (N, Len_q, n_heads, n_points, 2)
        # transpose to (N, n_heads, Len_q, n_points, 2) -> flatten to (N*n_heads, Len_q, n_points, 2)
        grid_l = sampling_grids[:, :, :, lid_]  # (N, Len_q, n_heads, n_points, 2)
        grid_l = ops.transpose(
            grid_l, (0, 2, 1, 3, 4)
        )  # (N, n_heads, Len_q, n_points, 2)
        grid_l = ops.reshape(grid_l, (N * n_heads, Len_q, n_points, 2))

        # Now sample
        # val: (Batch, C, H, W)
        # grid_l: (Batch, OutH, OutW, 2) where OutH=Len_q, OutW=n_points
        sampled = grid_sample(val, grid_l, align_corners=False)
        # Output: (Batch, C, OutH, OutW) -> (N*n_heads, C_head, Len_q, n_points)

        sampling_value_list.append(sampled)

    # Convert attention weights
    # attention_weights: (N, Len_q, n_heads, n_levels, n_points)
    # -> (N, n_heads, Len_q, n_levels * n_points) -> flatten first dim -> (N*n_heads, 1, Len_q, n_levels*n_points)
    attn = ops.transpose(
        attention_weights, (0, 2, 1, 3, 4)
    )  # (N, n_heads, Len_q, n_levels, n_points)
    attn = ops.reshape(attn, (N * n_heads, 1, Len_q, n_levels * n_points))

    # Process sampled values
    # Stack along level dim?
    # sampling_value_list elements are (N*n_heads, C_head, Len_q, n_points)
    # Stack levels: (N*n_heads, C_head, Len_q, n_levels, n_points)
    # But wait, original code:
    # list append -> stack dim=-2 -> flatten -2
    # stack dim=-2: (..., n_levels, n_points)

    # We have list of (N*n_heads, C_head, Len_q, n_points)
    # Stack on axis -2? No.
    # We want to combine n_levels and n_points.

    # Let's stack on a new axis 3 (before points):
    # (N*n_heads, C_head, Len_q, n_levels, n_points)
    stacked = ops.stack(sampling_value_list, axis=3)

    # Flatten last two dims: (N*n_heads, C_head, Len_q, n_levels * n_points)
    stacked_flat = ops.reshape(
        stacked, (N * n_heads, C_head, Len_q, n_levels * n_points)
    )

    # Apply attention
    # stacked: (B, C, Lq, P_tot)
    # attn: (B, 1, Lq, P_tot)
    # output = sum(stacked * attn, dim=-1) -> (B, C, Lq)
    output = ops.sum(stacked_flat * attn, axis=-1)

    # Reshape back
    # (N*n_heads, C_head, Len_q) -> (N, n_heads, C_head, Len_q)
    output = ops.reshape(output, (N, n_heads, C_head, Len_q))

    # Transpose to (N, Len_q, n_heads, C_head) ?
    # Original PyTorch output: (N, Len_q, C)
    # Original last line: output.transpose(1, 2).contiguous() -> (B, Lq, n_heads*head_dim)
    # Our output is (N, n_heads, C_head, Len_q)

    output = ops.transpose(output, (0, 3, 1, 2))  # (N, Len_q, n_heads, C_head)
    output = ops.reshape(output, (N, Len_q, n_heads * C_head))

    return output


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        # We can't raise Python errors inside graph trace easily but this is usually constant check
        return False
    return (n & (n - 1) == 0) and n != 0


@keras.saving.register_keras_serializable(package="RFDETR")
class MSDeformAttn(layers.Layer):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, **kwargs):
        super().__init__(**kwargs)
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}"
            )

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        # Layers
        # sampling_offsets: Linear(d_model, n_heads * n_levels * n_points * 2)
        self.sampling_offsets = layers.Dense(
            n_heads * n_levels * n_points * 2, name="sampling_offsets"
        )

        # attention_weights: Linear(d_model, n_heads * n_levels * n_points)
        self.attention_weights = layers.Dense(
            n_heads * n_levels * n_points, name="attention_weights"
        )

        # value_proj: Linear(d_model, d_model)
        self.value_proj = layers.Dense(d_model, name="value_proj")

        # output_proj: Linear(d_model, d_model)
        self.output_proj = layers.Dense(d_model, name="output_proj")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "n_levels": self.n_levels,
                "n_heads": self.n_heads,
                "n_points": self.n_points,
            }
        )
        return config

    def build(self, input_shape):
        # Initialize weights similar to PyTorch if needed, but standard init is fine.
        # Original code uses constant_ for offsets and specific init for others.
        # We can simulate this by setting weights after creation if we port them,
        # or rely on loading weights.
        # Since this task is about replication/porting, we assume weights will be loaded.
        # But if instantiated fresh, we might want similar init.
        # PyTorch init:
        # constant_(sampling_offsets.weight, 0.)
        # grid_init for sampling_offsets.bias
        # constant_(attention_weights, 0.)
        # xavier_uniform_(value_proj)
        # xavier_uniform_(output_proj)

        # We'll leave default initialization for now as we are likely loading weights.
        # But for 'testing', valid init helps.

        # Construct bias for sampling_offsets
        # bias calculation logic from PyTorch
        thetas = ops.arange(self.n_heads, dtype="float32") * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = ops.stack([ops.cos(thetas), ops.sin(thetas)], -1)  # (n_heads, 2)

        # grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        grid_norm = ops.max(ops.abs(grid_init), axis=-1, keepdims=True)
        grid_init = grid_init / grid_norm

        # view(n_heads, 1, 1, 2).repeat(1, n_levels, n_points, 1)
        grid_init = ops.reshape(grid_init, (self.n_heads, 1, 1, 2))
        # tile (repeat)
        grid_init = ops.tile(grid_init, (1, self.n_levels, self.n_points, 1))

        # for i in range(n_points): grid_init[:, :, i, :] *= i + 1
        # Make a multiplier mask
        multiplier = ops.arange(self.n_points, dtype="float32") + 1
        multiplier = ops.reshape(multiplier, (1, 1, self.n_points, 1))
        grid_init = grid_init * multiplier

        # Final bias shape: (n_heads * n_levels * n_points * 2)
        bias_init = ops.reshape(grid_init, (-1,))

        # We can't easily force bias init in `build` for standard Dense layer unless we use a custom initializer.
        # But for porting, we just load weights.

        super().build(input_shape)

    def call(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index=None,
        input_padding_mask=None,
        **kwargs,
    ):
        """
        Args:
            query: (N, Length_query, C)
            reference_points: (N, Length_query, n_levels, 2) or (N, Length_query, n_levels, 4)
            input_flatten: (N, Len_in, C)
            input_spatial_shapes: (n_levels, 2)
            input_level_start_index: (n_levels, ) - Not strictly used in core if we split manually, but kept for API.
            input_padding_mask: (N, Len_in)
        """
        N = ops.shape(query)[0]
        Len_q = ops.shape(query)[1]
        Len_in = ops.shape(input_flatten)[1]

        # Check spatial shapes consistency
        # assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            # mask: (N, Len_in) -> (N, Len_in, 1)
            mask = ops.expand_dims(input_padding_mask, axis=-1)
            value = ops.where(
                mask, 0.0, value
            )  # Assuming mask is True for padding (PyTorch convention) -> masked_fill(mask, 0)

        # value: (N, Len_in, C) -> (N, Len_in, n_heads, C_head)
        value = ops.reshape(value, (N, Len_in, self.n_heads, self.head_dim))

        # Sampling offsets
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = ops.reshape(
            sampling_offsets, (N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        )

        # Attention weights
        attention_weights = self.attention_weights(query)
        attention_weights = ops.reshape(
            attention_weights, (N, Len_q, self.n_heads, self.n_levels * self.n_points)
        )
        attention_weights = ops.softmax(attention_weights, axis=-1)
        attention_weights = ops.reshape(
            attention_weights, (N, Len_q, self.n_heads, self.n_levels, self.n_points)
        )

        # Sampling locations
        # reference_points: (N, Lq, n_levels, 2 or 4)
        if ops.shape(reference_points)[-1] == 2:
            # offset_normalizer: (n_levels, 2) -> (w, h)
            # input_spatial_shapes is (h, w)
            # normalizer should be (w, h) so we swap
            # input_spatial_shapes: [[h, w], ...]
            # We need to construct tensors

            # Note: input_spatial_shapes likely comes in as a tensor or list.
            # Convert to tensor if not
            spatial_shapes = ops.convert_to_tensor(
                input_spatial_shapes, dtype="float32"
            )

            # stack [w, h] -> [shape[1], shape[0]]
            offset_normalizer = ops.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], axis=-1
            )

            # reference_points[:, :, None, :, None, :] -> (N, Lq, 1, n_levels, 1, 2)
            ref_pts = ops.expand_dims(ops.expand_dims(reference_points, 2), 4)

            # offset_normalizer[None, None, None, :, None, :] -> (1, 1, 1, n_levels, 1, 2)
            normalizer_reshaped = ops.reshape(
                offset_normalizer, (1, 1, 1, self.n_levels, 1, 2)
            )

            sampling_locations = ref_pts + sampling_offsets / normalizer_reshaped

        elif ops.shape(reference_points)[-1] == 4:
            # reference_points: (x, y, w, h)
            ref_pts_xy = reference_points[..., :2]
            ref_pts_wh = reference_points[..., 2:]

            ref_pts_xy = ops.expand_dims(ops.expand_dims(ref_pts_xy, 2), 4)
            ref_pts_wh = ops.expand_dims(ops.expand_dims(ref_pts_wh, 2), 4)

            sampling_locations = (
                ref_pts_xy + sampling_offsets / self.n_points * ref_pts_wh * 0.5
            )
        else:
            raise ValueError("Last dim of reference_points must be 2 or 4.")

        # Call core
        output = ms_deform_attn_core(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )

        output = self.output_proj(output)
        return output
