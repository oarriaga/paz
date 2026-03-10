import math
import numpy as np

import keras
from keras import ops
from keras import layers


def grid_sample(input, grid, align_corners=False):
    """Bilinear sampling of a feature map at arbitrary grid locations.

    Samples from the input feature map using normalized coordinates in
    [-1, 1], with bilinear interpolation and zero-padding for out-of-bounds
    locations.

    Args:
        input (tensor): Feature map of shape (N, C, H_in, W_in) in
            channels-first format.
        grid (tensor): Sampling grid of shape (N, H_out, W_out, 2) with
            normalized coordinates in [-1, 1].
        align_corners (bool): If True, maps grid corners exactly to input
            corners. If False, uses half-pixel offset alignment.

    Returns:
        tensor: Sampled output of shape (N, C, H_out, W_out).
    """
    N, C, H_in, W_in = input.shape
    _, H_out, W_out, _ = grid.shape

    # Separate x (horizontal) and y (vertical) grid coordinates
    x = grid[..., 0]
    y = grid[..., 1]

    # Convert normalized [-1, 1] grid coordinates to pixel coordinates.
    # The two alignment modes differ in whether corners or pixel centers
    # map to the grid extremes.
    if align_corners:
        x = ((x + 1) / 2) * (W_in - 1)
        y = ((y + 1) / 2) * (H_in - 1)
    else:
        x = ((x + 1) * W_in - 1) / 2
        y = ((y + 1) * H_in - 1) / 2

    # Compute the four nearest integer pixel coordinates for bilinear
    # interpolation (top-left, top-right, bottom-left, bottom-right)
    x0 = ops.floor(x)
    x1 = x0 + 1
    y0 = ops.floor(y)
    y1 = y0 + 1

    # Clip coordinates to valid image bounds for safe indexing.
    # Out-of-bounds samples will be zeroed out via validity masks below.
    x0_c = ops.clip(x0, 0, W_in - 1)
    x1_c = ops.clip(x1, 0, W_in - 1)
    y0_c = ops.clip(y0, 0, H_in - 1)
    y1_c = ops.clip(y1, 0, H_in - 1)

    x0_i = ops.cast(x0_c, "int32")
    x1_i = ops.cast(x1_c, "int32")
    y0_i = ops.cast(y0_c, "int32")
    y1_i = ops.cast(y1_c, "int32")

    def gather_values(y_coords, x_coords):
        """Gather pixel values from the input at specified (y, x) coordinates.

        Flattens batch and spatial dims for efficient indexing, then
        reshapes back to (N, C, H_out, W_out).
        """
        # Build per-sample batch indices and broadcast to output spatial dims
        batch_idx = ops.arange(N)
        batch_idx = ops.reshape(batch_idx, (N, 1, 1))
        batch_idx = ops.broadcast_to(batch_idx, (N, H_out, W_out))

        # Compute flat 1-D indices into the (N * H_in * W_in) pixel array
        flat_indices = batch_idx * (H_in * W_in) + y_coords * W_in + x_coords
        flat_indices = ops.reshape(flat_indices, (-1,))

        # Transpose input to channels-last for contiguous spatial indexing
        input_permuted = ops.transpose(input, (0, 2, 3, 1))
        input_flat = ops.reshape(input_permuted, (-1, C))

        # Gather and reshape back to channels-first output
        values = ops.take(input_flat, flat_indices, axis=0)
        values = ops.reshape(values, (N, H_out, W_out, C))
        values = ops.transpose(values, (0, 3, 1, 2))
        return values

    # Gather pixel values at the four bilinear interpolation corners
    raw_Ia = gather_values(y0_i, x0_i)
    raw_Ib = gather_values(y0_i, x1_i)
    raw_Ic = gather_values(y1_i, x0_i)
    raw_Id = gather_values(y1_i, x1_i)

    # Compute bilinear interpolation weights based on fractional position
    step_x = x - x0
    step_y = y - y0

    def expand_weights(w):
        return ops.expand_dims(w, axis=1)

    wa = expand_weights((1 - step_x) * (1 - step_y))
    wb = expand_weights(step_x * (1 - step_y))
    wc = expand_weights((1 - step_x) * step_y)
    wd = expand_weights(step_x * step_y)

    def get_valid_mask(x_coord, y_coord):
        """Create a binary mask for coordinates within image bounds.

        Uses the original unclipped coordinates to identify out-of-bounds
        samples that should contribute zero to the interpolation.
        """
        vx = ops.logical_and(x_coord >= 0, x_coord <= W_in - 1)
        vy = ops.logical_and(y_coord >= 0, y_coord <= H_in - 1)
        valid = ops.logical_and(vx, vy)
        return ops.expand_dims(
            ops.cast(valid, input.dtype), axis=1
        )

    # Zero out contributions from out-of-bounds corner samples
    mask_a = get_valid_mask(x0, y0)
    mask_b = get_valid_mask(x1, y0)
    mask_c = get_valid_mask(x0, y1)
    mask_d = get_valid_mask(x1, y1)

    Ia = raw_Ia * mask_a
    Ib = raw_Ib * mask_b
    Ic = raw_Ic * mask_c
    Id = raw_Id * mask_d

    # Weighted sum of the four masked corner samples
    output = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return output


def ms_deform_attn_core(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """Core computation for multi-scale deformable attention.

    Samples values from multi-scale feature maps at learned offset locations,
    then applies learned attention weights to produce the output.

    Args:
        value (tensor): Projected values of shape
            (N, Len_in, n_heads, C_head) where Len_in is the total number
            of spatial positions across all feature levels.
        value_spatial_shapes (list): List of (H, W) tuples specifying the
            spatial dimensions of each feature level.
        sampling_locations (tensor): Normalized sampling coordinates of shape
            (N, Len_q, n_heads, n_levels, n_points, 2) in [0, 1].
        attention_weights (tensor): Per-point attention weights of shape
            (N, Len_q, n_heads, n_levels, n_points).

    Returns:
        tensor: Attended output of shape (N, Len_q, n_heads * C_head).
    """
    N, Len_in, n_heads, C_head = value.shape
    N, Len_q, n_heads, n_levels, n_points, _ = sampling_locations.shape

    # Split the flattened multi-scale values back into per-level segments
    split_sizes = [int(h * w) for h, w in value_spatial_shapes]
    assert sum(split_sizes) == Len_in

    value_list = []
    current_idx = 0
    for size in split_sizes:
        value_list.append(value[:, current_idx : current_idx + size, :, :])
        current_idx += size

    # Convert sampling locations from [0, 1] to grid coordinates in [-1, 1]
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    for lid_, (H, W) in enumerate(value_spatial_shapes):
        # Reshape per-level values: (N, H*W, n_heads, C_head)
        # -> (N*n_heads, C_head, H, W) for bilinear sampling
        val = value_list[lid_]
        val = ops.transpose(val, (0, 2, 3, 1))
        val = ops.reshape(
            val, (N * n_heads, C_head, int(H), int(W))
        )

        # Extract and reshape the sampling grid for this level:
        # (N, Len_q, n_heads, n_points, 2) -> (N*n_heads, Len_q, n_points, 2)
        grid_l = sampling_grids[:, :, :, lid_]
        grid_l = ops.transpose(
            grid_l, (0, 2, 1, 3, 4)
        )
        grid_l = ops.reshape(grid_l, (N * n_heads, Len_q, n_points, 2))

        # Bilinear sample: output is (N*n_heads, C_head, Len_q, n_points)
        sampled = grid_sample(val, grid_l, align_corners=False)
        sampling_value_list.append(sampled)

    # Reshape attention weights:
    # (N, Len_q, n_heads, n_levels, n_points)
    # -> (N*n_heads, 1, Len_q, n_levels * n_points)
    attn = ops.transpose(
        attention_weights, (0, 2, 1, 3, 4)
    )
    attn = ops.reshape(attn, (N * n_heads, 1, Len_q, n_levels * n_points))

    # Stack sampled values across levels and flatten the level/point dims:
    # list of (N*n_heads, C_head, Len_q, n_points)
    # -> (N*n_heads, C_head, Len_q, n_levels * n_points)
    stacked = ops.stack(sampling_value_list, axis=3)
    stacked_flat = ops.reshape(
        stacked, (N * n_heads, C_head, Len_q, n_levels * n_points)
    )

    # Apply attention weights and sum over all sampling points:
    # (N*n_heads, C_head, Len_q, n_levels*n_points) * (N*n_heads, 1, Len_q, n_levels*n_points)
    # -> sum over last axis -> (N*n_heads, C_head, Len_q)
    output = ops.sum(stacked_flat * attn, axis=-1)

    # Reshape to (N, Len_q, n_heads * C_head) as the final output
    output = ops.reshape(output, (N, n_heads, C_head, Len_q))
    output = ops.transpose(output, (0, 3, 1, 2))
    output = ops.reshape(output, (N, Len_q, n_heads * C_head))

    return output


def _is_power_of_2(n):
    """Check whether n is a positive power of 2."""
    if (not isinstance(n, int)) or (n < 0):
        return False
    return (n & (n - 1) == 0) and n != 0


@keras.saving.register_keras_serializable(package="RFDETR")
class MSDeformAttn(layers.Layer):
    """Multi-Scale Deformable Attention layer.

    For each query, this layer predicts a sparse set of sampling offsets
    and attention weights across multiple feature map scales, then
    aggregates the sampled values into the output.

    Attributes:
        d_model (int): Model embedding dimension.
        n_levels (int): Number of multi-scale feature levels.
        n_heads (int): Number of attention heads.
        n_points (int): Number of sampling points per head per level.
        head_dim (int): Dimension per attention head (d_model // n_heads).
        sampling_offsets (Dense): Predicts 2-D sampling offsets.
        attention_weights (Dense): Predicts per-point attention weights.
        value_proj (Dense): Projects input values to d_model dimensions.
        output_proj (Dense): Projects attended output to d_model dimensions.
    """

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

        # Predicts 2-D offsets for each head, level, and sampling point
        self.sampling_offsets = layers.Dense(
            n_heads * n_levels * n_points * 2, name="sampling_offsets"
        )

        # Predicts unnormalized attention weights (softmax applied later)
        self.attention_weights = layers.Dense(
            n_heads * n_levels * n_points, name="attention_weights"
        )

        # Linear projections for input values and final output
        self.value_proj = layers.Dense(d_model, name="value_proj")
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
        """Build layer and compute initial sampling offset bias.

        The sampling offset bias is initialized with a radial pattern:
        each attention head is assigned a base direction uniformly
        distributed around the unit circle, and each sampling point
        index scales this direction proportionally. This encourages
        diverse spatial coverage at initialization.
        """
        # Generate evenly-spaced angles for each attention head
        thetas = ops.arange(self.n_heads, dtype="float32") * (
            2.0 * math.pi / self.n_heads
        )
        # Unit direction vector per head, normalized by max component
        grid_init = ops.stack([ops.cos(thetas), ops.sin(thetas)], -1)
        grid_norm = ops.max(ops.abs(grid_init), axis=-1, keepdims=True)
        grid_init = grid_init / grid_norm

        # Broadcast the base direction across all levels and points
        grid_init = ops.reshape(grid_init, (self.n_heads, 1, 1, 2))
        grid_init = ops.tile(grid_init, (1, self.n_levels, self.n_points, 1))

        # Scale each sampling point's offset by its 1-based index so that
        # farther points sample at greater distances from the reference
        multiplier = ops.arange(self.n_points, dtype="float32") + 1
        multiplier = ops.reshape(multiplier, (1, 1, self.n_points, 1))
        grid_init = grid_init * multiplier

        bias_init = ops.reshape(grid_init, (-1,))

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
        """Compute multi-scale deformable attention.

        Args:
            query (tensor): Query features, shape (N, Length_query, C).
            reference_points (tensor): Reference coordinates per query per
                level, shape (N, Length_query, n_levels, 2) for (x, y) or
                (N, Length_query, n_levels, 4) for (x, y, w, h).
            input_flatten (tensor): Flattened multi-scale feature values,
                shape (N, Len_in, C).
            input_spatial_shapes (array): Spatial shapes per level,
                shape (n_levels, 2) as [(H, W), ...].
            input_level_start_index (tensor): Start index of each level
                in the flattened sequence. Kept for API compatibility.
            input_padding_mask (tensor): Boolean mask of shape (N, Len_in)
                where True indicates padding positions to be zeroed.

        Returns:
            tensor: Output of shape (N, Length_query, C).
        """
        N = ops.shape(query)[0]
        Len_q = ops.shape(query)[1]
        Len_in = ops.shape(input_flatten)[1]

        # Project input values and zero out padded positions
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            mask = ops.expand_dims(input_padding_mask, axis=-1)
            value = ops.where(mask, 0.0, value)

        # Split the projected values into per-head channels
        value = ops.reshape(value, (N, Len_in, self.n_heads, self.head_dim))

        # Predict sampling offsets from the query
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = ops.reshape(
            sampling_offsets, (N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        )

        # Predict and normalize attention weights via softmax across all
        # levels and points jointly
        attention_weights = self.attention_weights(query)
        attention_weights = ops.reshape(
            attention_weights, (N, Len_q, self.n_heads, self.n_levels * self.n_points)
        )
        attention_weights = ops.softmax(attention_weights, axis=-1)
        attention_weights = ops.reshape(
            attention_weights, (N, Len_q, self.n_heads, self.n_levels, self.n_points)
        )

        # Compute absolute sampling locations from reference points + offsets.
        # Two modes depending on reference point dimensionality:
        if ops.shape(reference_points)[-1] == 2:
            # 2-D reference points (x, y): normalize offsets by spatial dims.
            # Swap (H, W) to (W, H) to match (x, y) coordinate ordering.
            spatial_shapes = ops.convert_to_tensor(
                input_spatial_shapes, dtype="float32"
            )
            offset_normalizer = ops.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], axis=-1
            )

            # Broadcast reference points and normalizer to match offset shape
            ref_pts = ops.expand_dims(ops.expand_dims(reference_points, 2), 4)
            normalizer_reshaped = ops.reshape(
                offset_normalizer, (1, 1, 1, self.n_levels, 1, 2)
            )

            sampling_locations = ref_pts + sampling_offsets / normalizer_reshaped

        elif ops.shape(reference_points)[-1] == 4:
            # 4-D reference points (x, y, w, h): offsets are scaled relative
            # to the reference box width/height
            ref_pts_xy = reference_points[..., :2]
            ref_pts_wh = reference_points[..., 2:]

            ref_pts_xy = ops.expand_dims(ops.expand_dims(ref_pts_xy, 2), 4)
            ref_pts_wh = ops.expand_dims(ops.expand_dims(ref_pts_wh, 2), 4)

            sampling_locations = (
                ref_pts_xy + sampling_offsets / self.n_points * ref_pts_wh * 0.5
            )
        else:
            raise ValueError("Last dim of reference_points must be 2 or 4.")

        # Perform the core deformable attention computation and project output
        output = ms_deform_attn_core(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )

        output = self.output_proj(output)
        return output
