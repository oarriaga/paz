import keras
from keras import layers, ops
import math
import warnings


def bilinear_grid_sample(image, grid):
    """
    Performs a bilinear grid sample on a batch of images.
    Matches PyTorch grid_sample with align_corners=False and padding_mode='zeros'.

    Handles empty images (H=0, W=0) safely by returning zeros.

    Args:
        image: Tensor of shape (N, C, H, W)
        grid: Tensor of shape (N, H_out, W_out, 2) in range [-1, 1].
              Last dim is (x, y).

    Returns:
        Tensor of shape (N, C, H_out, W_out)
    """
    shape = ops.shape(image)
    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    H_float = ops.cast(H, "float32")
    W_float = ops.cast(W, "float32")

    H_out = ops.shape(grid)[1]
    W_out = ops.shape(grid)[2]

    # 1. Unpack coordinates
    x = grid[..., 0]
    y = grid[..., 1]

    # 2. Compute pixel coordinates (align_corners=False definition)
    # In this mode, grid=-1 -> -0.5 (pixel coords), grid=1 -> W-0.5
    x_pix = ((x + 1.0) * W_float - 1.0) / 2.0
    y_pix = ((y + 1.0) * H_float - 1.0) / 2.0

    # 3. Get corner coordinates (flooring)
    x0 = ops.floor(x_pix)
    x1 = x0 + 1.0
    y0 = ops.floor(y_pix)
    y1 = y0 + 1.0

    # 4. Compute interpolation weights
    wa = (x1 - x_pix) * (y1 - y_pix)
    wb = (x1 - x_pix) * (y_pix - y0)
    wc = (x_pix - x0) * (y1 - y_pix)
    wd = (x_pix - x0) * (y_pix - y0)

    # 5. Boundary Checks (Padding Mode = 'zeros')
    x0_i = ops.cast(x0, "int32")
    x1_i = ops.cast(x1, "int32")
    y0_i = ops.cast(y0, "int32")
    y1_i = ops.cast(y1, "int32")

    # Create masks for each corner
    valid_x0 = (x0_i >= 0) & (x0_i < W)
    valid_x1 = (x1_i >= 0) & (x1_i < W)
    valid_y0 = (y0_i >= 0) & (y0_i < H)
    valid_y1 = (y1_i >= 0) & (y1_i < H)

    mask_a = ops.cast(valid_x0 & valid_y0, image.dtype)
    mask_b = ops.cast(valid_x0 & valid_y1, image.dtype)
    mask_c = ops.cast(valid_x1 & valid_y0, image.dtype)
    mask_d = ops.cast(valid_x1 & valid_y1, image.dtype)

    # 6. Safe Gathering
    x0_c = ops.clip(x0_i, 0, W - 1)
    x1_c = ops.clip(x1_i, 0, W - 1)
    y0_c = ops.clip(y0_i, 0, H - 1)
    y1_c = ops.clip(y1_i, 0, H - 1)

    # Transpose image to (N, H, W, C) for flattening
    img_trans = ops.transpose(image, (0, 2, 3, 1))
    flat_img = ops.reshape(img_trans, (-1, C))

    # Add dummy row for safe gather (handle empty/out-of-bounds safety)
    dummy = ops.zeros((1, C), dtype=flat_img.dtype)
    flat_img_safe = ops.concatenate([flat_img, dummy], axis=0)
    max_index = ops.shape(flat_img)[0]

    # Precompute batch offsets
    batch_range = ops.arange(N, dtype="int32")
    batch_offset = ops.reshape(batch_range, (N, 1, 1)) * (H * W)

    def get_pixel_value_masked(x_idx, y_idx, mask):
        # Calculate flat indices: batch_offset + y * W + x
        flat_indices = batch_offset + y_idx * W + x_idx
        flat_indices = ops.reshape(flat_indices, (-1,))
        flat_indices = ops.clip(flat_indices, 0, max_index)
        vals = ops.take(flat_img_safe, flat_indices, axis=0)
        vals = ops.reshape(vals, (N, H_out, W_out, C))
        vals = ops.transpose(vals, (0, 3, 1, 2))
        return vals * ops.expand_dims(mask, 1)

    # Gather and Mask
    Ia = get_pixel_value_masked(x0_c, y0_c, mask_a)
    Ib = get_pixel_value_masked(x0_c, y1_c, mask_b)
    Ic = get_pixel_value_masked(x1_c, y0_c, mask_c)
    Id = get_pixel_value_masked(x1_c, y1_c, mask_d)

    out = (
        ops.expand_dims(wa, 1) * Ia
        + ops.expand_dims(wb, 1) * Ib
        + ops.expand_dims(wc, 1) * Ic
        + ops.expand_dims(wd, 1) * Id
    )
    return out


def ms_deform_attn_core(
    value_list,
    value_spatial_shapes,
    sampling_locations,
    attention_weights,
    value_spatial_shapes_list=None,  # NEW ARGUMENT
):
    """
    Core function for MSDeformAttn.
    Iterates over value_list instead of slicing to support JAX dynamic shapes.
    """
    shape_loc = ops.shape(sampling_locations)
    B = shape_loc[0]
    Len_q = shape_loc[1]
    n_heads = shape_loc[2]
    L = shape_loc[3]
    P = shape_loc[4]

    # Check head_dim from first value in list
    shape_val0 = ops.shape(value_list[0])
    head_dim = shape_val0[2]

    sampling_grids = 2.0 * sampling_locations - 1.0
    sampling_value_list = []

    for lid_ in range(L):
        value_l_ = value_list[lid_]

        if value_spatial_shapes_list is not None:
            H, W = value_spatial_shapes_list[lid_]
        else:
            H = value_spatial_shapes[lid_, 0]
            W = value_spatial_shapes[lid_, 1]

        # Reshape to (B*n_heads, head_dim, H, W)
        value_l_ = ops.reshape(value_l_, (B * n_heads, head_dim, H, W))

        grid_l = sampling_grids[:, :, :, lid_, :, :]
        grid_l = ops.transpose(grid_l, (0, 2, 1, 3, 4))
        grid_l = ops.reshape(grid_l, (B * n_heads, Len_q, P, 2))

        sampling_value_l_ = bilinear_grid_sample(value_l_, grid_l)
        sampling_value_list.append(sampling_value_l_)

    att_w = ops.transpose(attention_weights, (0, 2, 1, 3, 4))
    att_w = ops.reshape(att_w, (B * n_heads, 1, Len_q, L * P))

    stack_vals = ops.stack(sampling_value_list, axis=3)
    stack_vals = ops.reshape(stack_vals, (B * n_heads, head_dim, Len_q, L * P))

    output = ops.sum(stack_vals * att_w, axis=-1)
    output = ops.reshape(output, (B, n_heads * head_dim, Len_q))
    output = ops.transpose(output, (0, 2, 1))

    return output


@keras.saving.register_keras_serializable(package="DeformableDETR")
class DeformableAttentionBiasInitializer(keras.initializers.Initializer):
    def __init__(self, n_heads, n_levels, n_points):
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points

    def __call__(self, shape, dtype=None):
        expected_dim = self.n_heads * self.n_levels * self.n_points * 2
        thetas = ops.arange(self.n_heads, dtype="float32")
        scale_factor = ops.cast(2.0 * math.pi / self.n_heads, "float32")
        thetas = thetas * scale_factor
        grid_init = ops.stack([ops.cos(thetas), ops.sin(thetas)], axis=-1)
        grid_init = grid_init / ops.max(ops.abs(grid_init), axis=-1, keepdims=True)
        grid_init = ops.reshape(grid_init, (self.n_heads, 1, 1, 2))
        grid_init = ops.tile(grid_init, (1, self.n_levels, self.n_points, 1))
        scaler = ops.arange(1, self.n_points + 1, dtype="float32")
        scaler = ops.reshape(scaler, (1, 1, self.n_points, 1))
        grid_init = grid_init * scaler
        result = ops.reshape(grid_init, (-1,))
        if dtype is not None:
            result = ops.cast(result, dtype)
        return result

    def get_config(self):
        return {
            "n_heads": self.n_heads,
            "n_levels": self.n_levels,
            "n_points": self.n_points,
        }


def _is_power_of_2(n):
    return (n & (n - 1) == 0) and n != 0


@keras.saving.register_keras_serializable(package="DeformableDETR")
class MSDeformAttn(layers.Layer):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, **kwargs):
        super().__init__(**kwargs)
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}"
            )
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn("d_model per head should be power of 2 for efficiency.")

        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = layers.Dense(
            n_heads * n_levels * n_points * 2,
            kernel_initializer="zeros",
            bias_initializer=DeformableAttentionBiasInitializer(
                n_heads, n_levels, n_points
            ),
            name="sampling_offsets",
        )
        self.attention_weights = layers.Dense(
            n_heads * n_levels * n_points,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="attention_weights",
        )
        self.value_proj = layers.Dense(
            d_model,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name="value_proj",
        )
        self.output_proj = layers.Dense(
            d_model,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name="output_proj",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def call(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
        input_flatten_list=None,
        input_padding_mask_list=None,
        input_spatial_shapes_list=None,  # NEW ARGUMENT
    ):
        shape_q = ops.shape(query)
        N = shape_q[0]
        Len_q = shape_q[1]

        # Calculate offsets and weights
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = ops.reshape(
            sampling_offsets, (N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        )
        attention_weights = self.attention_weights(query)
        attention_weights = ops.reshape(
            attention_weights, (N, Len_q, self.n_heads, self.n_levels * self.n_points)
        )
        attention_weights = ops.softmax(attention_weights, axis=-1)
        attention_weights = ops.reshape(
            attention_weights, (N, Len_q, self.n_heads, self.n_levels, self.n_points)
        )

        # Sampling locations logic
        ref_dim = reference_points.shape[-1]
        if ref_dim == 2:
            spatial_shapes_f = ops.cast(input_spatial_shapes, "float32")
            offset_normalizer = ops.stack(
                [spatial_shapes_f[..., 1], spatial_shapes_f[..., 0]], axis=-1
            )
            offset_normalizer = ops.reshape(
                offset_normalizer, (1, 1, 1, self.n_levels, 1, 2)
            )
            ref_points_exp = ops.expand_dims(reference_points, axis=2)
            ref_points_exp = ops.expand_dims(ref_points_exp, axis=4)
            sampling_locations = ref_points_exp + sampling_offsets / offset_normalizer
        elif ref_dim == 4:
            ref_xy = reference_points[..., :2]
            ref_wh = reference_points[..., 2:]
            ref_xy = ops.expand_dims(ops.expand_dims(ref_xy, 2), 4)
            ref_wh = ops.expand_dims(ops.expand_dims(ref_wh, 2), 4)
            sampling_locations = (
                ref_xy
                + sampling_offsets / ops.cast(self.n_points, "float32") * ref_wh * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but get {ref_dim}."
            )

        # Handle value projection using list if available (Safe for JAX Dynamic Shapes)
        d_head = self.d_model // self.n_heads
        value_list = []

        if input_flatten_list is not None:
            # Iterate through list, project, and reshape per level
            for i, feat in enumerate(input_flatten_list):
                # feat: (N, Len_lvl, C)
                val = self.value_proj(feat)

                # Masking logic
                if input_padding_mask_list is not None:
                    # mask: (N, Len_lvl) or (N, Len_lvl, 1) if reshaped
                    mask = input_padding_mask_list[i]
                    mask = ops.expand_dims(mask, axis=-1)  # Ensure (N, Len_lvl, 1)
                    val = ops.where(mask, 0.0, val)

                # Reshape: (N, Len_lvl, n_heads, d_head) -> (N, n_heads, d_head, Len_lvl)
                shape_feat = ops.shape(feat)
                Len_lvl = shape_feat[1]
                val = ops.reshape(val, (N, Len_lvl, self.n_heads, d_head))
                val = ops.transpose(val, (0, 2, 3, 1))
                value_list.append(val)
        else:
            # Fallback for compatibility (will fail JAX tracing if shapes dynamic)
            value = self.value_proj(input_flatten)
            if input_padding_mask is not None:
                mask = ops.expand_dims(input_padding_mask, axis=-1)
                value = ops.where(mask, 0.0, value)
            shape_in = ops.shape(input_flatten)
            Len_in = shape_in[1]
            value = ops.reshape(value, (N, Len_in, self.n_heads, d_head))
            value = ops.transpose(value, (0, 2, 3, 1))
            # Slice manually based on spatial shapes (prone to failure)
            start = 0
            for i in range(self.n_levels):
                H = input_spatial_shapes[i, 0]
                W = input_spatial_shapes[i, 1]
                length = H * W
                val = ops.slice(
                    value, [0, 0, 0, start], [N, self.n_heads, d_head, length]
                )
                value_list.append(val)
                start += length

        output = ms_deform_attn_core(
            value_list,
            input_spatial_shapes,
            sampling_locations,
            attention_weights,
            value_spatial_shapes_list=input_spatial_shapes_list,  # NEW ARGUMENT
        )
        output = self.output_proj(output)
        return output

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
