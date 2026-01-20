import keras
from keras import layers, ops, activations
import copy
import math
import warnings
from keras.saving import register_keras_serializable


def _process_coord_embedding(coord_embed, dim_t):
    # Expand dims for broadcasting: (n_query, bs) -> (n_query, bs, 1)
    pos_coord = ops.expand_dims(coord_embed, axis=2)
    pos_coord = pos_coord / dim_t
    pos_coord_sin = ops.sin(pos_coord[:, :, 0::2])
    pos_coord_cos = ops.cos(pos_coord[:, :, 1::2])
    pos_coord_stacked = ops.stack([pos_coord_sin, pos_coord_cos], axis=3)
    shape = ops.shape(pos_coord_stacked)
    pos_coord = ops.reshape(pos_coord_stacked, (shape[0], shape[1], -1))
    return pos_coord


def gen_sineembed_for_position(pos_tensor, dim=128):
    scale = 2 * math.pi
    dim_t = ops.arange(dim, dtype=pos_tensor.dtype)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = _process_coord_embedding(x_embed, dim_t)
    pos_y = _process_coord_embedding(y_embed, dim_t)

    num_coords = ops.shape(pos_tensor)[-1]
    if num_coords == 2:
        pos = ops.concatenate([pos_y, pos_x], axis=2)
    elif num_coords == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        h_embed = pos_tensor[:, :, 3] * scale
        pos_w = _process_coord_embedding(w_embed, dim_t)
        pos_h = _process_coord_embedding(h_embed, dim_t)
        pos = ops.concatenate([pos_y, pos_x, pos_w, pos_h], axis=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(num_coords))
    return pos


def gen_encoder_output_proposals(
    memory, memory_padding_mask, spatial_shapes, masks_list=None, unsigmoid=True
):
    N_ = ops.shape(memory)[0]
    proposals = []
    _cur = 0

    for lvl, (H_, W_) in enumerate(spatial_shapes):
        if masks_list is not None and len(masks_list) > lvl:
            mask_flatten_ = masks_list[lvl]
            mask_flatten_ = ops.reshape(mask_flatten_, (N_, H_, W_, 1))
            valid_H = ops.sum(
                ops.cast(ops.logical_not(mask_flatten_[:, :, 0, 0]), "float32"), axis=1
            )
            valid_W = ops.sum(
                ops.cast(ops.logical_not(mask_flatten_[:, 0, :, 0]), "float32"), axis=1
            )
        elif memory_padding_mask is not None:
            length_ = H_ * W_
            mask_flatten_ = ops.slice(memory_padding_mask, [0, _cur], [N_, length_])
            mask_flatten_ = ops.reshape(mask_flatten_, (N_, H_, W_, 1))
            valid_H = ops.sum(
                ops.cast(ops.logical_not(mask_flatten_[:, :, 0, 0]), "float32"), axis=1
            )
            valid_W = ops.sum(
                ops.cast(ops.logical_not(mask_flatten_[:, 0, :, 0]), "float32"), axis=1
            )
            _cur = _cur + length_
        else:
            valid_H = ops.ones((N_,), dtype="float32") * ops.cast(H_, "float32")
            valid_W = ops.ones((N_,), dtype="float32") * ops.cast(W_, "float32")

        grid_y, grid_x = ops.meshgrid(
            ops.arange(0, H_, dtype="float32"),
            ops.arange(0, W_, dtype="float32"),
            indexing="ij",
        )
        grid = ops.concatenate(
            [ops.expand_dims(grid_x, axis=-1), ops.expand_dims(grid_y, axis=-1)],
            axis=-1,
        )
        scale = ops.concatenate(
            [ops.expand_dims(valid_W, axis=-1), ops.expand_dims(valid_H, axis=-1)],
            axis=1,
        )
        scale = ops.reshape(scale, (N_, 1, 1, 2))
        scale = ops.cast(scale, dtype="float32")
        grid = (
            ops.broadcast_to(ops.expand_dims(grid, axis=0), (N_, H_, W_, 2)) + 0.5
        ) / scale
        wh = ops.ones_like(grid) * 0.05 * (2.0**lvl)
        proposal = ops.concatenate([grid, wh], axis=-1)
        proposal = ops.reshape(proposal, (N_, -1, 4))
        proposals.append(proposal)

    output_proposals = ops.concatenate(proposals, axis=1)
    output_proposals_valid = ops.all(
        ops.logical_and(output_proposals > 0.01, output_proposals < 0.99),
        axis=-1,
        keepdims=True,
    )

    if unsigmoid:
        eps = 1e-6
        output_proposals = ops.clip(output_proposals, eps, 1 - eps)
        output_proposals = ops.log(output_proposals / (1 - output_proposals))
        fill_val = 1e9
        if memory_padding_mask is not None:
            mask_expanded = ops.expand_dims(memory_padding_mask, axis=-1)
            output_proposals = ops.where(mask_expanded, float("inf"), output_proposals)

        output_proposals = ops.where(
            ops.logical_not(output_proposals_valid), float("inf"), output_proposals
        )
    else:
        if memory_padding_mask is not None:
            mask_expanded = ops.expand_dims(memory_padding_mask, axis=-1)
            output_proposals = ops.where(mask_expanded, 0.0, output_proposals)
        output_proposals = ops.where(
            ops.logical_not(output_proposals_valid), 0.0, output_proposals
        )

    output_memory = memory
    if memory_padding_mask is not None:
        mask_expanded = ops.expand_dims(memory_padding_mask, axis=-1)
        output_memory = ops.where(mask_expanded, 0.0, output_memory)
    output_memory = ops.where(
        ops.logical_not(output_proposals_valid), 0.0, output_memory
    )

    return ops.cast(output_memory, memory.dtype), ops.cast(
        output_proposals, memory.dtype
    )


def _get_clones(module, N):
    return [copy.deepcopy(module) for i in range(N)]


def _get_activation_fn(activation):
    if activation == "relu":
        return activations.relu
    if activation == "gelu":
        return activations.gelu
    if activation == "glu":
        return activations.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def bilinear_grid_sample(image, grid):
    """
    Performs a bilinear grid sample on a batch of images.
    Matches PyTorch grid_sample with align_corners=False and padding_mode='zeros'.
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

    x = grid[..., 0]
    y = grid[..., 1]

    # Map grid coordinates to pixel coordinates
    x_pix = ((x + 1.0) * W_float - 1.0) / 2.0
    y_pix = ((y + 1.0) * H_float - 1.0) / 2.0

    x0 = ops.floor(x_pix)
    x1 = x0 + 1.0
    y0 = ops.floor(y_pix)
    y1 = y0 + 1.0

    wa = (x1 - x_pix) * (y1 - y_pix)
    wb = (x1 - x_pix) * (y_pix - y0)
    wc = (x_pix - x0) * (y1 - y_pix)
    wd = (x_pix - x0) * (y_pix - y0)

    x0_i = ops.cast(x0, "int32")
    x1_i = ops.cast(x1, "int32")
    y0_i = ops.cast(y0, "int32")
    y1_i = ops.cast(y1, "int32")

    valid_x0 = (x0_i >= 0) & (x0_i < W)
    valid_x1 = (x1_i >= 0) & (x1_i < W)
    valid_y0 = (y0_i >= 0) & (y0_i < H)
    valid_y1 = (y1_i >= 0) & (y1_i < H)

    mask_a = ops.cast(valid_x0 & valid_y0, image.dtype)
    mask_b = ops.cast(valid_x0 & valid_y1, image.dtype)
    mask_c = ops.cast(valid_x1 & valid_y0, image.dtype)
    mask_d = ops.cast(valid_x1 & valid_y1, image.dtype)

    x0_c = ops.clip(x0_i, 0, W - 1)
    x1_c = ops.clip(x1_i, 0, W - 1)
    y0_c = ops.clip(y0_i, 0, H - 1)
    y1_c = ops.clip(y1_i, 0, H - 1)

    img_trans = ops.transpose(image, (0, 2, 3, 1))
    flat_img = ops.reshape(img_trans, (-1, C))

    # Add dummy row for safe gather (handle empty/out-of-bounds safety)
    dummy = ops.zeros((1, C), dtype=flat_img.dtype)
    flat_img_safe = ops.concatenate([flat_img, dummy], axis=0)
    max_index = ops.shape(flat_img)[0]

    batch_range = ops.arange(N, dtype="int32")
    batch_offset = ops.reshape(batch_range, (N, 1, 1)) * (H * W)

    def get_pixel_value_masked(x_idx, y_idx, mask):
        flat_indices = batch_offset + y_idx * W + x_idx
        flat_indices = ops.reshape(flat_indices, (-1,))
        flat_indices = ops.clip(flat_indices, 0, max_index)
        vals = ops.take(flat_img_safe, flat_indices, axis=0)
        vals = ops.reshape(vals, (N, H_out, W_out, C))
        vals = ops.transpose(vals, (0, 3, 1, 2))
        return vals * ops.expand_dims(mask, 1)

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
    value_spatial_shapes_list=None,
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


def _is_power_of_2(n):
    return (n & (n - 1) == 0) and n != 0


@register_keras_serializable(package="MyLayers")
class MLP(keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers

        h = [hidden_dim] * (num_layers - 1)
        specs = list(zip([input_dim] + h, h + [output_dim]))

        self.mlp_layers = keras.Sequential()

        for in_d, out_d in specs:
            layer = layers.Dense(units=out_d, use_bias=True)
            layer.build((None, in_d))
            self.mlp_layers.add(layer)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        for i, layer in enumerate(self.mlp_layers.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = keras.activations.relu(x)
        return x


@register_keras_serializable(package="MyLayers")
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


@register_keras_serializable(package="MyLayers")
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
        input_spatial_shapes_list=None,
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


@register_keras_serializable(package="MyLayers")
class TransformerDecoder(keras.Model):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        d_model=256,
        lite_refpoint_refine=False,
        bbox_reparam=False,
    ):
        super().__init__()
        self.decoder_layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.lite_refpoint_refine = lite_refpoint_refine
        self.bbox_reparam = bbox_reparam

        if hasattr(decoder_layer, "bbox_embed"):
            self.bbox_embed = decoder_layer.bbox_embed
        else:
            self.bbox_embed = None

        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        self._export = False

    def export(self):
        self._export = True

    def refpoints_refine(self, refpoints_unsigmoid, new_refpoints_delta):
        if self.bbox_reparam:
            new_refpoints_cxcy = (
                new_refpoints_delta[..., :2] * refpoints_unsigmoid[..., 2:]
                + refpoints_unsigmoid[..., :2]
            )
            new_refpoints_wh = (
                ops.exp(new_refpoints_delta[..., 2:]) * refpoints_unsigmoid[..., 2:]
            )
            new_refpoints_unsigmoid = ops.concatenate(
                [new_refpoints_cxcy, new_refpoints_wh], axis=-1
            )
        else:
            new_refpoints_unsigmoid = refpoints_unsigmoid + new_refpoints_delta
        return new_refpoints_unsigmoid

    def build(self, input_shape):
        super().build(input_shape)

    def call(
        self,
        tgt,
        memory,
        tgt_mask,
        memory_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask,
        pos,
        refpoints_unsigmoid,
        level_start_index,
        spatial_shapes,
        valid_ratios,
        training=None,
        memory_list=None,
        memory_mask_list=None,
        spatial_shapes_list=None,
    ):
        output = tgt
        intermediate = []

        # Match PyTorch: Initialize with input refpoints
        hs_refpoints_unsigmoid = [refpoints_unsigmoid]

        def get_reference(refpoints):
            obj_center = refpoints[..., :4]
            if self._export:
                query_sine_embed = gen_sineembed_for_position(
                    obj_center, self.d_model / 2
                )
                refpoints_input = obj_center[:, :, None]
            else:
                # valid_ratios is (B, L, 2) -> [W_ratio, H_ratio]
                scales = ops.concatenate(
                    [valid_ratios, valid_ratios], axis=-1
                )  # (B, L, 4)
                scales = ops.expand_dims(scales, axis=1)  # (B, 1, L, 4)

                # obj_center is (B, Nq, 4). Expand to (B, Nq, 1, 4) for broadcasting with L
                refpoints_input = ops.expand_dims(obj_center, axis=2) * scales

                # Sine embed uses coordinates relative to the first level (DINO standard)
                query_sine_embed = gen_sineembed_for_position(
                    refpoints_input[:, :, 0, :], self.d_model / 2
                )
            query_pos = self.ref_point_head(query_sine_embed)
            return obj_center, refpoints_input, query_pos, query_sine_embed

        if self.lite_refpoint_refine:
            if self.bbox_reparam:
                obj_center, refpoints_input, query_pos, query_sine_embed = (
                    get_reference(refpoints_unsigmoid)
                )
            else:
                obj_center, refpoints_input, query_pos, query_sine_embed = (
                    get_reference(ops.sigmoid(refpoints_unsigmoid))
                )

        for layer_id, layer in enumerate(self.decoder_layers):
            if not self.lite_refpoint_refine:
                if self.bbox_reparam:
                    obj_center, refpoints_input, query_pos, query_sine_embed = (
                        get_reference(refpoints_unsigmoid)
                    )
                else:
                    obj_center, refpoints_input, query_pos, query_sine_embed = (
                        get_reference(ops.sigmoid(refpoints_unsigmoid))
                    )

            pos_transformation = 1
            query_pos = query_pos * pos_transformation

            output = layer(
                output,
                memory,
                training=training,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                query_sine_embed=query_sine_embed,
                is_first=(layer_id == 0),
                reference_points=refpoints_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                memory_list=memory_list,
                memory_mask_list=memory_mask_list,
                spatial_shapes_list=spatial_shapes_list,
            )

            if not self.lite_refpoint_refine:
                if self.bbox_embed is None:
                    raise ValueError("bbox_embed is not initialized.")
                new_refpoints_delta = self.bbox_embed(output)
                new_refpoints_unsigmoid = self.refpoints_refine(
                    refpoints_unsigmoid, new_refpoints_delta
                )


                if layer_id != self.num_layers - 1:
                    hs_refpoints_unsigmoid.append(new_refpoints_unsigmoid)

                # Update refpoints for next layer and detach gradients
                refpoints_unsigmoid = ops.stop_gradient(new_refpoints_unsigmoid)

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self._export:
                hs = intermediate[-1]
                ref = (
                    hs_refpoints_unsigmoid[-1]
                    if self.bbox_embed
                    else refpoints_unsigmoid
                )
                return hs, ref
            if self.bbox_embed is not None:
                return [
                    ops.stack(intermediate, axis=0),
                    ops.stack(hs_refpoints_unsigmoid, axis=0),
                ]
            else:
                return [
                    ops.stack(intermediate, axis=0),
                    ops.expand_dims(refpoints_unsigmoid, axis=0),
                ]

        return ops.expand_dims(output, axis=0)


@register_keras_serializable(package="MyLayers")
class TransformerDecoderLayer(keras.Model):

    def __init__(
        self,
        d_model,
        sa_nhead,
        ca_nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        group_detr=1,
        num_feature_levels=4,
        dec_n_points=4,
        skip_self_attn=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.sa_nhead = sa_nhead
        self.ca_nhead = ca_nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.activation_name = activation
        self.normalize_before = normalize_before
        self.group_detr = group_detr
        self.num_feature_levels = num_feature_levels
        self.dec_n_points = dec_n_points
        self.skip_self_attn = skip_self_attn

        self.self_attn = layers.MultiHeadAttention(
            num_heads=sa_nhead, key_dim=d_model // sa_nhead, dropout=dropout
        )
        self.dropout1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)

        self.cross_attn = MSDeformAttn(
            d_model,
            n_levels=num_feature_levels,
            n_heads=ca_nhead,
            n_points=dec_n_points,
        )

        self.nhead = ca_nhead
        self.linear1 = layers.Dense(units=dim_feedforward)
        self.ffn_dropout = layers.Dropout(dropout)
        self.linear2 = layers.Dense(units=d_model)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.norm3 = layers.LayerNormalization(epsilon=1e-5)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.group_detr = group_detr

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def build(self, input_shape):
        super().build(input_shape)

    def call(
        self,
        tgt,
        memory,
        training=None,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
        query_sine_embed=None,
        is_first=False,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        memory_list=None,
        memory_mask_list=None,
        spatial_shapes_list=None,
    ):
        bs, num_queries, _ = tgt.shape
        q = k = tgt + query_pos
        v = tgt

        if training and self.group_detr > 1:
            chunk_size = num_queries // self.group_detr

            def split_heads(x):
                # (B, Nq, D) -> (B, G, Nq//G, D) -> (B*G, Nq//G, D)
                x = ops.reshape(x, (bs, self.group_detr, chunk_size, -1))
                return ops.reshape(
                    ops.transpose(x, (1, 0, 2, 3)),
                    (bs * self.group_detr, chunk_size, -1),
                )

            q = split_heads(q)
            k = split_heads(k)
            v = split_heads(v)

        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = ops.logical_not(tgt_key_padding_mask)

        tgt2 = self.self_attn(
            query=q,
            value=v,
            key=k,
            attention_mask=tgt_mask if tgt_mask is not None else tgt_key_padding_mask,
            training=training,
        )

        if training:
            shape = ops.shape(tgt2)
            total_batch_dim = shape[0]
            seq_len = shape[1]
            embed_dim = shape[2]
            n_chunks = total_batch_dim // bs
            reshaped_tgt = ops.reshape(tgt2, (n_chunks, bs, seq_len, embed_dim))
            transposed_tgt = ops.transpose(reshaped_tgt, (1, 0, 2, 3))
            new_seq_len = n_chunks * seq_len
            tgt2 = ops.reshape(transposed_tgt, (bs, new_seq_len, embed_dim))

        tgt = tgt + self.dropout1(tgt2, training=training)
        tgt = self.norm1(tgt)

        # Pass memory_list to cross_attn
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            memory_key_padding_mask,
            input_flatten_list=memory_list,
            input_padding_mask_list=memory_mask_list,
            input_spatial_shapes_list=spatial_shapes_list,
        )

        tgt = tgt + self.dropout2(tgt2, training=training)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(
            self.ffn_dropout(self.activation(self.linear1(tgt)), training=training)
        )
        tgt = tgt + self.dropout3(tgt2, training=training)
        tgt = self.norm3(tgt)
        return tgt

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "sa_nhead": self.sa_nhead,
                "ca_nhead": self.ca_nhead,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout_rate,
                "activation": self.activation_name,
                "normalize_before": self.normalize_before,
                "group_detr": self.group_detr,
                "num_feature_levels": self.num_feature_levels,
                "dec_n_points": self.dec_n_points,
                "skip_self_attn": self.skip_self_attn,
            }
        )
        return config


@register_keras_serializable(package="MyLayers")
class Transformer(keras.Model):
    def __init__(
        self,
        d_model=512,
        sa_nhead=8,
        ca_nhead=8,
        num_queries=300,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        group_detr=1,
        two_stage=False,
        num_feature_levels=4,
        dec_n_points=4,
        lite_refpoint_refine=False,
        decoder_norm_type="LN",
        bbox_reparam=False,
        enc_out_class_embed=None,
        enc_out_bbox_embed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_queries = num_queries
        self.d_model = d_model
        self.dec_layers = num_decoder_layers
        self.group_detr = group_detr
        self.num_feature_levels = num_feature_levels
        self.bbox_reparam = bbox_reparam
        self.two_stage = two_stage

        # --- Decoder Construction ---
        decoder_layer = TransformerDecoderLayer(
            d_model,
            sa_nhead,
            ca_nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            group_detr=group_detr,
            num_feature_levels=num_feature_levels,
            dec_n_points=dec_n_points,
            skip_self_attn=False,
        )

        if decoder_norm_type == "LN":
            decoder_norm = layers.LayerNormalization(epsilon=1e-5)
        else:
            decoder_norm = layers.Identity()

        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            lite_refpoint_refine=lite_refpoint_refine,
            bbox_reparam=bbox_reparam,
        )

        # --- Two Stage Logic Components ---
        if two_stage:
            self.enc_output = [
                layers.Dense(d_model, name=f"enc_output_{i}") for i in range(group_detr)
            ]
            self.enc_output_norm = [
                layers.LayerNormalization(epsilon=1e-5, name=f"enc_output_norm_{i}")
                for i in range(group_detr)
            ]

        self.enc_out_class_embed = enc_out_class_embed
        self.enc_out_bbox_embed = enc_out_bbox_embed

        # --- Link decoder bbox_embed if provided ---
        if self.decoder.bbox_embed is None and self.enc_out_bbox_embed is not None:
            if isinstance(self.enc_out_bbox_embed, (list, tuple)):
                self.decoder.bbox_embed = self.enc_out_bbox_embed[0]
            else:
                self.decoder.bbox_embed = self.enc_out_bbox_embed

        self._export = False

    def export(self):
        self._export = True
        self.decoder.export()

    def get_valid_ratio(self, mask):
        _, H, W = ops.shape(mask)[0], ops.shape(mask)[1], ops.shape(mask)[2]

        valid_H = ops.sum(ops.cast(ops.logical_not(mask[:, :, 0]), "float32"), axis=1)
        valid_W = ops.sum(ops.cast(ops.logical_not(mask[:, 0, :]), "float32"), axis=1)

        valid_ratio_h = valid_H / ops.cast(H, "float32")
        valid_ratio_w = valid_W / ops.cast(W, "float32")

        valid_ratio = ops.stack([valid_ratio_w, valid_ratio_h], axis=-1)
        return valid_ratio

    def build(self, input_shape):
        super().build(input_shape)

    def call(
        self,
        srcs,
        masks,
        pos_embeds,
        refpoint_embed=None,
        query_feat=None,
        training=None,
    ):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = []

        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            # --- Robust BCHW to BHWC conversion ---
            shape = ops.shape(src)
            if shape[1] == self.d_model and shape[3] != self.d_model:
                src = ops.transpose(src, (0, 2, 3, 1))

            shape_p = ops.shape(pos_embed)
            if shape_p[1] == self.d_model and shape_p[3] != self.d_model:
                pos_embed = ops.transpose(pos_embed, (0, 2, 3, 1))

            shape_s = ops.shape(src)
            bs, h, w, c = shape_s[0], shape_s[1], shape_s[2], shape_s[3]

            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = ops.reshape(src, (bs, -1, c))
            pos_embed = ops.reshape(pos_embed, (bs, -1, c))

            src_flatten.append(src)
            lvl_pos_embed_flatten.append(pos_embed)

            if masks is not None:
                mask = masks[lvl]
                valid_ratios.append(self.get_valid_ratio(mask))
                mask = ops.reshape(mask, (bs, -1))
                mask_flatten.append(mask)

        memory = ops.concatenate(src_flatten, axis=1)
        memory_list = src_flatten
        memory_mask_list = mask_flatten if masks is not None else None

        if masks is not None:
            mask_flatten = ops.concatenate(mask_flatten, axis=1)
            valid_ratios = ops.stack(valid_ratios, axis=1)
        else:
            mask_flatten = None
            valid_ratios = None

        lvl_pos_embed_flatten = ops.concatenate(lvl_pos_embed_flatten, axis=1)
        spatial_shapes_tensor = ops.convert_to_tensor(spatial_shapes, dtype="int32")

        hw_counts = ops.prod(spatial_shapes_tensor, axis=1)
        cumulative = ops.cumsum(hw_counts, axis=0)
        zero = ops.zeros((1,), dtype="int32")
        level_start_index = ops.concatenate([zero, cumulative[:-1]], axis=0)
        level_start_index = ops.cast(level_start_index, "int64")

        memory_ts = None
        boxes_ts = None

        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory,
                mask_flatten,
                spatial_shapes,
                masks_list=memory_mask_list,
                unsigmoid=not self.bbox_reparam,
            )

            refpoint_embed_ts = []
            memory_ts_list = []
            boxes_ts_list = []

            current_group_detr = self.group_detr if training else 1

            for g_idx in range(current_group_detr):
                output_memory_gidx = self.enc_output_norm[g_idx](
                    self.enc_output[g_idx](output_memory)
                )

                enc_outputs_class_unselected_gidx = self.enc_out_class_embed[g_idx](
                    output_memory_gidx
                )

                if self.bbox_reparam:
                    enc_outputs_coord_delta_gidx = self.enc_out_bbox_embed[g_idx](
                        output_memory_gidx
                    )
                    enc_outputs_coord_cxcy_gidx = (
                        enc_outputs_coord_delta_gidx[..., :2]
                        * output_proposals[..., 2:]
                        + output_proposals[..., :2]
                    )
                    enc_outputs_coord_wh_gidx = (
                        ops.exp(enc_outputs_coord_delta_gidx[..., 2:])
                        * output_proposals[..., 2:]
                    )
                    enc_outputs_coord_unselected_gidx = ops.concatenate(
                        [enc_outputs_coord_cxcy_gidx, enc_outputs_coord_wh_gidx],
                        axis=-1,
                    )
                else:
                    enc_outputs_coord_unselected_gidx = (
                        self.enc_out_bbox_embed[g_idx](output_memory_gidx)
                        + output_proposals
                    )

                # 1. Get raw scores
                class_scores = ops.max(enc_outputs_class_unselected_gidx, axis=-1)

                # 2. Apply masking (Original code)
                if mask_flatten is not None:
                    class_scores = ops.where(mask_flatten, -1e9, class_scores)

                # --- Add Deterministic Tie-Breaker ---
                seq_len = ops.shape(class_scores)[1]
                index_range = ops.cast(ops.arange(seq_len), dtype=class_scores.dtype)
                tie_breaker = index_range * 1e-9
                tie_breaker = ops.expand_dims(tie_breaker, axis=0)

                # Apply tie-breaker
                class_scores = class_scores + tie_breaker

                topk = min(self.num_queries, seq_len)
                _, topk_indices = ops.top_k(class_scores, k=topk)

                gather_indices = ops.expand_dims(topk_indices, axis=-1)

                refpoint_embed_gidx_undetach = ops.take_along_axis(
                    enc_outputs_coord_unselected_gidx, gather_indices, axis=1
                )
                refpoint_embed_gidx = ops.stop_gradient(refpoint_embed_gidx_undetach)

                tgt_undetach_gidx = ops.take_along_axis(
                    output_memory_gidx, topk_indices[..., None], axis=1
                )

                refpoint_embed_ts.append(refpoint_embed_gidx)
                memory_ts_list.append(tgt_undetach_gidx)
                boxes_ts_list.append(refpoint_embed_gidx_undetach)

            refpoint_embed_ts = ops.concatenate(refpoint_embed_ts, axis=1)
            memory_ts = ops.concatenate(memory_ts_list, axis=1)
            boxes_ts = ops.concatenate(boxes_ts_list, axis=1)

        if self.dec_layers > 0:
            nq_ = ops.shape(query_feat)[0]
            tgt = ops.broadcast_to(
                ops.expand_dims(query_feat, 0), (bs, nq_, self.d_model)
            )
            nq_ref_ = ops.shape(refpoint_embed)[0]
            refpoint_embed = ops.broadcast_to(
                ops.expand_dims(refpoint_embed, 0), (bs, nq_ref_, 4)
            )

            if self.two_stage:
                ts_len = ops.shape(refpoint_embed_ts)[1]
                refpoint_embed_ts_subset = refpoint_embed[..., :ts_len, :]
                refpoint_embed_subset = refpoint_embed[..., ts_len:, :]

                if self.bbox_reparam:
                    refpoint_embed_cxcy = (
                        refpoint_embed_ts_subset[..., :2] * refpoint_embed_ts[..., 2:]
                        + refpoint_embed_ts[..., :2]
                    )
                    refpoint_embed_wh = (
                        ops.exp(refpoint_embed_ts_subset[..., 2:])
                        * refpoint_embed_ts[..., 2:]
                    )
                    refpoint_embed_ts_subset = ops.concatenate(
                        [refpoint_embed_cxcy, refpoint_embed_wh], axis=-1
                    )
                else:
                    refpoint_embed_ts_subset = (
                        refpoint_embed_ts_subset + refpoint_embed_ts
                    )

                refpoint_embed = ops.concatenate(
                    [refpoint_embed_ts_subset, refpoint_embed_subset], axis=-2
                )

            hs, references = self.decoder(
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=mask_flatten,
                pos=lvl_pos_embed_flatten,
                refpoints_unsigmoid=refpoint_embed,
                level_start_index=level_start_index,
                spatial_shapes=spatial_shapes_tensor,
                valid_ratios=(
                    ops.cast(valid_ratios, memory.dtype)
                    if valid_ratios is not None
                    else None
                ),
                training=training,
                memory_list=memory_list,
                memory_mask_list=memory_mask_list,
                spatial_shapes_list=spatial_shapes,
            )
        else:
            hs = None
            references = None

        if self.two_stage:
            if self.bbox_reparam:
                return hs, references, memory_ts, boxes_ts
            else:
                return hs, references, memory_ts, ops.sigmoid(boxes_ts)

        return hs, references, None, None


def build_transformer(args):

    try:
        two_stage = args.two_stage
    except:
        two_stage = False

    return Transformer(
        d_model=args.hidden_dim,
        sa_nhead=args.sa_nheads,
        ca_nhead=args.ca_nheads,
        num_queries=args.num_queries,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
        group_detr=args.group_detr,
        two_stage=two_stage,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        lite_refpoint_refine=args.lite_refpoint_refine,
        decoder_norm_type=args.decoder_norm,
        bbox_reparam=args.bbox_reparam,
    )
