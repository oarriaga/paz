import keras
from keras import layers, ops, activations
import math
import copy


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
    """
    Generates proposals from encoder memory.
    Args:
        masks_list: Optional list of masks per level to avoid slicing symbolic tensors.
    """
    N_ = ops.shape(memory)[0]
    proposals = []
    _cur = 0

    for lvl, (H_, W_) in enumerate(spatial_shapes):
        # Use provided masks_list to avoid ops.slice with symbolic size
        if masks_list is not None and len(masks_list) > lvl:
            mask_flatten_ = masks_list[lvl]
            # Reshape to (N_, H_, W_, 1)
            mask_flatten_ = ops.reshape(mask_flatten_, (N_, H_, W_, 1))
            valid_H = ops.sum(ops.logical_not(mask_flatten_[:, :, 0, 0]), axis=1)
            valid_W = ops.sum(ops.logical_not(mask_flatten_[:, 0, :, 0]), axis=1)
        elif memory_padding_mask is not None:
            # Fallback (may fail tracing if H_ is symbolic)
            length_ = H_ * W_
            mask_flatten_ = ops.slice(memory_padding_mask, [0, _cur], [N_, length_])
            mask_flatten_ = ops.reshape(mask_flatten_, (N_, H_, W_, 1))
            valid_H = ops.sum(ops.logical_not(mask_flatten_[:, :, 0, 0]), axis=1)
            valid_W = ops.sum(ops.logical_not(mask_flatten_[:, 0, :, 0]), axis=1)
            _cur = _cur + length_
        else:
            valid_H = ops.ones((N_,), dtype="int32") * ops.cast(H_, "int32")
            valid_W = ops.ones((N_,), dtype="int32") * ops.cast(W_, "int32")

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
            [
                ops.expand_dims(valid_W, axis=-1),
                ops.expand_dims(valid_H, axis=-1),
            ],
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

        if memory_padding_mask is not None:
            output_proposals = ops.where(
                ops.expand_dims(memory_padding_mask, axis=-1),
                float("inf"),
                output_proposals,
            )
        output_proposals = ops.where(
            ops.logical_not(output_proposals_valid), float("inf"), output_proposals
        )
    else:
        if memory_padding_mask is not None:
            output_proposals = ops.where(
                ops.expand_dims(memory_padding_mask, axis=-1), 0.0, output_proposals
            )
        output_proposals = ops.where(
            ops.logical_not(output_proposals_valid), 0.0, output_proposals
        )

    output_memory = memory
    if memory_padding_mask is not None:
        output_memory = ops.where(
            ops.expand_dims(memory_padding_mask, axis=-1), 0.0, output_memory
        )
    output_memory = ops.where(
        ops.logical_not(output_proposals_valid), 0.0, output_memory
    )

    return (
        ops.cast(output_memory, memory.dtype),
        ops.cast(output_proposals, memory.dtype),
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
