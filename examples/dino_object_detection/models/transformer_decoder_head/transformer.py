import keras
from keras import layers, ops
import copy

from examples.dino_object_detection.models.transformer_decoder_head.transformer_decoder import (
    TransformerDecoder,
)
from examples.dino_object_detection.models.transformer_decoder_head.transformer_decoder_layer import (
    TransformerDecoderLayer,
)
from examples.dino_object_detection.models.transformer_decoder_head.utils import (
    gen_encoder_output_proposals,
)


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
        """
        srcs: list of tensors [bs, C, H, W]
        """
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = []

        # Iterate over feature levels
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            # Optimally get static shape to avoid passing Tracers to MSDeformAttn reshaping
            h, w = None, None
            if hasattr(src, "shape"):
                # KerasTensor or JAX Array often has a .shape property
                if src.shape[2] is not None:
                    h = src.shape[2]
                if src.shape[3] is not None:
                    w = src.shape[3]

            if h is None or w is None:
                # Fallback to symbolic shape if static is not available
                shape = ops.shape(src)
                h, w = shape[2], shape[3]

            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            # Reshape logic remains symbolic-safe
            shape_s = ops.shape(src)
            bs, c = shape_s[0], shape_s[1]

            src = ops.reshape(src, (bs, c, -1))
            src = ops.transpose(src, (0, 2, 1))

            pos_embed = ops.reshape(pos_embed, (bs, c, -1))
            pos_embed = ops.transpose(pos_embed, (0, 2, 1))

            lvl_pos_embed_flatten.append(pos_embed)
            src_flatten.append(src)

            if masks is not None:
                mask = masks[lvl]
                valid_ratios.append(self.get_valid_ratio(mask))
                mask = ops.reshape(mask, (bs, -1))
                mask_flatten.append(mask)

        memory = ops.concatenate(src_flatten, axis=1)

        # Preserve the list for passing to decoder (avoids dynamic slicing)
        memory_list = src_flatten
        memory_mask_list = mask_flatten if masks is not None else None

        if masks is not None:
            mask_flatten = ops.concatenate(mask_flatten, axis=1)
            valid_ratios = ops.stack(valid_ratios, axis=1)
        else:
            mask_flatten = None
            valid_ratios = None

        lvl_pos_embed_flatten = ops.concatenate(lvl_pos_embed_flatten, axis=1)

        # IMPORTANT: Create a tensor for internal ops, but use the list for utils
        # This converts the list (which might contain ints) to a tensor for ops that need tensors
        spatial_shapes_tensor = ops.convert_to_tensor(spatial_shapes, dtype="int32")

        hw_counts = ops.prod(spatial_shapes_tensor, axis=1)
        cumulative = ops.cumsum(hw_counts, axis=0)
        zero = ops.zeros((1,), dtype="int32")
        level_start_index = ops.concatenate([zero, cumulative[:-1]], axis=0)
        level_start_index = ops.cast(level_start_index, "int64")

        memory_ts = None
        boxes_ts = None

        if self.two_stage:
            # FIX: Pass 'spatial_shapes' (list), NOT 'spatial_shapes_tensor'
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes, unsigmoid=not self.bbox_reparam
            )

            refpoint_embed_ts = []
            memory_ts_list = []
            boxes_ts_list = []

            # Logic for Group DETR
            current_group_detr = self.group_detr if training else 1

            for g_idx in range(current_group_detr):
                # 1. Project memory
                output_memory_gidx = self.enc_output_norm[g_idx](
                    self.enc_output[g_idx](output_memory)
                )

                # 2. Class predictions
                enc_outputs_class_unselected_gidx = self.enc_out_class_embed[g_idx](
                    output_memory_gidx
                )

                # 3. BBox predictions
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

                # 4. Top-K selection
                # enc_outputs_class_unselected_gidx: (BS, L, NumClasses)
                class_scores = ops.max(enc_outputs_class_unselected_gidx, axis=-1)
                topk = min(self.num_queries, ops.shape(class_scores)[1])
                _, topk_indices = ops.top_k(class_scores, k=topk)

                refpoint_embed_gidx_undetach = ops.take_along_axis(
                    enc_outputs_coord_unselected_gidx,
                    topk_indices[..., None],
                    axis=1,
                )

                refpoint_embed_gidx = ops.stop_gradient(refpoint_embed_gidx_undetach)

                # Gather Memory (Targets)
                tgt_undetach_gidx = ops.take_along_axis(
                    output_memory_gidx, topk_indices[..., None], axis=1
                )

                refpoint_embed_ts.append(refpoint_embed_gidx)
                memory_ts_list.append(tgt_undetach_gidx)
                boxes_ts_list.append(refpoint_embed_gidx_undetach)

            refpoint_embed_ts = ops.concatenate(refpoint_embed_ts, axis=1)
            memory_ts = ops.concatenate(memory_ts_list, axis=1)
            boxes_ts = ops.concatenate(boxes_ts_list, axis=1)

        # --- Decoder Call ---
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
                # Merge Two-Stage proposals with initial queries
                ts_len = ops.shape(refpoint_embed_ts)[1]
                refpoint_embed_ts_subset = refpoint_embed[..., :ts_len, :]
                refpoint_embed_subset = refpoint_embed[..., ts_len:, :]

                if self.bbox_reparam:
                    refpoint_embed_cxcy = (
                        refpoint_embed_ts_subset[..., :2] * refpoint_embed_ts[..., 2:]
                    )
                    refpoint_embed_cxcy = (
                        refpoint_embed_cxcy + refpoint_embed_ts[..., :2]
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
                spatial_shapes_list=spatial_shapes,  # Pass the list (hopefully static ints)
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
