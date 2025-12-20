import keras
from keras import ops

from examples.dino_object_detection.models.transformer_decoder_head.MLP import MLP
from examples.dino_object_detection.models.transformer_decoder_head.utils import (
    gen_sineembed_for_position,
    _get_clones,
)


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
                refpoints_input = ops.expand_dims(obj_center, axis=2) * ops.expand_dims(
                    ops.concatenate([valid_ratios, valid_ratios], axis=-1), axis=1
                )

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

                # Match PyTorch: Append only if NOT the last layer
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
