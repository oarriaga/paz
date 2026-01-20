import math
import copy
from typing import Callable, Optional, List, Dict, Any, Tuple

import keras
from keras import layers
from keras import ops
from keras import initializers
from keras import activations
from keras.saving import register_keras_serializable

# -------------------------------------------------------------------------
# Assumed External Imports
# -------------------------------------------------------------------------
from examples.dino_object_detection.models.utils.box_ops import (
    box_cxcywh_to_xyxy,
    box_iou,
    generalized_box_iou,
)
from examples.dino_object_detection.models.utils.misc import (
    accuracy,
    get_world_size,
    nested_tensor_from_tensor_list,
    NestedTensor,
)
from examples.dino_object_detection.models.segmentation_head.utils import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from examples.dino_object_detection.models.backbone.dinov2_backbone_wrapper import (
    build_backbone,
)
from examples.dino_object_detection.models.transformer_decoder_head.transformer_kerass import (
    build_transformer,
)
from examples.dino_object_detection.models.segmentation_head.segmentation_head import (
    SegmentationHead,
)
from examples.dino_object_detection.models.matcher.matcher import build_matcher

# -------------------------------------------------------------------------
# Loss Functions
# -------------------------------------------------------------------------


def sigmoid_focal_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2.0
):
    prob = ops.sigmoid(inputs)
    ce_loss = ops.binary_crossentropy(targets, inputs, from_logits=True)
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = ce_loss * ops.power(1.0 - p_t, gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    return ops.sum(loss) / num_boxes


def sigmoid_varifocal_loss(
    inputs, targets, num_boxes, alpha: float = 0.75, gamma: float = 2.0
):
    prob = ops.sigmoid(inputs)
    positive_mask = ops.cast(targets > 0.0, "float32")
    negative_mask = ops.cast(targets <= 0.0, "float32")
    focal_weight = (
        targets * positive_mask
        + (1.0 - alpha) * ops.power(ops.abs(prob - targets), gamma) * negative_mask
    )
    ce_loss = ops.binary_crossentropy(targets, inputs, from_logits=True)
    loss = ce_loss * focal_weight
    return ops.sum(loss) / num_boxes


def position_supervised_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2.0
):
    prob = ops.sigmoid(inputs)
    ce_loss = ops.binary_crossentropy(targets, inputs, from_logits=True)
    loss = ce_loss * ops.power(ops.abs(targets - prob), gamma)
    if alpha >= 0:
        positive_mask = ops.cast(targets > 0.0, "float32")
        negative_mask = ops.cast(targets <= 0.0, "float32")
        alpha_t = alpha * positive_mask + (1.0 - alpha) * negative_mask
        loss = alpha_t * loss
    return ops.sum(loss) / num_boxes


def dice_loss(inputs, targets, num_masks):
    inputs = ops.sigmoid(inputs)
    inputs = ops.reshape(inputs, (ops.shape(inputs)[0], -1))
    targets = ops.reshape(targets, (ops.shape(targets)[0], -1))
    numerator = 2.0 * ops.sum(inputs * targets, axis=-1)
    denominator = ops.sum(inputs, axis=-1) + ops.sum(targets, axis=-1)
    loss = 1.0 - (numerator + 1.0) / (denominator + 1.0)
    return ops.sum(loss) / num_masks


def sigmoid_ce_loss(inputs, targets, num_masks):
    inputs_flat = ops.reshape(inputs, (ops.shape(inputs)[0], -1))
    targets_flat = ops.reshape(targets, (ops.shape(targets)[0], -1))
    loss = ops.binary_crossentropy(targets_flat, inputs_flat, from_logits=True)
    return ops.sum(ops.mean(loss, axis=1)) / num_masks


def calculate_uncertainty(logits):
    return -(ops.abs(logits))


# -------------------------------------------------------------------------
# Layers
# -------------------------------------------------------------------------


@register_keras_serializable(package="MyLayers")
class MLP(keras.Layer):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        output_kernel_initializer=None,
        output_bias_initializer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)

        layers_list = []
        # Hidden layers
        for _ in range(num_layers - 1):
            layers_list.append(layers.Dense(hidden_dim))

        # Output layer
        layers_list.append(
            layers.Dense(
                output_dim,
                kernel_initializer=(
                    output_kernel_initializer
                    if output_kernel_initializer
                    else "glorot_uniform"
                ),
                bias_initializer=(
                    output_bias_initializer if output_bias_initializer else "zeros"
                ),
            )
        )

        self.model = keras.Sequential(layers_list)

    # --- ADD THIS METHOD ---
    def build(self, input_shape):
        self.model.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        for i, layer in enumerate(self.model.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = activations.relu(x)
        return x


@register_keras_serializable(package="MyLayers")
class SetCriterion(keras.Layer):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        focal_alpha,
        losses,
        group_detr=1,
        sum_group_losses=False,
        use_varifocal_loss=False,
        use_position_supervised_loss=False,
        ia_bce_loss=False,
        mask_point_sample_ratio=16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.loss_names = losses
        self.focal_alpha = focal_alpha
        self.group_detr = group_detr
        self.sum_group_losses = sum_group_losses
        self.use_varifocal_loss = use_varifocal_loss
        self.use_position_supervised_loss = use_position_supervised_loss
        self.ia_bce_loss = ia_bce_loss
        self.mask_point_sample_ratio = mask_point_sample_ratio

    def call(self, outputs, targets):
        group_detr = self.group_detr
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets, group_detr=group_detr)

        if isinstance(targets, dict) and "labels" in targets:
            valid_mask = ops.cast(targets["labels"] != -1, "float32")
            total_boxes = ops.sum(valid_mask)
        else:
            total_boxes = sum(len(t["labels"]) for t in targets)
            total_boxes = ops.convert_to_tensor(total_boxes, dtype="float32")

        num_boxes = total_boxes
        if not self.sum_group_losses:
            num_boxes = num_boxes * group_detr
        world_size = ops.convert_to_tensor(get_world_size(), dtype="float32")
        num_boxes = ops.maximum(num_boxes / world_size, 1.0)

        losses = {}
        for loss in self.loss_names:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices_aux = self.matcher(aux_outputs, targets, group_detr=group_detr)
                for loss in self.loss_names:
                    kwargs = {}
                    if loss == "labels":
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_aux, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            indices_enc = self.matcher(enc_outputs, targets, group_detr=group_detr)
            for loss in self.loss_names:
                kwargs = {}
                if loss == "labels":
                    kwargs["log"] = False
                l_dict = self.get_loss(
                    loss, enc_outputs, targets, indices_enc, num_boxes, **kwargs
                )
                l_dict = {k + f"_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def _gather_targets(self, targets, indices, key):
        vals = []
        is_tensor_dict = isinstance(targets, dict)
        for i, (_, tgt_idx) in enumerate(indices):
            t = targets[key][i] if is_tensor_dict else targets[i][key]
            vals.append(ops.take(t, tgt_idx, axis=0))
        return ops.concatenate(vals, axis=0)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs["pred_logits"]
        idx_batch, idx_src = self._get_src_permutation_idx(indices)
        target_classes_o = ops.cast(
            self._gather_targets(targets, indices, "labels"), "int32"
        )
        src_boxes = ops.stop_gradient(
            ops.gather_nd(
                outputs["pred_boxes"], ops.stack([idx_batch, idx_src], axis=-1)
            )
        )
        target_boxes = self._gather_targets(targets, indices, "boxes")

        if self.ia_bce_loss:
            alpha, gamma = self.focal_alpha, 2.0
            ious = box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )[0]
            pos_ious = ops.diag(ious)
            prob = ops.sigmoid(src_logits)
            neg_weights = ops.power(prob, gamma)
            pos_weights = ops.zeros_like(prob)
            scatter_indices = ops.stack([idx_batch, idx_src, target_classes_o], axis=-1)
            t = ops.power(ops.gather_nd(prob, scatter_indices), alpha) * ops.power(
                pos_ious, 1.0 - alpha
            )
            t = ops.stop_gradient(ops.maximum(t, 0.01))
            pos_weights = ops.scatter_update(pos_weights, scatter_indices, t)
            neg_weights = ops.scatter_update(neg_weights, scatter_indices, 1.0 - t)
            loss_ce = (
                ops.sum(
                    neg_weights * src_logits
                    - ops.log_sigmoid(src_logits) * (pos_weights + neg_weights)
                )
                / num_boxes
            )
        elif self.use_position_supervised_loss:
            ious = box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )[0]
            pos_ious = ops.diag(ious)
            B, Q, C = (
                ops.shape(src_logits)[0],
                ops.shape(src_logits)[1],
                self.num_classes,
            )
            cls_iou_func_targets = ops.scatter_update(
                ops.zeros((B, Q, C), dtype=src_logits.dtype),
                ops.stack([idx_batch, idx_src, target_classes_o], axis=-1),
                pos_ious,
            )
            norm_cls_iou_func_targets = cls_iou_func_targets / (
                ops.max(cls_iou_func_targets, axis=-1, keepdims=True) + 1e-8
            )
            loss_ce = position_supervised_loss(
                src_logits,
                norm_cls_iou_func_targets,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
            ) * ops.cast(Q, "float32")
        elif self.use_varifocal_loss:
            ious = box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )[0]
            pos_ious = ops.diag(ious)
            B, Q, C = (
                ops.shape(src_logits)[0],
                ops.shape(src_logits)[1],
                self.num_classes,
            )
            cls_iou_targets = ops.scatter_update(
                ops.zeros((B, Q, C), dtype=src_logits.dtype),
                ops.stack([idx_batch, idx_src, target_classes_o], axis=-1),
                pos_ious,
            )
            loss_ce = sigmoid_varifocal_loss(
                src_logits, cls_iou_targets, num_boxes, alpha=self.focal_alpha, gamma=2
            ) * ops.cast(Q, "float32")
        else:
            B, Q = ops.shape(src_logits)[0], ops.shape(src_logits)[1]
            target_classes_onehot = ops.scatter_update(
                ops.zeros((B, Q, self.num_classes + 1), dtype=src_logits.dtype),
                ops.stack([idx_batch, idx_src, target_classes_o], axis=-1),
                ops.ones_like(target_classes_o, dtype=src_logits.dtype),
            )
            loss_ce = sigmoid_focal_loss(
                src_logits,
                target_classes_onehot[..., :-1],
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
            ) * ops.cast(Q, "float32")

        losses = {"loss_ce": loss_ce}
        if log:
            losses["class_error"] = (
                100
                - accuracy(
                    ops.gather_nd(src_logits, ops.stack([idx_batch, idx_src], axis=-1)),
                    target_classes_o,
                )[0]
            )
        return losses

    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = ops.stop_gradient(outputs["pred_logits"])
        tgt_lengths = (
            ops.sum(ops.cast(targets["labels"] != -1, "float32"), axis=1)
            if (isinstance(targets, dict) and "labels" in targets)
            else ops.convert_to_tensor(
                [len(t["labels"]) for t in targets], dtype="float32"
            )
        )
        card_pred = ops.sum(
            ops.cast(ops.max(ops.sigmoid(pred_logits), axis=-1) > 0.5, "float32"),
            axis=1,
        )
        return {"cardinality_error": ops.mean(ops.abs(card_pred - tgt_lengths))}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx_batch, idx_src = self._get_src_permutation_idx(indices)
        src_boxes = ops.gather_nd(
            outputs["pred_boxes"], ops.stack([idx_batch, idx_src], axis=-1)
        )
        target_boxes = self._gather_targets(targets, indices, "boxes")
        loss_bbox = ops.sum(ops.abs(src_boxes - target_boxes)) / num_boxes
        loss_giou = (
            ops.sum(
                1
                - ops.diag(
                    generalized_box_iou(
                        box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
                    )
                )
            )
            / num_boxes
        )
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def loss_masks(self, outputs, targets, indices, num_boxes):
        if "pred_masks" not in outputs:
            return {}
        idx_batch, idx_src = self._get_src_permutation_idx(indices)
        src_masks = ops.gather_nd(
            outputs["pred_masks"], ops.stack([idx_batch, idx_src], axis=-1)
        )
        if ops.size(src_masks) == 0:
            return {
                "loss_mask_ce": ops.sum(src_masks),
                "loss_mask_dice": ops.sum(src_masks),
            }
        target_masks = self._gather_targets(targets, indices, "masks")
        src_masks, target_masks = ops.expand_dims(src_masks, 1), ops.cast(
            ops.expand_dims(target_masks, 1), "float32"
        )
        num_points = ops.maximum(
            ops.shape(src_masks)[-2],
            (ops.shape(src_masks)[-2] * ops.shape(src_masks)[-1])
            // self.mask_point_sample_ratio,
        )
        point_coords = get_uncertain_point_coords_with_randomness(
            ops.stop_gradient(src_masks), calculate_uncertainty, num_points, 3, 0.75
        )
        point_labels = ops.squeeze(
            point_sample(
                target_masks, point_coords, align_corners=False, mode="nearest"
            ),
            axis=1,
        )
        point_logits = ops.squeeze(
            point_sample(src_masks, point_coords, align_corners=False), axis=1
        )
        return {
            "loss_mask_ce": sigmoid_ce_loss(point_logits, point_labels, num_boxes),
            "loss_mask_dice": dice_loss(point_logits, point_labels, num_boxes),
        }

    def _get_src_permutation_idx(self, indices):
        batch_idx_list, src_idx_list = [], []
        for i, (src, _) in enumerate(indices):
            batch_idx_list.append(ops.full_like(src, i, dtype="int32"))
            src_idx_list.append(ops.cast(src, "int32"))
        return ops.concatenate(batch_idx_list, axis=0), ops.concatenate(
            src_idx_list, axis=0
        )


@register_keras_serializable(package="MyLayers")
class LWDETR(keras.Model):
    def __init__(
        self,
        backbone,
        transformer,
        segmentation_head,
        num_classes,
        num_queries,
        aux_loss=False,
        group_detr=1,
        two_stage=False,
        lite_refpoint_refine=False,
        bbox_reparam=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_queries, self.transformer, self.backbone = (
            num_queries,
            transformer,
            backbone,
        )
        self.segmentation_head, self.aux_loss, self.group_detr = (
            segmentation_head,
            aux_loss,
            group_detr,
        )
        self.two_stage, self.lite_refpoint_refine, self.bbox_reparam = (
            two_stage,
            lite_refpoint_refine,
            bbox_reparam,
        )
        self.num_classes, self.hidden_dim = num_classes, transformer.d_model

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed = layers.Dense(
            num_classes,
            bias_initializer=initializers.Constant(bias_value),
            name="class_embed",
        )
        self.bbox_embed = MLP(
            self.hidden_dim,
            self.hidden_dim,
            4,
            3,
            output_kernel_initializer="zeros",
            output_bias_initializer="zeros",
            name="bbox_embed",
        )

        self.refpoint_embed = self.add_weight(
            shape=(num_queries * group_detr, 4),
            initializer="zeros",
            name="refpoint_embed",
        )
        self.query_feat = self.add_weight(
            shape=(num_queries * group_detr, self.hidden_dim),
            initializer="glorot_uniform",
            name="query_feat",
        )

        self.transformer.decoder.bbox_embed = (
            None if self.lite_refpoint_refine else self.bbox_embed
        )

        if self.two_stage:
            self.enc_out_bbox_embed = []
            self.enc_out_class_embed = []

            for i in range(group_detr):
                bbox_layer = MLP(
                    self.hidden_dim,
                    self.hidden_dim,
                    4,
                    3,
                    output_kernel_initializer="zeros",
                    output_bias_initializer="zeros",
                    name=f"enc_bbox_{i}",
                )
                setattr(self, f"enc_bbox_{i}", bbox_layer)
                self.enc_out_bbox_embed.append(bbox_layer)

                class_layer = layers.Dense(
                    num_classes,
                    bias_initializer=initializers.Constant(bias_value),
                    name=f"enc_class_{i}",
                )
                setattr(self, f"enc_class_{i}", class_layer)
                self.enc_out_class_embed.append(class_layer)

            self.transformer.enc_out_bbox_embed = self.enc_out_bbox_embed
            self.transformer.enc_out_class_embed = self.enc_out_class_embed

        self._export_mode = False

    def reinitialize_detection_head(self, num_classes):
        if not self.class_embed.built:
            self.class_embed.build((None, self.hidden_dim))
        old_k, old_b = self.class_embed.kernel, self.class_embed.bias
        new_layer = layers.Dense(num_classes, name="class_embed_new")
        new_layer.build((None, self.hidden_dim))
        num_repeats = int(math.ceil(num_classes / ops.shape(old_k)[1]))
        new_layer.kernel.assign(ops.tile(old_k, [1, num_repeats])[:, :num_classes])
        new_layer.bias.assign(ops.tile(old_b, [num_repeats])[:num_classes])
        self.class_embed = new_layer

        if self.two_stage:
            new_enc_list = []
            for idx, old_enc in enumerate(self.enc_out_class_embed):
                if not old_enc.built:
                    old_enc.build((None, self.hidden_dim))
                nl = layers.Dense(num_classes)
                nl.build((None, self.hidden_dim))
                nl.kernel.assign(new_layer.kernel)
                nl.bias.assign(new_layer.bias)

                # Update tracking attribute
                setattr(self, f"enc_class_{idx}", nl)
                new_enc_list.append(nl)

            self.enc_out_class_embed = new_enc_list
            self.transformer.enc_out_class_embed = new_enc_list

    def export(self):
        self._export_mode = True
        if hasattr(self.backbone, "export"):
            self.backbone.export()
        if hasattr(self.transformer, "export"):
            self.transformer.export()

    def call(self, samples, training=None):
        if self._export_mode:
            return self.call_export(samples)

        # Handle inputs
        if isinstance(samples, dict):
            img_tensors = samples["tensors"]
            img_masks = samples.get("mask", None)
        elif isinstance(samples, (list, tuple)):
            img_tensors = samples[0]
            img_masks = samples[1] if len(samples) > 1 else None
        else:
            img_tensors = samples
            img_masks = None

        if img_masks is None:
            img_masks = ops.zeros(ops.shape(img_tensors)[:-1], dtype="bool")

        # Pass pure tensor to backbone (Keras convention)
        features, poss = self.backbone(img_tensors)

        srcs, masks = [], []
        for feat in features:
            if hasattr(feat, "decompose"):
                s, m = feat.decompose()
            else:
                s = feat
                m = ops.zeros(ops.shape(s)[:-1], dtype="bool")
            srcs.append(s)
            masks.append(m)

        ref_w = (
            self.refpoint_embed if training else self.refpoint_embed[: self.num_queries]
        )
        feat_w = self.query_feat if training else self.query_feat[: self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, ref_w, feat_w, training=training
        )

        out = {}
        if hs is not None:
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy = (
                    outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:]
                    + ref_unsigmoid[..., :2]
                )
                outputs_coord_wh = (
                    ops.exp(outputs_coord_delta[..., 2:]) * ref_unsigmoid[..., 2:]
                )
                outputs_coord = ops.concatenate(
                    [outputs_coord_cxcy, outputs_coord_wh], axis=-1
                )
            else:
                outputs_coord = ops.sigmoid(self.bbox_embed(hs) + ref_unsigmoid)

            outputs_class = self.class_embed(hs)
            outputs_masks = (
                self.segmentation_head(
                    features[0].tensors, hs, ops.shape(img_tensors)[-2:]
                )
                if self.segmentation_head
                else None
            )

            out.update(
                {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
            )
            if outputs_masks is not None:
                out["pred_masks"] = outputs_masks[-1]
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(
                    outputs_class, outputs_coord, outputs_masks
                )

        if self.two_stage:
            group_detr = self.group_detr if training else 1
            hs_enc_list = ops.split(hs_enc, group_detr, axis=1)
            cls_enc = ops.concatenate(
                [
                    self.enc_out_class_embed[g](hs_enc_list[g])
                    for g in range(group_detr)
                ],
                axis=1,
            )
            out["enc_outputs"] = {"pred_logits": cls_enc, "pred_boxes": ref_enc}

        return out

    def call_export(self, tensors):
        m = ops.zeros(ops.shape(tensors)[:-1], dtype="bool")
        srcs, _, poss = self.backbone(tensors)
        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs,
            None,
            poss,
            self.refpoint_embed[: self.num_queries],
            self.query_feat[: self.num_queries],
        )
        if hs is not None:
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy = (
                    outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:]
                    + ref_unsigmoid[..., :2]
                )
                outputs_coord_wh = (
                    ops.exp(outputs_coord_delta[..., 2:]) * ref_unsigmoid[..., 2:]
                )
                outputs_coord = ops.concatenate(
                    [outputs_coord_cxcy, outputs_coord_wh], axis=-1
                )
            else:
                outputs_coord = ops.sigmoid(self.bbox_embed(hs) + ref_unsigmoid)

            outputs_masks = (
                self.segmentation_head(srcs[0], [hs], ops.shape(tensors)[-2:])[0]
                if self.segmentation_head
                else None
            )
            return outputs_coord[-1], self.class_embed(hs)[-1], outputs_masks
        return ref_enc, self.enc_out_class_embed[0](hs_enc)

    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_masks):
        return [
            {
                "pred_logits": outputs_class[i],
                "pred_boxes": outputs_coord[i],
                **(
                    {"pred_masks": outputs_masks[i]}
                    if outputs_masks is not None
                    else {}
                ),
            }
            for i in range(ops.shape(outputs_class)[0] - 1)
        ]


@register_keras_serializable(package="MyLayers")
class PostProcess(layers.Layer):
    def __init__(self, num_select=300, **kwargs):
        super().__init__(**kwargs)
        self.num_select = num_select

    def call(self, outputs, target_sizes):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        B, C = ops.shape(out_logits)[0], ops.shape(out_logits)[-1]
        prob_flat = ops.reshape(ops.sigmoid(out_logits), (B, -1))
        topk_values, topk_indexes = ops.top_k(prob_flat, k=self.num_select)
        gather_indices = ops.stack(
            [
                ops.tile(ops.expand_dims(ops.arange(B), 1), [1, self.num_select]),
                topk_indexes // C,
            ],
            axis=-1,
        )
        boxes = box_cxcywh_to_xyxy(
            ops.gather_nd(out_bbox, gather_indices)
        ) * ops.expand_dims(
            ops.stack(
                [
                    target_sizes[:, 1],
                    target_sizes[:, 0],
                    target_sizes[:, 1],
                    target_sizes[:, 0],
                ],
                axis=1,
            ),
            1,
        )
        results = {"scores": topk_values, "labels": topk_indexes % C, "boxes": boxes}
        if "pred_masks" in outputs:
            results["masks"] = ops.gather_nd(outputs["pred_masks"], gather_indices)
        return results


def build_model(args):
    num_classes = args.num_classes + 1
    if hasattr(args, "shape"):
        target_shape = args.shape
    elif hasattr(args, "resolution"):
        target_shape = (args.resolution, args.resolution)
    else:
        target_shape = (640, 640)

    backbone = build_backbone(
        encoder=args.encoder,
        vit_encoder_num_layers=args.vit_encoder_num_layers,
        pretrained_encoder=args.pretrained_encoder,
        window_block_indexes=args.window_block_indexes,
        drop_path=args.drop_path,
        out_channels=args.hidden_dim,
        out_feature_indexes=args.out_feature_indexes,
        projector_scale=args.projector_scale,
        use_cls_token=args.use_cls_token,
        hidden_dim=args.hidden_dim,
        position_embedding=args.position_embedding,
        freeze_encoder=args.freeze_encoder,
        layer_norm=args.layer_norm,
        target_shape=target_shape,
        rms_norm=args.rms_norm,
        backbone_lora=args.backbone_lora,
        force_no_pretrain=args.force_no_pretrain,
        gradient_checkpointing=args.gradient_checkpointing,
        load_dinov2_weights=args.pretrain_weights is None,
        patch_size=args.patch_size,
        num_windows=args.num_windows,
        positional_encoding_size=args.positional_encoding_size,
        num_register_tokens=getattr(args, "num_register_tokens", 0),
        init_values=getattr(args, "init_values", 1e-5),
    )

    if args.encoder_only:
        if isinstance(backbone, (tuple, list)):
            return backbone[0], None, None
        return backbone, None, None
    if args.backbone_only:
        return backbone, None, None

    transformer = build_transformer(args)
    segmentation_head = None
    if args.segmentation_head:
        segmentation_head = SegmentationHead(
            args.hidden_dim,
            args.dec_layers,
            downsample_ratio=args.mask_downsample_ratio,
        )

    model = LWDETR(
        backbone,
        transformer,
        segmentation_head,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        group_detr=args.group_detr,
        two_stage=args.two_stage,
        lite_refpoint_refine=args.lite_refpoint_refine,
        bbox_reparam=args.bbox_reparam,
    )
    return model


def build_criterion_and_postprocessors(args):
    matcher = build_matcher(args)
    weight_dict = {
        "loss_ce": args.cls_loss_coef,
        "loss_bbox": args.bbox_loss_coef,
        "loss_giou": args.giou_loss_coef,
    }
    if args.segmentation_head:
        weight_dict["loss_mask_ce"] = args.mask_ce_loss_coef
        weight_dict["loss_mask_dice"] = args.mask_dice_loss_coef
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        if args.two_stage:
            aux_weight_dict.update({k + f"_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    if args.segmentation_head:
        losses.append("masks")

    criterion = SetCriterion(
        num_classes=args.num_classes + 1,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=args.focal_alpha,
        losses=losses,
        group_detr=args.group_detr,
        sum_group_losses=getattr(args, "sum_group_losses", False),
        use_varifocal_loss=args.use_varifocal_loss,
        use_position_supervised_loss=args.use_position_supervised_loss,
        ia_bce_loss=args.ia_bce_loss,
        mask_point_sample_ratio=getattr(args, "mask_point_sample_ratio", 16),
    )
    postprocess = PostProcess(num_select=args.num_select)
    return criterion, postprocess
