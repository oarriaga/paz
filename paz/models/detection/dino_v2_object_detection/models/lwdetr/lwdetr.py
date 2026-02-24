import math
import copy
from typing import Callable, Optional, List, Dict

import keras
from keras import layers
from keras import ops
import numpy as np

# Adjust imports based on your actual directory structure
from paz.models.detection.dino_v2_object_detection.utils import box_ops
from paz.models.detection.dino_v2_object_detection.utils.misc import (
    NestedTensor,
    interpolate,
    inverse_sigmoid,
)


# Loss functions ported from original lwdetr.py
def sigmoid_focal_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = ops.sigmoid(inputs)
    # Binary cross entropy with logits
    # Keras binary_crossentropy(from_logits=True)
    ce_loss = ops.binary_crossentropy(targets, inputs, from_logits=True)
    ce_loss = (
        ops.expand_dims(ce_loss, axis=-1)
        if ops.ndim(ce_loss) < ops.ndim(inputs)
        else ce_loss
    )

    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return ops.sum(ops.mean(loss, axis=1)) / num_boxes


def sigmoid_varifocal_loss(
    inputs, targets, num_boxes, alpha: float = 0.75, gamma: float = 2
):
    """
    Varifocal Loss
    """
    prob = ops.sigmoid(inputs)
    focal_weight = targets * ops.cast(targets > 0.0, "float32") + (1 - alpha) * (
        ops.abs(prob - targets) ** gamma
    ) * ops.cast(targets <= 0.0, "float32")

    ce_loss = ops.binary_crossentropy(targets, inputs, from_logits=True)
    ce_loss = (
        ops.expand_dims(ce_loss, axis=-1)
        if ops.ndim(ce_loss) < ops.ndim(inputs)
        else ce_loss
    )

    loss = ce_loss * focal_weight
    return ops.sum(ops.mean(loss, axis=1)) / num_boxes


def position_supervised_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2
):
    prob = ops.sigmoid(inputs)
    ce_loss = ops.binary_crossentropy(targets, inputs, from_logits=True)
    ce_loss = (
        ops.expand_dims(ce_loss, axis=-1)
        if ops.ndim(ce_loss) < ops.ndim(inputs)
        else ce_loss
    )

    loss = ce_loss * (ops.abs(targets - prob) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * ops.cast(targets > 0.0, "float32") + (1 - alpha) * ops.cast(
            targets <= 0.0, "float32"
        )
        loss = alpha_t * loss

    return ops.sum(ops.mean(loss, axis=1)) / num_boxes


def dice_loss(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    """
    inputs = ops.sigmoid(inputs)
    inputs = ops.reshape(inputs, (ops.shape(inputs)[0], -1))
    targets = ops.reshape(targets, (ops.shape(targets)[0], -1))

    numerator = 2 * ops.sum(inputs * targets, axis=-1)
    denominator = ops.sum(inputs, axis=-1) + ops.sum(targets, axis=-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return ops.sum(loss) / num_masks


def sigmoid_ce_loss(inputs, targets, num_masks):
    loss = ops.binary_crossentropy(targets, inputs, from_logits=True)
    return ops.sum(ops.mean(loss, axis=1)) / num_masks


@keras.saving.register_keras_serializable(package="RFDETR")
class MLP(layers.Layer):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        h = [hidden_dim] * (num_layers - 1)
        self.layers_list = []

        dims = [input_dim] + h + [output_dim]
        for i in range(len(dims) - 1):
            self.layers_list.append(layers.Dense(dims[i + 1], name=f"dense_{i}"))

    def build(self, input_shape):
        # input_shape is (batch, input_dim)
        curr_dim = input_shape[-1]
        for layer in self.layers_list:
            layer.build((None, curr_dim))
            curr_dim = layer.units
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self._input_dim,
                "hidden_dim": self._hidden_dim,
                "output_dim": self._output_dim,
                "num_layers": self.num_layers,
            }
        )
        return config

    def call(self, x):
        for i, layer in enumerate(self.layers_list):
            x = layer(x)
            if i < self.num_layers - 1:
                x = ops.relu(x)
        return x


@keras.saving.register_keras_serializable(package="RFDETR")
class LWDETR(keras.Model):
    """This is the Group DETR v3 module that performs object detection"""

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
        """Initializes the model."""
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = transformer.d_model
        hidden_dim = self.hidden_dim
        self._aux_loss = aux_loss
        self._segmentation_head_config = (
            None  # segmentation_head is keras layer or None
        )

        # In Keras, we prefer explicit activation in layer or separate layer, but Linear implies just Dense
        self.class_embed = layers.Dense(num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.segmentation_head = segmentation_head

        query_dim = 4
        self.refpoint_embed = self.add_weight(
            name="refpoint_embed",
            shape=(num_queries * group_detr, query_dim),
            initializer="zeros",
        )
        self.query_feat = self.add_weight(
            name="query_feat",
            shape=(num_queries * group_detr, self.hidden_dim),
            initializer="glorot_uniform",  # Common for embedding-like weights
        )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.group_detr = group_detr

        self.lite_refpoint_refine = lite_refpoint_refine
        # Note: In PyTorch, transformer.decoder.bbox_embed is assigned.
        # In Keras, we should probably pass it or handle it in call.
        # For now, we mirror structure but properties might need care in Keras logic.
        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        self.bbox_reparam = bbox_reparam

        self.two_stage = two_stage

        # Creating shared layers for two_stage if needed
        # In PyTorch: copy.deepcopy(self.bbox_embed) for _ in range(group_detr)
        # In Keras, we create distinct layers.
        if self.two_stage:
            self.enc_out_bbox_embed = []
            self.enc_out_class_embed = []
            for _ in range(group_detr):
                # We need structurally identical MLPs/Dense
                # Best to instantiate new ones
                self.enc_out_bbox_embed.append(MLP(hidden_dim, hidden_dim, 4, 3))
                self.enc_out_class_embed.append(layers.Dense(num_classes))

            # Need to assign these to transformer because transformer calls them in two_stage
            self.transformer.enc_out_bbox_embed = self.enc_out_bbox_embed
            self.transformer.enc_out_class_embed = self.enc_out_class_embed

        self.built_flag = False

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone": keras.saving.serialize_keras_object(self.backbone),
                "transformer": keras.saving.serialize_keras_object(self.transformer),
                "segmentation_head": (
                    keras.saving.serialize_keras_object(self.segmentation_head)
                    if self.segmentation_head is not None
                    else None
                ),
                "num_classes": self.num_classes,
                "num_queries": self.num_queries,
                "aux_loss": self.aux_loss,
                "group_detr": self.group_detr,
                "two_stage": self.two_stage,
                "lite_refpoint_refine": self.lite_refpoint_refine,
                "bbox_reparam": self.bbox_reparam,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["backbone"] = keras.saving.deserialize_keras_object(config["backbone"])
        config["transformer"] = keras.saving.deserialize_keras_object(
            config["transformer"]
        )
        if config["segmentation_head"] is not None:
            config["segmentation_head"] = keras.saving.deserialize_keras_object(
                config["segmentation_head"]
            )
        return cls(**config)

    def build(self, input_shape):
        # Initialize weights if needed, or rely on lazy build
        if not self.class_embed.built:
            self.class_embed.build((None, self.hidden_dim))
        if not self.bbox_embed.built:
            self.bbox_embed.build((None, self.hidden_dim))

        if self.two_stage:
            for bbox, cls in zip(self.enc_out_bbox_embed, self.enc_out_class_embed):
                if not bbox.built:
                    bbox.build((None, self.hidden_dim))
                if not cls.built:
                    cls.build((None, self.hidden_dim))

        # Setting built_flag
        self.built_flag = True
        super().build(input_shape)

    def call(self, samples, training=False):
        # Unpack samples if it's a list/tuple
        if isinstance(samples, (list, tuple)) and len(samples) == 2:
            # (tensors, mask)
            tensors, mask = samples
        elif hasattr(samples, "tensors") and hasattr(samples, "mask"):
            tensors = samples.tensors
            mask = samples.mask
        else:
            # Assume just tensors
            tensors = samples
            mask = None

        # Backbone forward
        # backbone is a Joiner that wraps Backbone + PositionEmbeddingSine.
        # Returns (features, poss_all) where:
        #   features = list of (feat, mask) tuples
        #   poss_all = list of position encoding tensors
        features, poss_all = self.backbone(tensors, mask=mask, training=training)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            if isinstance(feat, (list, tuple)):
                src, m = feat
            elif hasattr(feat, "decompose"):
                src, m = feat.decompose()
            else:
                src = feat
                if mask is not None:
                    m = interpolate(
                        mask[:, None], size=ops.shape(src)[1:3], mode="nearest"
                    )[:, 0]
                else:
                    m = ops.cast(ops.zeros_like(src[..., 0]), "bool")

            srcs.append(src)
            masks.append(m)

        # Prepare embeddings
        if training:
            refpoint_embed_weight = self.refpoint_embed
            query_feat_weight = self.query_feat
        else:
            # only use one group in inference
            refpoint_embed_weight = self.refpoint_embed[: self.num_queries]
            query_feat_weight = self.query_feat[: self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs,
            masks,
            poss_all,
            refpoint_embed=refpoint_embed_weight,
            query_feat=query_feat_weight,
            training=training,
        )

        outputs_coord = None
        outputs_class = None
        outputs_masks = None

        # Pre-compute NCHW spatial features for segmentation head (if present)
        spatial_nchw = (
            ops.transpose(srcs[0], (0, 3, 1, 2))
            if self.segmentation_head is not None
            else None
        )

        if hs is not None:
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                # Correct reparameterization: CXCY delta scaled by WH, added to reference CXCY
                outputs_coord_cxcy = (
                    outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:]
                    + ref_unsigmoid[..., :2]
                )
                # WH delta exponentiated and scaled by reference WH
                outputs_coord_wh = (
                    ops.exp(outputs_coord_delta[..., 2:]) * ref_unsigmoid[..., 2:]
                )
                outputs_coord = ops.concatenate(
                    [outputs_coord_cxcy, outputs_coord_wh], axis=-1
                )
            else:
                outputs_coord = self.bbox_embed(hs) + ref_unsigmoid
                outputs_coord = ops.sigmoid(outputs_coord)

            outputs_class = self.class_embed(hs)

            if self.segmentation_head is not None:
                # Logic for segmentation head
                # input_shape should be (H, W) of the input tensor
                target_shape = (
                    ops.shape(tensors)[1:3]
                    if ops.ndim(tensors) == 4
                    else ops.shape(tensors)[0:2]
                )

                # The SegmentationHead.call expects (spatial_features, query_features, image_size)
                # spatial_nchw was pre-computed above (NHWC → NCHW)
                outputs_masks = self.segmentation_head(
                    spatial_nchw, hs, image_size=target_shape
                )

            out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
            if self.segmentation_head is not None:
                out["pred_masks"] = outputs_masks[-1]

            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(
                    outputs_class, outputs_coord, outputs_masks
                )

        if self.two_stage:
            group_detr = self.group_detr if training else 1
            hs_enc_list = ops.split(
                hs_enc, group_detr, axis=1
            )  # split on dim 1 (queries)? Check chunks

            # Actually in PyTorch chunk(group_detr, dim=1)

            cls_enc = []
            # We need to iterate over group_detr and apply corresponding enc_out_class_embed
            # hs_enc_list = [t1, t2...]
            for g_idx in range(group_detr):
                # We need to slice if we didn't split perfectly or just use split logic
                # ops.split returns list
                cls_enc_gidx = self.transformer.enc_out_class_embed[g_idx](
                    hs_enc_list[g_idx]
                )
                cls_enc.append(cls_enc_gidx)

            cls_enc = ops.concatenate(cls_enc, axis=1)

            masks_enc = None
            if self.segmentation_head is not None:
                # Compute masks for encoded features
                target_shape = (
                    ops.shape(tensors)[1:3]
                    if ops.ndim(tensors) == 4
                    else ops.shape(tensors)[0:2]
                )
                # Pass [hs_enc] as list to skip blocks logic in head if needed, or just hs_enc
                # spatial_features must be NCHW; srcs[0] is NHWC from Keras backbone
                masks_enc_list = self.segmentation_head(
                    spatial_nchw, [hs_enc], image_size=target_shape, skip_blocks=True
                )
                masks_enc = masks_enc_list[0]

            if hs is not None:
                out["enc_outputs"] = {"pred_logits": cls_enc, "pred_boxes": ref_enc}
                if self.segmentation_head is not None:
                    out["enc_outputs"]["pred_masks"] = masks_enc
            else:
                out = {"pred_logits": cls_enc, "pred_boxes": ref_enc}
                if self.segmentation_head is not None:
                    out["pred_masks"] = masks_enc

        return out

    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_masks):
        # outputs_class: [layers, B, Q, C]
        # Return list of dicts
        aux_outputs = []
        num_layers = ops.shape(outputs_class)[0]
        for i in range(num_layers - 1):
            d = {"pred_logits": outputs_class[i], "pred_boxes": outputs_coord[i]}
            if outputs_masks is not None:
                d["pred_masks"] = outputs_masks[i]
            aux_outputs.append(d)
        return aux_outputs


class SetCriterion(layers.Layer):
    """This class computes the loss for LWDETR."""

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        focal_alpha,
        loss_types,
        group_detr=1,
        sum_group_losses=False,
        use_varifocal_loss=False,
        use_position_supervised_loss=False,
        ia_bce_loss=False,
        mask_point_sample_ratio: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.loss_types = loss_types
        self.focal_alpha = focal_alpha
        self.group_detr = group_detr
        self.sum_group_losses = sum_group_losses
        self.use_varifocal_loss = use_varifocal_loss
        self.use_position_supervised_loss = use_position_supervised_loss
        self.ia_bce_loss = ia_bce_loss
        self.mask_point_sample_ratio = mask_point_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)

        # Get target classes corresponding to the selected indices
        # targets is list of dicts.
        target_classes_o = ops.concatenate(
            [ops.take(t["labels"], J, axis=0) for t, (_, J) in zip(targets, indices)]
        )

        # Default Focal Loss logic
        target_classes = ops.full(src_logits.shape[:2], self.num_classes, dtype="int64")
        # Scatter indices?
        # In Keras/Jax, scatter update is specific.
        # target_classes[idx] = target_classes_o

        # Helper to do scatter nd update
        # idx is (batch_idx, src_idx)
        # We need to stack them for gather_nd / scatter_nd
        indices_nd = ops.stack(idx, axis=-1)
        target_classes = ops.scatter_update(
            target_classes, indices_nd, target_classes_o
        )

        # One-hot
        # src_logits shape: (B, Q, num_classes)
        num_classes_logits = ops.shape(src_logits)[2]
        target_classes_onehot = ops.one_hot(target_classes, num_classes_logits + 1)
        target_classes_onehot = target_classes_onehot[..., :-1]

        loss_ce = sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            num_boxes,
            alpha=self.focal_alpha,
            gamma=2,
        ) * ops.cast(ops.shape(src_logits)[1], "float32")

        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = ops.take(
            outputs["pred_boxes"], ops.stack(idx, axis=-1)
        )  # gather_nd logic needed if direct indexing fails
        # Actually ops.take works on flattened or simple axis. For Multi-dim index:
        # src_boxes = ops.gather_nd(outputs['pred_boxes'], ops.stack(idx, axis=-1))
        # Wait, indices is list of tuples (src_idx, tgt_idx).
        # idx from _get_src... returns (batch_indices, src_indices)

        # Implement gathering using reshape + take as gather_nd is missing in keras.ops
        B, Q, C = ops.shape(outputs["pred_boxes"])
        outputs_boxes_flat = ops.reshape(outputs["pred_boxes"], (-1, C))
        # idx is (batch_idx, src_idx)
        flat_indices = idx[0] * Q + idx[1]
        src_boxes = ops.take(outputs_boxes_flat, flat_indices, axis=0)

        target_boxes = ops.concatenate(
            [ops.take(t["boxes"], i, axis=0) for t, (_, i) in zip(targets, indices)],
            axis=0,
        )

        loss_bbox = ops.abs(src_boxes - target_boxes)  # L1 loss
        loss_bbox = ops.sum(loss_bbox) / num_boxes

        loss_giou = 1 - ops.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        loss_giou = ops.sum(loss_giou) / num_boxes

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = ops.concatenate(
            [ops.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = ops.concatenate([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            # 'cardinality': self.loss_cardinality,
            "boxes": self.loss_boxes,
            # 'masks': self.loss_masks,
        }
        if loss not in loss_map:
            return {}
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def call(self, outputs, targets):
        group_detr = self.group_detr  # ignore training flag for now or pass it
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        indices = self.matcher(outputs_without_aux, targets, group_detr=group_detr)

        num_boxes = sum(len(t["labels"]) for t in targets)
        if not self.sum_group_losses:
            num_boxes = num_boxes * group_detr
        num_boxes = ops.cast(num_boxes, "float32")
        # Distributed reduce would go here

        losses = {}
        for loss in self.loss_types:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, group_detr=group_detr)
                for loss in self.loss_types:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


@keras.saving.register_keras_serializable(package="RFDETR")
class PostProcess(layers.Layer):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=300, **kwargs):
        super().__init__(**kwargs)
        self.num_select = num_select

    def get_config(self):
        config = super().get_config()
        config.update({"num_select": self.num_select})
        return config

    def call(self, outputs, target_sizes):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        # Avoid len() and as_list() on symbolic tensors
        # assert len(out_logits) == len(target_sizes)
        # assert target_sizes.shape[1] == 2

        prob = ops.sigmoid(out_logits)
        prob_flat = ops.reshape(prob, (ops.shape(out_logits)[0], -1))

        topk_values, topk_indexes = ops.top_k(prob_flat, self.num_select)

        scores = topk_values
        num_classes_logits = ops.shape(out_logits)[2]
        topk_boxes = topk_indexes // num_classes_logits
        labels = topk_indexes % num_classes_logits

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # Gather boxes
        # boxes has shape (B, Q, 4)
        # topk_boxes has shape (B, K)
        # We need to gather (B, K, 4)

        # Using batch gather or gather_nd
        # B = boxes.shape[0]
        # gathered_boxes = []
        # for i in range(B):
        #      gathered_boxes.append(ops.take(boxes[i], topk_boxes[i], axis=0))
        # boxes = ops.stack(gathered_boxes)

        # Vectorized gather using take_along_axis? No, indices are not same shape.
        # Construct indices for gather_nd
        # Implement gathering using reshape + take as gather_nd is missing in keras.ops
        B, Q, C = ops.shape(boxes)
        boxes_flat = ops.reshape(boxes, (-1, C))
        # batch_indices[:, None] * Q + topk_boxes
        batch_indices = ops.arange(B)[:, None]
        flat_indices = ops.reshape(batch_indices * Q + topk_boxes, (-1,))
        boxes = ops.take(boxes_flat, flat_indices, axis=0)
        boxes = ops.reshape(boxes, (B, self.num_select, C))

        # Scale boxes
        img_h = target_sizes[:, 0]
        img_w = target_sizes[:, 1]
        scale_fct = ops.stack([img_w, img_h, img_w, img_h], axis=1)
        scale_fct = ops.cast(scale_fct, "float32")
        boxes = boxes * scale_fct[:, None, :]

        return scores, labels, boxes
