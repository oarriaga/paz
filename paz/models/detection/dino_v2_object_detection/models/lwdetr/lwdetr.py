import math
import copy
from typing import Callable, Optional, List, Dict

import keras
from keras import layers
from keras import ops
import numpy as np

from paz.models.detection.dino_v2_object_detection.utils import box_ops
from paz.models.detection.dino_v2_object_detection.utils.misc import (
    NestedTensor,
    interpolate,
    inverse_sigmoid,
)


def sigmoid_focal_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2
):
    """Computes the sigmoid focal loss for dense object detection.

    Applies a modulating factor to the standard cross-entropy loss to
    downweight easy examples and focus training on hard negatives.

    Args:
        inputs (Tensor): Prediction logits of arbitrary shape.
        targets (Tensor): Binary classification labels with the same shape
            as inputs (0 for negative, 1 for positive).
        num_boxes (int): Number of ground-truth boxes for loss normalization.
        alpha (float): Balancing factor for positive vs negative examples.
        gamma (float): Focusing parameter that reduces the loss contribution
            from easy examples.

    Returns:
        Tensor: Scalar focal loss normalized by num_boxes.
    """
    prob = ops.sigmoid(inputs)
    # Compute binary cross-entropy from logits for numerical stability
    ce_loss = ops.binary_crossentropy(targets, inputs, from_logits=True)
    # Ensure ce_loss has the same number of dimensions as inputs
    ce_loss = (
        ops.expand_dims(ce_loss, axis=-1)
        if ops.ndim(ce_loss) < ops.ndim(inputs)
        else ce_loss
    )

    # Modulating factor: p_t is the probability of the correct class,
    # so (1 - p_t)^gamma downweights well-classified examples
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    # Apply class-balancing alpha weight
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return ops.sum(ops.mean(loss, axis=1)) / num_boxes


def sigmoid_varifocal_loss(
    inputs, targets, num_boxes, alpha: float = 0.75, gamma: float = 2
):
    """Computes the varifocal loss for quality-aware classification.

    Weighs positive samples by their target quality score and negative
    samples by a focal-style modulating factor based on prediction error.

    Args:
        inputs (Tensor): Prediction logits.
        targets (Tensor): Continuous quality targets (IoU scores for
            positives, 0 for negatives).
        num_boxes (int): Number of ground-truth boxes for normalization.
        alpha (float): Weighting factor for negative samples.
        gamma (float): Focusing exponent for negative samples.

    Returns:
        Tensor: Scalar varifocal loss normalized by num_boxes.
    """
    prob = ops.sigmoid(inputs)
    # Positive samples weighted by target quality; negative samples
    # weighted by focal modulation based on prediction-target distance
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
    """Computes position-supervised loss for localization-aware classification.

    Modulates cross-entropy by the absolute difference between predicted
    probability and target, focusing on hard-to-localize examples.

    Args:
        inputs (Tensor): Prediction logits.
        targets (Tensor): Continuous target values.
        num_boxes (int): Number of ground-truth boxes for normalization.
        alpha (float): Balancing factor for positive vs negative targets.
        gamma (float): Exponent for the position-aware modulating factor.

    Returns:
        Tensor: Scalar position-supervised loss normalized by num_boxes.
    """
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
    """Computes dice loss between predicted and target binary masks.

    Measures the overlap between predicted and target masks, analogous
    to generalized IoU for bounding boxes. A smoothing constant of 1
    is added to numerator and denominator to prevent division by zero.

    Args:
        inputs (Tensor): Mask prediction logits of shape (N, H*W).
        targets (Tensor): Binary mask targets of shape (N, H*W).
        num_masks (int): Number of masks for loss normalization.

    Returns:
        Tensor: Scalar dice loss normalized by num_masks.
    """
    inputs = ops.sigmoid(inputs)
    inputs = ops.reshape(inputs, (ops.shape(inputs)[0], -1))
    targets = ops.reshape(targets, (ops.shape(targets)[0], -1))

    numerator = 2 * ops.sum(inputs * targets, axis=-1)
    denominator = ops.sum(inputs, axis=-1) + ops.sum(targets, axis=-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return ops.sum(loss) / num_masks


def sigmoid_ce_loss(inputs, targets, num_masks):
    """Computes sigmoid binary cross-entropy loss for mask predictions.

    Args:
        inputs (Tensor): Mask prediction logits.
        targets (Tensor): Binary mask targets.
        num_masks (int): Number of masks for loss normalization.

    Returns:
        Tensor: Scalar cross-entropy loss normalized by num_masks.
    """
    loss = ops.binary_crossentropy(targets, inputs, from_logits=True)
    return ops.sum(ops.mean(loss, axis=1)) / num_masks


@keras.saving.register_keras_serializable(package="RFDETR")
class MLP(layers.Layer):
    """Multi-layer perceptron with ReLU activations between hidden layers.

    Used as the bounding-box regression head. All intermediate layers use
    ReLU activation; the final layer has no activation.

    Attributes:
        num_layers (int): Total number of dense layers.
        layers_list (list): Ordered list of Dense layers.
    """

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
    """End-to-end object detection model using Group DETR v3 architecture.

    Combines a DINOv2 backbone, a deformable transformer decoder with
    multi-scale features, and classification/bounding-box regression heads.
    Supports two-stage detection with encoder output proposals, group DETR
    query splitting, auxiliary losses, bounding-box reparameterization,
    and optional segmentation.

    Attributes:
        num_queries (int): Number of object queries per group.
        transformer: Deformable transformer decoder.
        num_classes (int): Number of object categories.
        hidden_dim (int): Feature dimensionality from the transformer.
        class_embed (Dense): Classification head.
        bbox_embed (MLP): Bounding-box regression head.
        segmentation_head: Optional segmentation head.
        refpoint_embed (Variable): Learnable reference point embeddings.
        query_feat (Variable): Learnable query feature embeddings.
        backbone: DINOv2 backbone with multi-scale projector.
        aux_loss (bool): Whether to compute auxiliary losses from
            intermediate decoder layers.
        group_detr (int): Number of query groups for group DETR.
        lite_refpoint_refine (bool): Whether to use lightweight reference
            point refinement.
        bbox_reparam (bool): Whether to use bounding-box reparameterization
            relative to reference points.
        two_stage (bool): Whether to use two-stage detection with encoder
            output proposals.
    """

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
        """Initializes the LWDETR detection model.

        Args:
            backbone: DINOv2 backbone returning multi-scale features.
            transformer: Deformable transformer decoder.
            segmentation_head: Optional segmentation head, or None.
            num_classes (int): Number of object categories.
            num_queries (int): Number of queries per group.
            aux_loss (bool): If True, compute losses at intermediate layers.
            group_detr (int): Number of query groups.
            two_stage (bool): If True, use encoder output proposals.
            lite_refpoint_refine (bool): Use lightweight refpoint refinement.
            bbox_reparam (bool): Use bbox reparameterization.
            **kwargs: Additional Keras model arguments.
        """
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = transformer.d_model
        hidden_dim = self.hidden_dim
        self._aux_loss = aux_loss
        self._segmentation_head_config = None

        # Detection heads: classification (linear) and bbox regression (MLP)
        self.class_embed = layers.Dense(num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.segmentation_head = segmentation_head

        # Learnable query embeddings: reference points (4D) and query
        # features (hidden_dim). Total queries = num_queries * group_detr.
        query_dim = 4
        self.refpoint_embed = self.add_weight(
            name="refpoint_embed",
            shape=(num_queries * group_detr, query_dim),
            initializer="zeros",
        )
        self.query_feat = self.add_weight(
            name="query_feat",
            shape=(num_queries * group_detr, self.hidden_dim),
            initializer="glorot_uniform",
        )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.group_detr = group_detr

        self.lite_refpoint_refine = lite_refpoint_refine
        # Assign bbox_embed to decoder for iterative refinement (or None
        # when using lightweight refinement where decoder handles it)
        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        self.bbox_reparam = bbox_reparam

        self.two_stage = two_stage

        # Two-stage encoder output heads: separate class/bbox heads per
        # group for generating encoder proposals
        if self.two_stage:
            self.enc_out_bbox_embed = []
            self.enc_out_class_embed = []
            for _ in range(group_detr):
                self.enc_out_bbox_embed.append(MLP(hidden_dim, hidden_dim, 4, 3))
                self.enc_out_class_embed.append(layers.Dense(num_classes))

            # Make encoder output heads accessible to the transformer
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

    def update_drop_path(self, drop_path_rate, vit_encoder_num_layers):
        """Set stochastic depth rate with per-layer linear scaling.

        Linearly scales the rate from 0 (first layer) to
        ``drop_path_rate`` (last layer).

        Args:
            drop_path_rate (float): Target drop path rate for the last layer.
            vit_encoder_num_layers (int): Number of ViT encoder layers.
        """
        from paz.models.foundation.dinov2.layers.drop_path import DropPath

        encoder = self.backbone.backbone.encoder.encoder.encoder
        num_layers = vit_encoder_num_layers or len(encoder.encoder_layers)
        for i in range(num_layers):
            block_rate = (
                drop_path_rate * i / max(1, num_layers - 1)
                if num_layers > 1
                else 0.0
            )
            layer = encoder.encoder_layers[i]
            if isinstance(layer.drop_path1, DropPath):
                layer.drop_path1.drop_probability = block_rate
            if isinstance(layer.drop_path2, DropPath):
                layer.drop_path2.drop_probability = block_rate

    def update_dropout(self, dropout_rate):
        """Update all Dropout layers in the transformer to *dropout_rate*.

        Walks the transformer encoder/decoder and sets the ``rate``
        attribute on every ``keras.layers.Dropout`` instance found.

        Args:
            dropout_rate (float): New dropout probability in [0, 1].
        """
        for layer in self._flatten_layers():
            if isinstance(layer, keras.layers.Dropout):
                layer.rate = dropout_rate

    def call(self, samples, training=False):
        """Forward pass of the LWDETR detection model.

        Args:
            samples: Input images as a tensor (B, H, W, 3), a tuple of
                (tensors, mask), or a NestedTensor.
            training (bool): Whether in training mode. Controls whether
                all group_detr queries or only one group is used.

        Returns:
            dict: Detection outputs containing:
                - "pred_logits": Classification logits (B, Q, num_classes).
                - "pred_boxes": Predicted boxes (B, Q, 4).
                - "pred_masks" (optional): Mask predictions.
                - "aux_outputs" (optional): Intermediate layer outputs.
                - "enc_outputs" (optional): Encoder proposal outputs.
        """
        # Unpack input format
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

        # Backbone forward pass: extracts multi-scale features and their
        # corresponding position encodings
        features, poss_all = self.backbone(tensors, mask=mask, training=training)

        srcs = []
        masks = []
        # Decompose features into source tensors and attention masks
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

        # During training, use all group_detr queries; during inference,
        # use only the first group for efficiency
        if training:
            refpoint_embed_weight = self.refpoint_embed
            query_feat_weight = self.query_feat
        else:
            refpoint_embed_weight = self.refpoint_embed[: self.num_queries]
            query_feat_weight = self.query_feat[: self.num_queries]

        # Transformer forward: deformable decoder with multi-scale features
        # Returns decoder hidden states, reference points, and encoder outputs
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

        # Pre-compute NCHW spatial features for the segmentation head
        spatial_nchw = (
            ops.transpose(srcs[0], (0, 3, 1, 2))
            if self.segmentation_head is not None
            else None
        )

        # Compute detection outputs from decoder hidden states
        if hs is not None:
            if self.bbox_reparam:
                # Bounding-box reparameterization: predict deltas relative
                # to reference points rather than absolute coordinates.
                # CX, CY are offset by delta scaled by reference WH;
                # W, H are exponentiated delta scaled by reference WH.
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
                # Standard coordinate prediction: add bbox offset to
                # reference point and apply sigmoid for normalization
                outputs_coord = self.bbox_embed(hs) + ref_unsigmoid
                outputs_coord = ops.sigmoid(outputs_coord)

            outputs_class = self.class_embed(hs)

            if self.segmentation_head is not None:
                # Compute mask predictions from spatial features and
                # decoder query features
                target_shape = (
                    ops.shape(tensors)[1:3]
                    if ops.ndim(tensors) == 4
                    else ops.shape(tensors)[0:2]
                )

                # Segmentation head expects (spatial_features_NCHW,
                # query_features, image_size)
                outputs_masks = self.segmentation_head(
                    spatial_nchw, hs, image_size=target_shape
                )

            out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
            if self.segmentation_head is not None:
                out["pred_masks"] = outputs_masks[-1]

            # Collect intermediate decoder layer outputs for auxiliary loss
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(
                    outputs_class, outputs_coord, outputs_masks
                )

        # Two-stage: compute encoder output proposals by applying
        # per-group class/bbox heads to the encoder hidden states
        if self.two_stage:
            group_detr = self.group_detr if training else 1
            # Split encoder hidden states into query groups
            hs_enc_list = ops.split(
                hs_enc, group_detr, axis=1
            )

            # Apply per-group classification heads
            cls_enc = []
            for g_idx in range(group_detr):
                # ops.split returns a list of sub-tensors
                cls_enc_gidx = self.transformer.enc_out_class_embed[g_idx](
                    hs_enc_list[g_idx]
                )
                cls_enc.append(cls_enc_gidx)

            # Recombine group classifications into a single tensor
            cls_enc = ops.concatenate(cls_enc, axis=1)

            masks_enc = None
            if self.segmentation_head is not None:
                # Compute segmentation masks for encoder proposals
                target_shape = (
                    ops.shape(tensors)[1:3]
                    if ops.ndim(tensors) == 4
                    else ops.shape(tensors)[0:2]
                )
                # Pass encoder hidden states through segmentation head
                # with skip_blocks=True to bypass decoder block processing
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

    def call_export(self, samples):
        """Forward pass for export/inference (no auxiliary outputs).

        Runs the model without auxiliary or encoder outputs, producing
        only the final-layer ``pred_logits`` and ``pred_boxes``.  This
        is the lightweight path used when exporting the model or running
        inference without training overhead.

        Args:
            samples: Input images ``(B, H, W, 3)`` or ``(tensor, mask)``.

        Returns:
            dict: ``{"pred_logits": ..., "pred_boxes": ...}`` and
                optionally ``"pred_masks"`` if segmentation is enabled.
        """
        out = self.call(samples, training=False)
        # Strip aux_outputs and enc_outputs for clean export
        return {
            k: v for k, v in out.items()
            if k not in ("aux_outputs", "enc_outputs")
        }

    def export(self, export_path, input_shape=None):
        """Export the model to a SavedModel or ONNX-compatible format.

        Traces ``call_export`` with a concrete input spec and saves.

        Args:
            export_path (str): Target directory for the exported model.
            input_shape (tuple or None): ``(H, W, C)`` shape. Uses the
                model's configured resolution if ``None``.
        """
        import keras

        if input_shape is None:
            # Try to infer from the model's configuration
            res = getattr(self, "_resolution", None)
            if res is None:
                raise ValueError(
                    "input_shape must be provided or the model must have "
                    "a _resolution attribute."
                )
            input_shape = (res, res, 3)

        # Build a concrete input spec for tracing
        input_spec = keras.Input(shape=input_shape, dtype="float32")
        export_model = keras.Model(
            inputs=input_spec,
            outputs=self.call_export(input_spec),
        )
        export_model.export(export_path)

    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_masks):
        """Collects intermediate decoder layer outputs for auxiliary loss.

        Args:
            outputs_class (Tensor): Class predictions from all layers
                with shape (num_layers, B, Q, C).
            outputs_coord (Tensor): Box predictions from all layers
                with shape (num_layers, B, Q, 4).
            outputs_masks: Optional mask predictions from all layers.

        Returns:
            list[dict]: Per-layer output dicts excluding the last layer.
        """
        aux_outputs = []
        num_layers = ops.shape(outputs_class)[0]
        for i in range(num_layers - 1):
            d = {"pred_logits": outputs_class[i], "pred_boxes": outputs_coord[i]}
            if outputs_masks is not None:
                d["pred_masks"] = outputs_masks[i]
            aux_outputs.append(d)
        return aux_outputs


class SetCriterion(layers.Layer):
    """Computes the training loss for LWDETR via Hungarian matching.

    Performs bipartite matching between predictions and ground-truth
    targets, then computes classification (focal loss), bounding-box
    (L1 + GIoU), and optional mask losses.

    Attributes:
        num_classes (int): Number of object categories.
        matcher: Hungarian matcher for bipartite assignment.
        weight_dict (dict): Loss term weights keyed by loss name.
        loss_types (list[str]): Active loss types (e.g. 'labels', 'boxes').
        focal_alpha (float): Alpha parameter for focal loss.
        group_detr (int): Number of query groups.
        sum_group_losses (bool): If True, sum rather than average across
            groups when normalizing by num_boxes.
        use_varifocal_loss (bool): Use varifocal loss instead of focal.
        use_position_supervised_loss (bool): Use position-supervised loss.
        ia_bce_loss (bool): Use instance-aware BCE loss.
        mask_point_sample_ratio (int): Downsampling ratio for mask loss.
    """

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
        """Computes classification loss.

        When ``ia_bce_loss`` is enabled, uses
        an IoU-aware instance-adaptive BCE loss that blends predicted
        confidence with localization IoU as soft targets.  Otherwise
        falls back to standard sigmoid focal loss.

        Args:
            outputs (dict): Contains "pred_logits" of shape (B, Q, C)
                and "pred_boxes" of shape (B, Q, 4).
            targets (list[dict]): Per-image targets with "labels" and
                "boxes".
            indices (list[tuple]): Matched (src, tgt) index pairs.
            num_boxes (int): Total matched boxes for normalization.
            log (bool): Reserved for logging metrics.

        Returns:
            dict: {"loss_ce": scalar classification loss}.
        """
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)

        # Gather target class labels at the matched positions
        target_classes_o = ops.concatenate(
            [ops.take(t["labels"], J, axis=0)
             for t, (_, J) in zip(targets, indices)]
        )

        if self.ia_bce_loss:
            alpha = self.focal_alpha
            gamma = 2

            # Gather matched predicted boxes (flat indexing)
            B, Q, C = ops.shape(src_logits)
            box_dim = ops.shape(outputs["pred_boxes"])[2]
            pred_boxes_flat = ops.reshape(
                outputs["pred_boxes"], (-1, box_dim)
            )
            flat_idx_2d = idx[0] * Q + idx[1]
            src_boxes = ops.take(pred_boxes_flat, flat_idx_2d, axis=0)

            target_boxes = ops.concatenate(
                [ops.take(t["boxes"], i, axis=0)
                 for t, (_, i) in zip(targets, indices)],
                axis=0,
            )

            # IoU between each matched pair (detached from gradients)
            iou_matrix, _ = box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(ops.stop_gradient(src_boxes)),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
            pos_ious = ops.stop_gradient(ops.diag(iou_matrix))

            prob = ops.sigmoid(src_logits)

            # Positive and negative weights — full (B, Q, C) tensors
            pos_weights = ops.zeros_like(src_logits)
            neg_weights = prob ** gamma

            # Gather probabilities at matched (batch, query, class) positions
            num_classes_logits = C
            flat_idx_3d = (
                idx[0] * Q * num_classes_logits
                + idx[1] * num_classes_logits
                + ops.cast(target_classes_o, idx[0].dtype)
            )
            prob_flat = ops.reshape(prob, (-1,))
            matched_probs = ops.take(prob_flat, flat_idx_3d)

            # Soft target: prob^alpha * iou^(1-alpha), clamped at 0.01
            t = ops.stop_gradient(
                ops.maximum(
                    matched_probs ** alpha * pos_ious ** (1 - alpha),
                    0.01,
                )
            )

            # Scatter t into pos_weights and (1-t) into neg_weights at
            # matched positions using flat 3D indexing
            pos_weights_flat = ops.reshape(pos_weights, (-1,))
            neg_weights_flat = ops.reshape(neg_weights, (-1,))

            # Build scatter indices (N_matched, 1)
            scatter_idx = ops.expand_dims(flat_idx_3d, axis=-1)
            pos_weights_flat = ops.scatter_update(
                pos_weights_flat, scatter_idx, ops.cast(t, pos_weights.dtype)
            )
            neg_weights_flat = ops.scatter_update(
                neg_weights_flat, scatter_idx,
                1.0 - ops.cast(t, neg_weights.dtype),
            )
            pos_weights = ops.reshape(pos_weights_flat, ops.shape(src_logits))
            neg_weights = ops.reshape(neg_weights_flat, ops.shape(src_logits))

            # Numerically stable reformulation:
            # loss = neg_w * logits - log_sigmoid(logits) * (pos_w + neg_w)
            log_sigmoid = -ops.softplus(-src_logits)  # == log(sigmoid(x))
            loss_ce = (
                neg_weights * src_logits
                - log_sigmoid * (pos_weights + neg_weights)
            )
            loss_ce = ops.sum(loss_ce) / num_boxes

        else:
            # Standard sigmoid focal loss (fallback)
            target_classes = ops.full(
                src_logits.shape[:2], self.num_classes, dtype="int64"
            )
            indices_nd = ops.stack(idx, axis=-1)
            target_classes = ops.scatter_update(
                target_classes, indices_nd, target_classes_o
            )

            num_classes_logits = ops.shape(src_logits)[2]
            target_classes_onehot = ops.one_hot(
                target_classes, num_classes_logits + 1
            )
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
        """Computes bounding-box L1 and generalized IoU losses.

        Gathers matched predicted and target boxes, then computes L1
        regression loss and GIoU loss for the matched pairs.

        Args:
            outputs (dict): Contains "pred_boxes" of shape (B, Q, 4).
            targets (list[dict]): Per-image targets with "boxes".
            indices (list[tuple]): Matched (src, tgt) index pairs.
            num_boxes (int): Total matched boxes for normalization.

        Returns:
            dict: {"loss_bbox": L1 loss, "loss_giou": GIoU loss}.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = ops.take(
            outputs["pred_boxes"], ops.stack(idx, axis=-1)
        )

        # Gather matched predictions using flattened indexing since
        # ops.gather_nd is not available in keras.ops
        B, Q, C = ops.shape(outputs["pred_boxes"])
        outputs_boxes_flat = ops.reshape(outputs["pred_boxes"], (-1, C))
        flat_indices = idx[0] * Q + idx[1]
        src_boxes = ops.take(outputs_boxes_flat, flat_indices, axis=0)

        # Concatenate matched target boxes across the batch
        target_boxes = ops.concatenate(
            [ops.take(t["boxes"], i, axis=0) for t, (_, i) in zip(targets, indices)],
            axis=0,
        )

        loss_bbox = ops.abs(src_boxes - target_boxes)
        loss_bbox = ops.sum(loss_bbox) / num_boxes

        # Generalized IoU loss: 1 - GIoU for each matched pair
        loss_giou = 1 - ops.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        loss_giou = ops.sum(loss_giou) / num_boxes

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def loss_cardinality(self, outputs, targets, indices, num_boxes, **kwargs):
        """Compute cardinality error (for logging only, not backpropagated).

        Counts how many predictions are classified as non-background
        and compares against the actual number of objects per image.

        Args:
            outputs (dict): Contains "pred_logits" (B, Q, C).
            targets (list[dict]): Per-image targets with "labels".
            indices: Unused (kept for dispatcher signature consistency).
            num_boxes: Unused.

        Returns:
            dict: {"cardinality_error": scalar} — wrapped in
                stop_gradient so it does not affect training.
        """
        pred_logits = outputs["pred_logits"]
        tgt_lengths = ops.convert_to_tensor(
            [len(t["labels"]) for t in targets], dtype="float32",
        )
        # Count predictions whose argmax is not the last (background) class
        card_pred = ops.cast(
            ops.sum(
                ops.cast(
                    ops.argmax(pred_logits, axis=-1)
                    != ops.shape(pred_logits)[-1] - 1,
                    "int32",
                ),
                axis=1,
            ),
            "float32",
        )
        card_err = ops.mean(ops.abs(card_pred - tgt_lengths))
        return {"cardinality_error": ops.stop_gradient(card_err)}

    def loss_masks(self, outputs, targets, indices, num_boxes, **kwargs):
        """Compute BCE and Dice losses for segmentation masks.

        Gathers matched predicted and target masks, then computes
        binary cross-entropy and dice losses over spatially sampled
        points for efficiency.

        Args:
            outputs (dict): Contains "pred_masks" (B, Q, H, W).
            targets (list[dict]): Per-image targets with "masks".
            indices (list[tuple]): Matched (src, tgt) index pairs.
            num_boxes (int): Normalization factor.

        Returns:
            dict: {"loss_mask_ce": ..., "loss_mask_dice": ...}.
        """
        if "pred_masks" not in outputs:
            zero = ops.convert_to_tensor(0.0, dtype="float32")
            return {"loss_mask_ce": zero, "loss_mask_dice": zero}

        idx = self._get_src_permutation_idx(indices)
        pred_masks = outputs["pred_masks"]  # (B, Q, H, W)

        # Gather matched prediction masks using flat indexing
        B = ops.shape(pred_masks)[0]
        Q = ops.shape(pred_masks)[1]
        H = ops.shape(pred_masks)[2]
        W = ops.shape(pred_masks)[3]
        pred_flat = ops.reshape(pred_masks, (B * Q, H, W))
        flat_idx = idx[0] * Q + idx[1]
        src_masks = ops.take(pred_flat, flat_idx, axis=0)  # (N, H, W)

        # Handle empty matches
        if ops.shape(src_masks)[0] == 0:
            zero = ops.convert_to_tensor(0.0, dtype="float32")
            return {"loss_mask_ce": zero, "loss_mask_dice": zero}

        # Gather matched target masks
        target_masks_list = []
        for t, (_, j) in zip(targets, indices):
            if "masks" in t:
                target_masks_list.append(ops.take(t["masks"], j, axis=0))
        if not target_masks_list:
            zero = ops.convert_to_tensor(0.0, dtype="float32")
            return {"loss_mask_ce": zero, "loss_mask_dice": zero}
        target_masks = ops.concatenate(target_masks_list, axis=0)  # (N, Ht, Wt)

        # Interpolate target masks to prediction resolution if needed
        tgt_h = ops.shape(target_masks)[1]
        tgt_w = ops.shape(target_masks)[2]
        if tgt_h != H or tgt_w != W:
            target_masks = ops.cast(target_masks, "float32")
            target_masks = ops.expand_dims(target_masks, axis=-1)
            target_masks = ops.image.resize(
                target_masks, (int(H), int(W)), interpolation="nearest",
            )
            target_masks = target_masks[..., 0]

        # Downsample by mask_point_sample_ratio for efficiency
        ratio = self.mask_point_sample_ratio
        if ratio > 1:
            ds_h = max(1, int(H) // ratio)
            ds_w = max(1, int(W) // ratio)
            src_ds = ops.expand_dims(src_masks, axis=-1)
            src_ds = ops.image.resize(
                src_ds, (ds_h, ds_w), interpolation="bilinear",
            )[..., 0]
            tgt_ds = ops.expand_dims(
                ops.cast(target_masks, "float32"), axis=-1,
            )
            tgt_ds = ops.image.resize(
                tgt_ds, (ds_h, ds_w), interpolation="nearest",
            )[..., 0]
        else:
            src_ds = src_masks
            tgt_ds = ops.cast(target_masks, "float32")

        # Flatten spatial dims and compute losses
        src_flat = ops.reshape(src_ds, (ops.shape(src_ds)[0], -1))
        tgt_flat = ops.reshape(tgt_ds, (ops.shape(tgt_ds)[0], -1))

        loss_ce = sigmoid_ce_loss(src_flat, tgt_flat, num_boxes)
        loss_dice = dice_loss(src_flat, tgt_flat, num_boxes)

        return {"loss_mask_ce": loss_ce, "loss_mask_dice": loss_dice}

    def _get_src_permutation_idx(self, indices):
        """Builds batch and source index arrays from matching results.

        Args:
            indices (list[tuple]): Per-image (src_indices, tgt_indices).

        Returns:
            tuple: (batch_indices, src_indices) tensors for gathering.
        """
        batch_idx = ops.concatenate(
            [ops.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = ops.concatenate([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """Dispatches to the appropriate loss computation method.

        Args:
            loss (str): Loss type name (``'labels'``, ``'boxes'``,
                ``'cardinality'``, or ``'masks'``).
            outputs (dict): Model predictions.
            targets (list[dict]): Ground-truth targets.
            indices (list[tuple]): Matched index pairs.
            num_boxes (int): Normalization factor.

        Returns:
            dict: Computed loss values, or empty dict if loss type unknown.
        """
        loss_map = {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
            "cardinality": self.loss_cardinality,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            return {}
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def call(self, outputs, targets, training=True):
        """Computes all training losses for the model outputs.

        Runs Hungarian matching, computes per-type losses on the main
        output, and optionally on auxiliary and encoder outputs.

        Args:
            outputs (dict): Model predictions including "pred_logits",
                "pred_boxes", and optionally "aux_outputs" / "enc_outputs".
            targets (list[dict]): Per-image ground-truth targets.
            training (bool): When False, ``group_detr`` is forced to 1
                (evaluation mode uses a single query group).

        Returns:
            dict: All computed losses keyed by name (e.g. "loss_ce",
                "loss_bbox", "loss_giou", with "_i" suffixes for
                auxiliary layers and "_enc" for encoder outputs).
        """
        group_detr = self.group_detr if training else 1
        outputs_without_aux = {
            k: v for k, v in outputs.items()
            if k not in ("aux_outputs", "enc_outputs")
        }

        # Bipartite matching between predictions and targets
        indices = self.matcher(outputs_without_aux, targets, group_detr=group_detr)

        # Normalize by total number of matched boxes
        num_boxes = sum(len(t["labels"]) for t in targets)
        if not self.sum_group_losses:
            num_boxes = num_boxes * group_detr
        num_boxes = ops.cast(ops.maximum(num_boxes, 1), "float32")

        # Compute losses for each loss type on the main output
        losses = {}
        for loss in self.loss_types:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # Compute losses on auxiliary intermediate decoder outputs
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, group_detr=group_detr)
                for loss in self.loss_types:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Compute losses on two-stage encoder output proposals
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            enc_indices = self.matcher(
                enc_outputs, targets, group_detr=group_detr,
            )
            for loss in self.loss_types:
                kwargs = {}
                if loss == "labels":
                    kwargs["log"] = False
                l_dict = self.get_loss(
                    loss, enc_outputs, targets, enc_indices, num_boxes,
                    **kwargs,
                )
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


@keras.saving.register_keras_serializable(package="RFDETR")
class PostProcess(layers.Layer):
    """Converts raw model outputs to scaled detection results.

    Selects the top-k most confident predictions across all classes,
    converts boxes from normalized (cx, cy, w, h) to absolute (x1, y1,
    x2, y2) coordinates, and scales them to the target image size.

    Attributes:
        num_select (int): Number of top detections to return per image.
    """

    def __init__(self, num_select=300, **kwargs):
        super().__init__(**kwargs)
        self.num_select = num_select

    def get_config(self):
        config = super().get_config()
        config.update({"num_select": self.num_select})
        return config

    def call(self, outputs, target_sizes):
        """Post-processes model outputs into scaled detections.

        Args:
            outputs (dict): Contains "pred_logits" (B, Q, C) and
                "pred_boxes" (B, Q, 4) in normalized coordinates.
                Optionally "pred_masks" (B, Q, Hm, Wm).
            target_sizes (Tensor): Original image sizes (B, 2) as
                (height, width) for coordinate scaling.

        Returns:
            tuple: (scores, labels, boxes) or (scores, labels, boxes, masks)
                - scores: (B, K) confidence scores.
                - labels: (B, K) class label indices.
                - boxes: (B, K, 4) in absolute (x1, y1, x2, y2) coords.
                - masks: list of (K, 1, H, W) boolean arrays per image
                    (only when pred_masks is present).
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        out_masks = outputs.get("pred_masks", None)

        # Flatten class probabilities and select top-k across all classes
        prob = ops.sigmoid(out_logits)
        prob_flat = ops.reshape(prob, (ops.shape(out_logits)[0], -1))

        topk_values, topk_indexes = ops.top_k(prob_flat, self.num_select)

        scores = topk_values
        # Recover which query and class each top-k entry belongs to
        num_classes_logits = ops.shape(out_logits)[2]
        topk_boxes = topk_indexes // num_classes_logits
        labels = topk_indexes % num_classes_logits

        # Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # Gather the boxes corresponding to top-k queries using
        # flattened indexing (batch_idx * Q + query_idx)
        B, Q, C = ops.shape(boxes)
        boxes_flat = ops.reshape(boxes, (-1, C))
        batch_indices = ops.arange(B)[:, None]
        flat_indices = ops.reshape(batch_indices * Q + topk_boxes, (-1,))
        boxes = ops.take(boxes_flat, flat_indices, axis=0)
        boxes = ops.reshape(boxes, (B, self.num_select, C))

        # Scale boxes to absolute pixel coordinates using target image sizes
        img_h = target_sizes[:, 0]
        img_w = target_sizes[:, 1]
        scale_fct = ops.stack([img_w, img_h, img_w, img_h], axis=1)
        scale_fct = ops.cast(scale_fct, "float32")
        boxes = boxes * scale_fct[:, None, :]

        if out_masks is not None:
            # Gather masks for top-K queries, resize, and threshold
            masks_list = []
            for i in range(int(ops.convert_to_numpy(ops.shape(out_masks)[0]))):
                k_idx = topk_boxes[i]  # (K,)
                masks_i = ops.take(out_masks[i], k_idx, axis=0)  # (K, Hm, Wm)
                h = int(ops.convert_to_numpy(target_sizes[i, 0]))
                w = int(ops.convert_to_numpy(target_sizes[i, 1]))
                # Add channel dim for resize: (K, Hm, Wm) -> (K, Hm, Wm, 1)
                masks_i = ops.expand_dims(masks_i, axis=-1)
                masks_i = ops.image.resize(
                    masks_i, (h, w), interpolation="bilinear"
                )
                # (K, H, W, 1) -> (K, 1, H, W) to match expected format
                masks_i = ops.transpose(masks_i, (0, 3, 1, 2))
                masks_i = masks_i > 0.0
                masks_list.append(masks_i)
            return scores, labels, boxes, masks_list

        return scores, labels, boxes
