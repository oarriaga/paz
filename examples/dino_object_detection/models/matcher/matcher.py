import numpy as np
import keras
from keras import ops
from scipy.optimize import linear_sum_assignment

# Assuming these are available in your utils as stated
from examples.dino_object_detection.models.utils.box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
    batch_sigmoid_ce_loss,
    batch_dice_loss,
)
from examples.dino_object_detection.models.segmentation_head.utils import point_sample


class HungarianMatcher(keras.Layer):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_alpha: float = 0.25,
        use_pos_only: bool = False,
        use_position_modulated_cost: bool = False,
        mask_point_sample_ratio: int = 16,
        cost_mask_ce: float = 1,
        cost_mask_dice: float = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise ValueError("all costs cant be 0")

        self.focal_alpha = focal_alpha
        self.mask_point_sample_ratio = mask_point_sample_ratio
        self.cost_mask_ce = cost_mask_ce
        self.cost_mask_dice = cost_mask_dice

    def call(self, outputs, targets, group_detr=1):
        """Performs the matching"""
        # We need to perform operations without gradient tracking for matching
        pred_logits = ops.stop_gradient(outputs["pred_logits"])
        pred_boxes = ops.stop_gradient(outputs["pred_boxes"])

        # Get shapes
        shape_logits = ops.shape(pred_logits)
        bs, num_queries = shape_logits[0], shape_logits[1]

        # Flatten predictions to [batch_size * num_queries, ...]
        flat_pred_logits = ops.reshape(pred_logits, (-1, shape_logits[-1]))
        out_prob = ops.sigmoid(flat_pred_logits)
        out_bbox = ops.reshape(pred_boxes, (-1, 4))

        # Concatenate the target labels and boxes from the list of dicts
        tgt_ids = ops.concatenate([v["labels"] for v in targets], axis=0)
        tgt_ids = ops.cast(tgt_ids, "int64")
        tgt_bbox = ops.concatenate([v["boxes"] for v in targets], axis=0)

        masks_present = "masks" in targets[0]

        cost_mask_ce = 0.0
        cost_mask_dice = 0.0

        if masks_present:
            tgt_masks = ops.concatenate([v["masks"] for v in targets], axis=0)
            out_masks = ops.stop_gradient(outputs["pred_masks"])
            out_masks = ops.reshape(
                out_masks, (-1, ops.shape(out_masks)[-2], ops.shape(out_masks)[-1])
            )

            # Point sampling logic
            mask_h, mask_w = ops.shape(out_masks)[-2], ops.shape(out_masks)[-1]
            num_points = (mask_h * mask_w) // self.mask_point_sample_ratio

            tgt_masks = ops.cast(tgt_masks, out_masks.dtype)

            # Generate random points [1, num_points, 2]
            point_coords = keras.random.uniform(
                (1, num_points, 2), dtype=out_masks.dtype
            )

            # Repeat point coords for preds and targets
            points_pred = ops.repeat(point_coords, ops.shape(out_masks)[0], axis=0)
            points_tgt = ops.repeat(point_coords, ops.shape(tgt_masks)[0], axis=0)

            # Sample points
            pred_masks_logits = point_sample(
                ops.expand_dims(out_masks, 1), points_pred, align_corners=False
            )
            pred_masks_logits = ops.squeeze(pred_masks_logits, 1)

            tgt_masks_flat = point_sample(
                ops.expand_dims(tgt_masks, 1),
                points_tgt,
                align_corners=False,
                mode="bilinear",
            )
            tgt_masks_flat = ops.squeeze(tgt_masks_flat, 1)

            # Compute costs
            cost_mask_ce = batch_sigmoid_ce_loss(pred_masks_logits, tgt_masks_flat)
            cost_mask_dice = batch_dice_loss(pred_masks_logits, tgt_masks_flat)

        # --- Compute Costs ---

        # 1. GIoU Cost
        giou = generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )
        cost_giou = -giou

        # 2. Classification Cost (Focal Loss)
        alpha = self.focal_alpha
        gamma = 2.0

        neg_log_sig = ops.log_sigmoid(-flat_pred_logits)
        pos_log_sig = ops.log_sigmoid(flat_pred_logits)

        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-neg_log_sig)
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-pos_log_sig)

        cost_class = ops.take(pos_cost_class, tgt_ids, axis=1) - ops.take(
            neg_cost_class, tgt_ids, axis=1
        )

        # 3. L1 Cost (BBox)
        diff = ops.abs(ops.expand_dims(out_bbox, 1) - ops.expand_dims(tgt_bbox, 0))
        cost_bbox = ops.sum(diff, axis=-1)

        # --- Final Cost Matrix ---
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )

        if masks_present:
            C = (
                C
                + self.cost_mask_ce * cost_mask_ce
                + self.cost_mask_dice * cost_mask_dice
            )

        C = ops.reshape(C, (bs, num_queries, -1))

        # Handle NaNs/Infs
        is_empty = ops.equal(ops.size(C), 0)
        max_cost = ops.cond(
            is_empty,
            lambda: ops.convert_to_tensor(0.0, dtype=C.dtype),
            lambda: ops.max(C),
        )

        safe_max = max_cost * 2 + 1.0
        mask_nan_inf = ops.logical_or(ops.isnan(C), ops.isinf(C))
        C = ops.where(mask_nan_inf, safe_max, C)

        # --- Hungarian Algorithm (CPU/SciPy) ---
        C_cpu = ops.convert_to_numpy(C)
        sizes = [len(v["boxes"]) for v in targets]
        indices = []

        g_num_queries = num_queries // group_detr
        C_list = np.split(C_cpu, group_detr, axis=1)

        for g_i in range(group_detr):
            C_g = C_list[g_i]
            split_indices = np.cumsum(sizes)[:-1]
            C_g_split = np.split(C_g, split_indices, axis=-1)

            indices_g = []
            for i, c in enumerate(C_g_split):
                c_mat = c[i]
                ind = linear_sum_assignment(c_mat)
                indices_g.append(ind)

            if g_i == 0:
                indices = indices_g
            else:
                new_indices = []
                for (i1, j1), (i2, j2) in zip(indices, indices_g):
                    new_i = np.concatenate([i1, i2 + g_num_queries * g_i])
                    new_j = np.concatenate([j1, j2])
                    new_indices.append((new_i, new_j))
                indices = new_indices

        return [
            (
                ops.convert_to_tensor(i, dtype="int64"),
                ops.convert_to_tensor(j, dtype="int64"),
            )
            for i, j in indices
        ]


def build_matcher(args):
    # Standard argparse access
    segmentation_head = getattr(args, "segmentation_head", False)

    if segmentation_head:
        return HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,
            cost_mask_ce=args.mask_ce_loss_coef,
            cost_mask_dice=args.mask_dice_loss_coef,
            mask_point_sample_ratio=args.mask_point_sample_ratio,
        )
    else:
        return HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,
        )
