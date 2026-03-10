import numpy as np
import keras
from keras import ops
from scipy.optimize import linear_sum_assignment

from paz.models.detection.dino_v2_object_detection.utils.box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
    batch_sigmoid_ce_loss,
    batch_dice_loss
)
from paz.models.detection.dino_v2_object_detection.models.segmentation_head.segmentation_head_keras import point_sample


class HungarianMatcher(keras.layers.Layer):
    """Bipartite matcher using the Hungarian algorithm for object detection.

    Computes an optimal one-to-one assignment between predicted queries and
    ground-truth targets by minimizing a weighted cost matrix composed of
    classification, bounding box L1, generalized IoU, and optional mask costs.

    Since predictions typically outnumber targets (no "no-object" class in
    targets), unmatched predictions are treated as background.

    Attributes:
        cost_class (float): Weight for the focal loss classification cost.
        cost_bbox (float): Weight for the L1 bounding box cost.
        cost_giou (float): Weight for the generalized IoU cost.
        focal_alpha (float): Alpha parameter for focal loss cost computation.
        mask_point_sample_ratio (int): Downsampling ratio for mask point
            sampling.
        cost_mask_ce (float): Weight for mask cross-entropy cost.
        cost_mask_dice (float): Weight for mask dice cost.
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha: float = 0.25,
                 use_pos_only: bool = False, use_position_modulated_cost: bool = False, 
                 mask_point_sample_ratio: int = 16, cost_mask_ce: float = 1, cost_mask_dice: float = 1, **kwargs):
        """Initializes the Hungarian Matcher with cost weights.

        Args:
            cost_class (float): Relative weight of classification error in
                the matching cost.
            cost_bbox (float): Relative weight of L1 bounding box error in
                the matching cost.
            cost_giou (float): Relative weight of generalized IoU loss in
                the matching cost.
            focal_alpha (float): Balancing factor for focal loss computation.
            use_pos_only (bool): Reserved for position-only matching mode.
            use_position_modulated_cost (bool): Reserved for
                position-modulated cost mode.
            mask_point_sample_ratio (int): Ratio for downsampling mask
                spatial resolution during point-based cost computation.
            cost_mask_ce (float): Weight for mask binary cross-entropy cost.
            cost_mask_dice (float): Weight for mask dice cost.
            **kwargs: Additional Keras layer arguments.

        Raises:
            AssertionError: If all three primary cost weights are zero.
        """
        super().__init__(**kwargs)
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        self.focal_alpha = focal_alpha
        self.mask_point_sample_ratio = mask_point_sample_ratio
        self.cost_mask_ce = cost_mask_ce
        self.cost_mask_dice = cost_mask_dice

    def compute_cost_matrix(self, outputs, targets):
        """Computes the full cost matrix between all predictions and targets.

        Constructs a combined cost matrix from classification focal cost,
        bounding box L1 cost, generalized IoU cost, and optional mask costs.
        The matrix has shape (batch_size * num_queries, total_targets) where
        total_targets is the sum of targets across all batch elements.

        Args:
            outputs (dict): Model predictions containing:
                "pred_logits": Tensor [B, Q, C] with classification logits.
                "pred_boxes": Tensor [B, Q, 4] with predicted boxes in
                    (cx, cy, w, h) format.
                "pred_masks" (optional): Either a tensor [B, Q, H, W] or a
                    dict with "spatial_features", "query_features", "bias"
                    for deferred mask computation.
            targets (list[dict]): Per-image target dicts with:
                "labels": Tensor [T_i] of class indices.
                "boxes": Tensor [T_i, 4] of ground-truth boxes.
                "masks" (optional): Tensor [T_i, H, W] of binary masks.

        Returns:
            Tensor: Cost matrix of shape (B * Q, sum(T_i)).
        """
        # Flatten predictions across the batch dimension for pairwise cost
        # computation: (B, Q, C) -> (B*Q, C) and (B, Q, 4) -> (B*Q, 4)
        flat_pred_logits = ops.reshape(outputs["pred_logits"], (-1, ops.shape(outputs["pred_logits"])[-1]))
        out_bbox = ops.reshape(outputs["pred_boxes"], (-1, 4))
        
        # Concatenate all per-image targets into single tensors for
        # batch-wide pairwise cost computation
        tgt_ids = ops.concatenate([v["labels"] for v in targets], axis=0)
        tgt_bbox = ops.concatenate([v["boxes"] for v in targets], axis=0)
        
        # Convert logits to probabilities for focal loss cost computation
        out_prob = ops.sigmoid(flat_pred_logits)
        
        masks_present = "masks" in targets[0]

        # Generalized IoU cost: pairwise GIoU between all predicted and
        # target boxes, negated because GIoU is a similarity measure but
        # we need a cost to minimize
        giou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -giou

        # Classification cost using focal loss formulation. Alpha balances
        # positive/negative classes; gamma=2.0 downweights easy examples.
        alpha = self.focal_alpha
        gamma = 2.0
        
        def log_sigmoid(x):
            """Numerically stable log-sigmoid: log(sigmoid(x)) = -softplus(-x)."""
            return -ops.softplus(-x) 
        
        # Compute per-class positive and negative focal costs
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-log_sigmoid(-flat_pred_logits))
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-log_sigmoid(flat_pred_logits))
        
        # Select costs only at the target class columns and compute the
        # per-pair focal cost as (positive cost - negative cost)
        tgt_ids = ops.cast(tgt_ids, "int32")
        pos_cost_subset = ops.take(pos_cost_class, tgt_ids, axis=1)
        neg_cost_subset = ops.take(neg_cost_class, tgt_ids, axis=1)
        cost_class = pos_cost_subset - neg_cost_subset

        # L1 bounding box cost: pairwise absolute difference between
        # predicted and target box coordinates, summed over 4 dimensions
        out_bbox_exp = ops.expand_dims(out_bbox, 1)  # (B*Q, 1, 4)
        tgt_bbox_exp = ops.expand_dims(tgt_bbox, 0)  # (1, T, 4)
        cost_bbox = ops.sum(ops.abs(out_bbox_exp - tgt_bbox_exp), axis=-1)

        cost_mask_ce = 0.0
        cost_mask_dice = 0.0

        # Mask cost computation: when masks are present, compute binary
        # cross-entropy and dice costs between predicted and target masks
        # at randomly sampled spatial points for efficiency
        if masks_present:
            tgt_masks = ops.concatenate([v["masks"] for v in targets], axis=0)

            if "pred_masks" in outputs and (ops.is_tensor(outputs["pred_masks"]) or isinstance(outputs["pred_masks"], (keras.KerasTensor, np.ndarray))):
                 # Dense mask tensor path: predicted mask logits are
                 # available as a dense tensor of shape (B, Q, H, W)
                 out_masks = outputs["pred_masks"]
                 out_masks = ops.reshape(out_masks, (-1, ops.shape(out_masks)[-2], ops.shape(out_masks)[-1]))
                 
                 # Determine number of sample points as a fraction of
                 # the total spatial resolution
                 H_mask = ops.shape(out_masks)[-2]
                 W_mask = ops.shape(out_masks)[-1]
                 num_points = (H_mask * W_mask) // self.mask_point_sample_ratio
                 
                 # Sample random spatial coordinates in [0, 1] range
                 point_coords = keras.random.uniform((1, num_points, 2), minval=0.0, maxval=1.0)
                 
                 # Sample predicted mask values at the random points
                 out_masks_exp = ops.expand_dims(out_masks, 1)
                 batch_len = ops.shape(out_masks)[0]
                 point_coords_rep = ops.broadcast_to(point_coords, (batch_len, num_points, 2))
                 
                 sample_out = point_sample(out_masks_exp, point_coords_rep, align_corners=False)
                 pred_masks_logits = ops.squeeze(sample_out, 1)
                 
            else:
                 # Deferred mask computation path: masks are defined by
                 # spatial features, query features, and bias. Mask logits
                 # are produced via einsum contraction after point sampling.
                 spatial_features = outputs["pred_masks"]["spatial_features"]
                 query_features = outputs["pred_masks"]["query_features"]
                 bias = outputs["pred_masks"]["bias"]
                 
                 H_sf = ops.shape(spatial_features)[-2]
                 W_sf = ops.shape(spatial_features)[-1]
                 num_points = (H_sf * W_sf) // self.mask_point_sample_ratio
                 
                 point_coords = keras.random.uniform((1, num_points, 2), minval=0.0, maxval=1.0)
                 
                 batch_len = ops.shape(spatial_features)[0]
                 point_coords_rep = ops.broadcast_to(point_coords, (batch_len, num_points, 2))
                 
                 # Sample spatial features at random points, then contract
                 # with query features to produce per-query mask logits
                 pred_masks_logits = point_sample(spatial_features, point_coords_rep, align_corners=False)
                 pred_masks_logits = ops.einsum('bcp,bnc->bnp', pred_masks_logits, query_features) + bias
                 pred_masks_logits = ops.reshape(pred_masks_logits, (-1, ops.shape(pred_masks_logits)[-1]))
            
            # Sample target masks at the same random points for a fair
            # comparison with predicted mask logits
            tgt_masks = ops.cast(tgt_masks, pred_masks_logits.dtype)
            tgt_masks_exp = ops.expand_dims(tgt_masks, 1)
            tgt_len = ops.shape(tgt_masks)[0]
            point_coords_tgt = ops.broadcast_to(point_coords, (tgt_len, num_points, 2))
            
            tgt_masks_flat = point_sample(tgt_masks_exp, point_coords_tgt, align_corners=False)
            tgt_masks_flat = ops.squeeze(tgt_masks_flat, 1)

            # Compute pairwise mask costs: binary CE and dice loss
            # between all predicted and target mask point samples
            cost_mask_ce = batch_sigmoid_ce_loss(pred_masks_logits, tgt_masks_flat)
            cost_mask_dice = batch_dice_loss(pred_masks_logits, tgt_masks_flat)

        # Combine all cost terms with their respective weights
        C = (self.cost_bbox * cost_bbox) + (self.cost_class * cost_class) + (self.cost_giou * cost_giou)
        
        if masks_present:
            C = C + (self.cost_mask_ce * cost_mask_ce) + (self.cost_mask_dice * cost_mask_dice)
            
        return C

    def call(self, outputs, targets, group_detr=1):
        """Performs bipartite matching between predictions and targets.

        Computes the cost matrix and solves the linear sum assignment
        problem per batch element. Supports group DETR where queries are
        split into groups, each independently matched to the full target set.

        Args:
            outputs (dict): Model predictions containing:
                "pred_logits": Tensor [B, Q, C] with classification logits.
                "pred_boxes": Tensor [B, Q, 4] with predicted boxes.
                "pred_masks" (optional): Mask predictions.
            targets (list[dict]): Per-image target dicts with "labels",
                "boxes", and optionally "masks".
            group_detr (int): Number of query groups for group DETR
                matching. Total queries Q must be divisible by group_detr.

        Returns:
            list[tuple]: List of (row_indices, col_indices) tuples per
                batch element where row_indices are matched query indices
                and col_indices are matched target indices.
        """
        
        bs = ops.shape(outputs["pred_logits"])[0]
        num_queries = ops.shape(outputs["pred_logits"])[1]

        # Compute the full cost matrix: shape (B*Q, total_targets)
        C = self.compute_cost_matrix(outputs, targets)
            
        # Cast to float32 and reshape from (B*Q, total_targets) to
        # (B, Q, total_targets) for per-image processing
        C = ops.cast(C, "float32")

        # Number of targets per batch element, used for splitting the
        # cost matrix along the target dimension
        sizes = [len(v["boxes"]) for v in targets]
        
        C_reshaped = ops.reshape(C, (bs, num_queries, -1))

        indices = []
        g_num_queries = num_queries // group_detr
        
        # Split the cost matrix along the query dimension for group DETR.
        # Each group of queries is independently matched to all targets.
        if group_detr > 1:
            C_lists = ops.split(C_reshaped, group_detr, axis=1)
        else:
            C_lists = [C_reshaped]
        
        import tensorflow as tf
        
        def optimize_linear_assignment(cost_matrix):
            """Solves the linear sum assignment on a numpy cost matrix.

            Args:
                cost_matrix: Array of shape (num_queries, num_targets).

            Returns:
                tuple: (row_indices, col_indices) as int64 arrays.
            """
            cost_matrix = np.array(cost_matrix)
            # Replace non-finite values to ensure the solver converges
            cost_matrix[np.isinf(cost_matrix) | np.isnan(cost_matrix)] = 1e6
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            return row_ind.astype(np.int64), col_ind.astype(np.int64)

        # Iterate over each query group and solve the assignment per image
        for g_i in range(group_detr):
            C_g = C_lists[g_i]  # (B, Q_per_group, total_targets)
            
            indices_g = []
            start = 0
            
            for i in range(bs):
                c_i = C_g[i]  # (Q_per_group, total_targets)
                end = start + sizes[i]
                
                # Extract cost sub-matrix for this image's targets only
                c_i_split = c_i[:, start:end] 
                
                # Solve the linear assignment using scipy. Uses
                # tf.numpy_function for TensorFlow graph mode compatibility.
                if keras.backend.backend() == 'tensorflow':
                    row_ind, col_ind = tf.numpy_function(
                        optimize_linear_assignment, 
                        [c_i_split], 
                        [tf.int64, tf.int64]
                    )
                else:
                    c_i_np = ops.convert_to_numpy(c_i_split)
                    c_i_np[np.isinf(c_i_np) | np.isnan(c_i_np)] = 1e6
                    row_ind, col_ind = linear_sum_assignment(c_i_np)
                    row_ind = ops.convert_to_tensor(row_ind, dtype="int64")
                    col_ind = ops.convert_to_tensor(col_ind, dtype="int64")

                indices_g.append((row_ind, col_ind))
                
                start = end
            
            # Merge group indices: for the first group, store directly;
            # for subsequent groups, offset query indices by the group
            # stride and concatenate with previous results
            if g_i == 0:
                indices = indices_g
            else:
                 new_indices = []
                 for (r1, c1), (r2, c2) in zip(indices, indices_g):
                     r_cat = ops.concatenate([r1, r2 + g_num_queries * g_i], axis=0)
                     c_cat = ops.concatenate([c1, c2], axis=0)
                     new_indices.append((r_cat, c_cat))
                 indices = new_indices

        return indices

