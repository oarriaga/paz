import numpy as np
import keras
from keras import ops
from scipy.optimize import linear_sum_assignment

# Adjust imports based on your project structure
# Adjust imports based on your project structure
from paz.models.detection.dino_v2_object_detection.utils.box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
    batch_sigmoid_ce_loss,
    batch_dice_loss
)
from paz.models.detection.dino_v2_object_detection.models.segmentation_head.segmentation_head_keras import point_sample


class HungarianMatcher(keras.layers.Layer):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha: float = 0.25,
                 use_pos_only: bool = False, use_position_modulated_cost: bool = False, 
                 mask_point_sample_ratio: int = 16, cost_mask_ce: float = 1, cost_mask_dice: float = 1, **kwargs):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
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

    def call(self, outputs, targets, group_detr=1):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 "masks": Tensor of dim [num_target_boxes, H, W] containing the target mask coordinates
            group_detr: Number of groups used for matching.
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        
    def compute_cost_matrix(self, outputs, targets):
        """
        Computes the cost matrix C between outputs and targets.
        """
        # outputs["pred_logits"] shape: [Batch, NumQueries, NumClasses]
        flat_pred_logits = ops.reshape(outputs["pred_logits"], (-1, ops.shape(outputs["pred_logits"])[-1]))
        out_bbox = ops.reshape(outputs["pred_boxes"], (-1, 4))
        
        tgt_ids = ops.concatenate([v["labels"] for v in targets], axis=0)
        tgt_bbox = ops.concatenate([v["boxes"] for v in targets], axis=0)
        
        out_prob = ops.sigmoid(flat_pred_logits)
        
        masks_present = "masks" in targets[0]

        # Compute the giou cost betwen boxes
        giou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -giou

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        
        def log_sigmoid(x):
            return -ops.softplus(-x) 
        
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-log_sigmoid(-flat_pred_logits))
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-log_sigmoid(flat_pred_logits))
        
        tgt_ids = ops.cast(tgt_ids, "int32")
        pos_cost_subset = ops.take(pos_cost_class, tgt_ids, axis=1)
        neg_cost_subset = ops.take(neg_cost_class, tgt_ids, axis=1)
        cost_class = pos_cost_subset - neg_cost_subset

        # Compute the L1 cost between boxes
        out_bbox_exp = ops.expand_dims(out_bbox, 1) # (N, 1, 4)
        tgt_bbox_exp = ops.expand_dims(tgt_bbox, 0) # (1, M, 4)
        cost_bbox = ops.sum(ops.abs(out_bbox_exp - tgt_bbox_exp), axis=-1)

        cost_mask_ce = 0.0
        cost_mask_dice = 0.0

        if masks_present:
            tgt_masks = ops.concatenate([v["masks"] for v in targets], axis=0)

            if "pred_masks" in outputs and (ops.is_tensor(outputs["pred_masks"]) or isinstance(outputs["pred_masks"], (keras.KerasTensor, np.ndarray))):
                 out_masks = outputs["pred_masks"]
                 out_masks = ops.reshape(out_masks, (-1, ops.shape(out_masks)[-2], ops.shape(out_masks)[-1]))
                 
                 H_mask = ops.shape(out_masks)[-2]
                 W_mask = ops.shape(out_masks)[-1]
                 num_points = (H_mask * W_mask) // self.mask_point_sample_ratio
                 
                 point_coords = keras.random.uniform((1, num_points, 2), minval=0.0, maxval=1.0)
                 
                 out_masks_exp = ops.expand_dims(out_masks, 1)
                 batch_len = ops.shape(out_masks)[0]
                 point_coords_rep = ops.broadcast_to(point_coords, (batch_len, num_points, 2))
                 
                 sample_out = point_sample(out_masks_exp, point_coords_rep, align_corners=False)
                 pred_masks_logits = ops.squeeze(sample_out, 1)
                 
            else:
                 spatial_features = outputs["pred_masks"]["spatial_features"]
                 query_features = outputs["pred_masks"]["query_features"]
                 bias = outputs["pred_masks"]["bias"]
                 
                 H_sf = ops.shape(spatial_features)[-2]
                 W_sf = ops.shape(spatial_features)[-1]
                 num_points = (H_sf * W_sf) // self.mask_point_sample_ratio
                 
                 point_coords = keras.random.uniform((1, num_points, 2), minval=0.0, maxval=1.0)
                 
                 batch_len = ops.shape(spatial_features)[0]
                 point_coords_rep = ops.broadcast_to(point_coords, (batch_len, num_points, 2))
                 
                 pred_masks_logits = point_sample(spatial_features, point_coords_rep, align_corners=False)
                 pred_masks_logits = ops.einsum('bcp,bnc->bnp', pred_masks_logits, query_features) + bias
                 pred_masks_logits = ops.reshape(pred_masks_logits, (-1, ops.shape(pred_masks_logits)[-1]))
            
            tgt_masks = ops.cast(tgt_masks, pred_masks_logits.dtype)
            tgt_masks_exp = ops.expand_dims(tgt_masks, 1)
            tgt_len = ops.shape(tgt_masks)[0]
            point_coords_tgt = ops.broadcast_to(point_coords, (tgt_len, num_points, 2))
            
            tgt_masks_flat = point_sample(tgt_masks_exp, point_coords_tgt, align_corners=False)
            tgt_masks_flat = ops.squeeze(tgt_masks_flat, 1)

            cost_mask_ce = batch_sigmoid_ce_loss(pred_masks_logits, tgt_masks_flat)
            cost_mask_dice = batch_dice_loss(pred_masks_logits, tgt_masks_flat)

        C = (self.cost_bbox * cost_bbox) + (self.cost_class * cost_class) + (self.cost_giou * cost_giou)
        
        if masks_present:
            C = C + (self.cost_mask_ce * cost_mask_ce) + (self.cost_mask_dice * cost_mask_dice)
            
        return C

    def call(self, outputs, targets, group_detr=1):
        """ Performs the matching """
        
        # outputs["pred_logits"] shape: [Batch, NumQueries, NumClasses]
        bs = ops.shape(outputs["pred_logits"])[0]
        num_queries = ops.shape(outputs["pred_logits"])[1]

        C = self.compute_cost_matrix(outputs, targets)
            
        # C is (N_total, M_total). N_total = batch * num_queries. M_total = sum(num_targets_per_image).
        
        # We need to split C back into batch elements.
        C = ops.cast(C, "float32") # ensure float for cpu?

        sizes = [len(v["boxes"]) for v in targets]
        
        # Reshape C
        C_reshaped = ops.reshape(C, (bs, num_queries, -1))

        indices = []
        g_num_queries = num_queries // group_detr
        
        # Split C for group DETR
        if group_detr > 1:
            C_lists = ops.split(C_reshaped, group_detr, axis=1)
        else:
            C_lists = [C_reshaped]
        
        import tensorflow as tf
        
        def optimize_linear_assignment(cost_matrix):
            # cost_matrix: (num_queries, num_targets)
            # Replace NaNs and Infs (numpy logic inside wrapper)
            cost_matrix = np.array(cost_matrix)
            cost_matrix[np.isinf(cost_matrix) | np.isnan(cost_matrix)] = 1e6 # simple clamp
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            return row_ind.astype(np.int64), col_ind.astype(np.int64)

        for g_i in range(group_detr):
            C_g = C_lists[g_i] # (Batch, G_NumQueries, TotalTargets)
            
            indices_g = []
            start = 0
            
            for i in range(bs):
                c_i = C_g[i] # (G_NumQueries, TotalTargets)
                end = start + sizes[i]
                
                # Split along last dim
                c_i_split = c_i[:, start:end] 
                
                # TF Graph Mode Support
                if keras.backend.backend() == 'tensorflow':
                    row_ind, col_ind = tf.numpy_function(
                        optimize_linear_assignment, 
                        [c_i_split], 
                        [tf.int64, tf.int64]
                    )
                    # Set shapes because numpy_function loses them
                    # row_ind.set_shape([None]) 
                    # col_ind.set_shape([None]) # Actually we know length is min(queries, targets)
                else:
                    # Eager/JAX/Torch
                    c_i_np = ops.convert_to_numpy(c_i_split)
                    # Handle nan/inf
                    c_i_np[np.isinf(c_i_np) | np.isnan(c_i_np)] = 1e6
                    row_ind, col_ind = linear_sum_assignment(c_i_np)
                    row_ind = ops.convert_to_tensor(row_ind, dtype="int64")
                    col_ind = ops.convert_to_tensor(col_ind, dtype="int64")

                indices_g.append((row_ind, col_ind))
                
                start = end
            
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

        # Return list of tuples of tensors (int64)
        # [(tensor(i), tensor(j)), ...]
        
        return [(ops.convert_to_tensor(i, dtype="int64"), ops.convert_to_tensor(j, dtype="int64")) for i, j in indices]

