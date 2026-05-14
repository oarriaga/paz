import keras
import keras.ops as k
import numpy as np


def get_coco_pretrain_from_obj365(cur_weights, pretrain_weights):
    """Initialize COCO weights from an Objects365-pretrained model.

    For each of the 80 COCO categories, copies the corresponding row
    from *pretrain_weights* (indexed by the Objects365 category mapping)
    into *cur_weights*.  If the shapes already match, *pretrain_weights*
    is returned directly.

    Args:
        cur_weights: Current model weights (tensor or variable) for
            the COCO classification head, shaped ``(num_classes, ...)``.
        pretrain_weights: Weights from the Objects365 pretrained model,
            shaped ``(365_classes, ...)``.

    Returns:
        np.ndarray: Updated weight array with COCO rows filled from
            the Objects365 mapping.
    """
    cur_shape = k.shape(cur_weights)
    pretrain_shape = k.shape(pretrain_weights)
    
    # If shapes match, no remapping is needed
    if tuple(cur_shape) == tuple(pretrain_shape):
        return pretrain_weights

    # COCO category IDs (1-indexed, non-contiguous: 80 categories)
    coco_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74,
        75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
    ]
    # Corresponding Objects365 category indices (0-indexed)
    obj365_ids = [
        0, 46, 5, 58, 114, 55, 116, 65, 21, 40, 176, 127, 249, 24, 56, 139, 92, 78, 99, 96,
        144, 295, 178, 180, 38, 39, 13, 43, 120, 219, 148, 173, 165, 154, 137, 113, 145, 146,
        204, 8, 35, 10, 88, 84, 93, 26, 112, 82, 265, 104, 141, 152, 234, 143, 150, 97, 2,
        50, 25, 75, 98, 153, 37, 73, 115, 132, 106, 61, 163, 134, 277, 81, 133, 18, 94, 30,
        169, 70, 328, 226
    ]

    # Copy weights row-by-row using the COCO-to-Objects365 mapping.
    # cur_weights[coco_id] = pretrain_weights[obj365_id + 1]
    cur_weights_np = k.convert_to_numpy(cur_weights).copy()
    pretrain_weights_np = k.convert_to_numpy(pretrain_weights)
    
    for i, cid in enumerate(coco_ids):
        oid = obj365_ids[i]
        cur_weights_np[cid] = pretrain_weights_np[oid + 1]
        
    return cur_weights_np
