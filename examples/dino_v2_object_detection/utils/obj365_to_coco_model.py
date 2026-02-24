import keras
import keras.ops as k
import numpy as np

def get_coco_pretrain_from_obj365(cur_weights, pretrain_weights):
    """
    Get coco weights from obj365 pretrained model.
    
    Args:
        cur_weights: logic/tensor for current model weights (e.g. classification head bias/kernel)
        pretrain_weights: weights from obj365 pretrained model
    
    Returns:
        Updated weights with obj365 initialization for COCO classes.
    """
    # Assuming inputs are Keras tensors or params (Variable) or Numpy arrays.
    # We will work with them as Tensors/Numpy.
    
    # Check shapes. If they match, just return pretrain.
    # Note: Keras shapes are typically (N, C) or (H, W, C, N). 
    # RF-DETR class embed via obj365 is usually (num_classes, hidden_dim).
    # If the input is the weight matrix of a Dense layer (input_dim, num_classes), we might need to transpose logic
    # or assume this function is called on the bias or specific Embedding weights.
    # The original code `pretrain_tensor.size() == cur_tensor.size()` implies direct match check.
    
    cur_shape = k.shape(cur_weights)
    pretrain_shape = k.shape(pretrain_weights)
    
    # Simple shape check
    if tuple(cur_shape) == tuple(pretrain_shape):
        return pretrain_weights

    # We need to perform the mapping.
    # Using Numpy for index manipulation is easiest as this is typically done once at init.
    
    # If inputs are Variables/Tensors, convert to numpy for assignment logic if possible, 
    # or use scatter update. Keras ops scatter_update exists.
    
    # COCO ids (1-indexed in original list, but let's see usage)
    # Original logic: cur_tensor[coco_id] = pretrain_tensor[obj_id + 1]
    # This implies cur_tensor is (NumClasses, ...)
    
    coco_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74,
        75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
    ]
    obj365_ids = [
        0, 46, 5, 58, 114, 55, 116, 65, 21, 40, 176, 127, 249, 24, 56, 139, 92, 78, 99, 96,
        144, 295, 178, 180, 38, 39, 13, 43, 120, 219, 148, 173, 165, 154, 137, 113, 145, 146,
        204, 8, 35, 10, 88, 84, 93, 26, 112, 82, 265, 104, 141, 152, 234, 143, 150, 97, 2,
        50, 25, 75, 98, 153, 37, 73, 115, 132, 106, 61, 163, 134, 277, 81, 133, 18, 94, 30,
        169, 70, 328, 226
    ]

    # Convert to standard Keras tensor ops
    # We create a copy of cur_weights to modify
    # But wait, Keras tensors are immutable (mostly). variables are mutable.
    # If this is called during model building with symbolic tensors, we can't do in-place assignment easily.
    # If this is called after loading weights (numpy), we can.
    # Keras 3 style: assume we return a NEW tensor.
    
    # Use scatter update approach.
    indices = [[cid] for cid in coco_ids]
    updates = [pretrain_weights[oid + 1] for oid in obj365_ids]
    
    cur_weights_np = k.convert_to_numpy(cur_weights).copy()
    pretrain_weights_np = k.convert_to_numpy(pretrain_weights)
    
    for i, cid in enumerate(coco_ids):
        oid = obj365_ids[i]
        cur_weights_np[cid] = pretrain_weights_np[oid + 1]
        
    return cur_weights_np
