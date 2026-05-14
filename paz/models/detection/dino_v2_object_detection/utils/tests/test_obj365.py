import os
import sys
import importlib.util
import numpy as np
import pytest
import keras
import keras.ops as k

# Dynamic import for obj365_to_coco_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import obj365_to_coco_model

def test_get_coco_pretrain_from_obj365():
    # Obj365 classes: 365
    # COCO classes: 91
    # The function maps specific indices.
    
    # Create dummy pretrain weights (366, 10) - 365 classes + 1 background? Index 0 is background?
    # Original code: pretrain_tensor[obj_id + 1]. 
    # obj365_ids values go up to 328. Max is 365.
    
    pretrain = np.zeros((366, 10), dtype=np.float32)
    for i in range(366):
        pretrain[i] = i # Set value to index for easy verification
        
    cur_weights = np.zeros((92, 10), dtype=np.float32) # COCO has 91 classes + 1?
    # The function expects cur_tensor to be modified in place or returned new.
    # The function iterates coco_ids which go up to 90.
    
    # We call the function
    new_weights = obj365_to_coco_model.get_coco_pretrain_from_obj365(cur_weights, pretrain)
    
    # Check a few mappings
    # coco_id 1 -> obj_id 0 -> pretrain index 1 (value 1)
    # coco_id 2 -> obj_id 46 -> pretrain index 47 (value 47)
    
    res = k.convert_to_numpy(new_weights)
    
    assert np.allclose(res[1], 1.0)
    assert np.allclose(res[2], 47.0)
    
    # Check that untouched indices (e.g. 0 if not in list) remain 0
    if 0 not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74,
        75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]:
        assert np.allclose(res[0], 0.0)

