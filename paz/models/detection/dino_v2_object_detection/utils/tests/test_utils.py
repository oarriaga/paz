import os
import sys
import importlib.util

# Add parent directory to path to allow importing utils - keeping this for potential other deps
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import keras
import keras.ops as k

# Load utils.py explicitly by path to avoid conflict with standard 'utils' modules
utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils.py')
spec = importlib.util.spec_from_file_location("keras_utils", utils_path)
keras_utils = importlib.util.module_from_spec(spec)
sys.modules["keras_utils"] = keras_utils
spec.loader.exec_module(keras_utils)

def test_best_metric_single():
    bm = keras_utils.BestMetricSingle(init_res=0.0, better='large')
    assert bm.best_res == 0.0
    
    updated = bm.update(0.5, 1)
    assert updated
    assert bm.best_res == 0.5
    assert bm.best_ep == 1
    
    updated = bm.update(0.4, 2)
    assert not updated
    assert bm.best_res == 0.5

    bm_small = keras_utils.BestMetricSingle(init_res=10.0, better='small')
    updated = bm_small.update(5.0, 1)
    assert updated
    assert bm_small.best_res == 5.0

def test_best_metric_holder():
    bmh = keras_utils.BestMetricHolder(init_res=0.0, better='large', use_ema=True)
    
    # Update regular
    updated = bmh.update(0.5, 1, is_ema=False)
    assert updated # best_all updated
    assert bmh.best_regular.best_res == 0.5
    assert bmh.best_all.best_res == 0.5
    
    # Update EMA with better
    updated = bmh.update(0.6, 1, is_ema=True)
    assert updated
    assert bmh.best_ema.best_res == 0.6
    assert bmh.best_all.best_res == 0.6
    
    # Update regular with worse
    updated = bmh.update(0.4, 2, is_ema=False)
    assert not updated
    
def test_model_ema():
    # Simple model
    inputs = keras.Input(shape=(10,))
    outputs = keras.layers.Dense(1, kernel_initializer='ones', bias_initializer='zeros')(inputs)
    model = keras.Model(inputs, outputs)
    
    ema = keras_utils.ModelEma(model, decay=0.5)
    
    # Initial weights: kernel=1, bias=0
    w_initial = model.get_weights()
    assert np.all(w_initial[0] == 1.0)
    
    # Update model weights
    new_w = [np.full((10, 1), 2.0, dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    model.set_weights(new_w)
    
    # Update EMA: 0.5 * 1.0 + 0.5 * 2.0 = 1.5
    ema.update(model)
    
    # model_weights is keyed by variable path (e.g. "dense/kernel")
    kernel_path = model.weights[0].path
    assert np.allclose(ema.model_weights[kernel_path], 1.5)
    
    # Apply to model
    ema.apply_to(model)
    w_applied = model.get_weights()
    assert np.allclose(w_applied[0], 1.5)
