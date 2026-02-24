import time
import os
import json
import numpy as np
import tqdm
import keras
import keras.ops as k
from typing import Any, Dict, Sequence

def warmup(model, inputs, N=10):
    for i in range(N):
        _ = model(inputs)
    # Synchronization?
    # Converting to numpy forces synchronization in most backends
    # _ = k.convert_to_numpy(_[0] if isinstance(_, (list, tuple)) else _)

def measure_time(model, inputs, N=10):
    # Warmup
    warmup(model, inputs, N=5)
    
    start_time = time.time()
    for i in range(N):
        res = model(inputs)
        # Force sync
        if isinstance(res, (list, tuple)):
            _ = k.convert_to_numpy(res[0])
        elif isinstance(res, dict):
             _ = k.convert_to_numpy(list(res.values())[0])
        else:
             _ = k.convert_to_numpy(res)
             
    end_time = time.time()
    avg_time = (end_time - start_time) / N
    return avg_time

def fmt_res(data):
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }

def benchmark(model, dataset, output_dir):
    print("Get model size and FPS")
    _outputs = {}
    
    # Parameter count
    n_parameters = sum([np.prod(v.shape) for v in model.trainable_variables])
    _outputs.update({"nparam": int(n_parameters)})

    # model.eval() # Keras models in inference mode by default or controlled by training=False argument
    
    warmup_step = 5
    total_step = 20
    
    # Collect images
    images = []
    # Dataset assumption: it yields (img, target)
    # We need to iterate it.
    
    # We assume dataset is iterable or indexable
    iterator = iter(dataset)
    for _ in range(total_step):
        try:
            data = next(iterator)
            if isinstance(data, (tuple, list)):
                images.append(data[0]) 
            else:
                images.append(data) # specific to data loader
        except StopIteration:
            break
            
    if not images:
        print("No images found in dataset for benchmarking.")
        return _outputs

    latencies = []
    
    for img_id, img in enumerate(tqdm.tqdm(images)):
        # Prepare input
        # img shape? model expects batch?
        # Assuming img is already a tensor or numpy array.
        # If it's a single image (C, H, W) or (H, W, C), add batch dim if needed
        
        # Check if batch dim exists
        if len(img.shape) == 3:
            inputs = k.expand_dims(img, 0)
        else:
            inputs = img
            
        t = measure_time(model, inputs, N=1) # Measure single inference? or N? Original code used measure_time which did N loops?
        # Original code: t = measure_time(model, inputs) -> loops N times inside?
        # Original code default N=10 in measure_time.
        # But inside loop it calls measure_time with N=10 default?
        # That means for each image it runs 10 times? Yes.
        
        if img_id >= warmup_step:
            latencies.append(t)
            
    _outputs.update({"time": fmt_res(np.array(latencies))})
    
    mean_infer_time = float(fmt_res(np.array(latencies))["mean"])
    if mean_infer_time > 0:
        _outputs.update({"fps": 1 / mean_infer_time})
        
    res = _outputs
    
    if output_dir:
        os.makedirs(os.path.join(output_dir, "benchmark"), exist_ok=True)
        with open(os.path.join(output_dir, "benchmark", "log.txt"), "a") as f:
            f.write("Test benchmark on Val Dataset" + "\n")
            f.write(json.dumps(_outputs, indent=2) + "\n")
            
    return _outputs
