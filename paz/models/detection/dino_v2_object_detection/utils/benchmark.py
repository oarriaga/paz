import time
import os
import json
import numpy as np
import tqdm
import keras
import keras.ops as k
from typing import Any, Dict, Sequence


def warmup(model, inputs, N=10):
    """Run *N* forward passes to warm up the compute graph and caches.

    Args:
        model: Keras model to warm up.
        inputs: Input tensor (batched).
        N (int): Number of warm-up iterations.
    """
    for i in range(N):
        _ = model(inputs)


def measure_time(model, inputs, N=10):
    """Measure the average inference time over *N* forward passes.

    A short warm-up phase is performed before timing begins.
    After each forward pass the result is converted to NumPy to
    force device synchronization before the next timing sample.

    Args:
        model: Keras model to benchmark.
        inputs: Batched input tensor.
        N (int): Number of timed forward passes.

    Returns:
        float: Average wall-clock time per forward pass (seconds).
    """
    warmup(model, inputs, N=5)
    
    start_time = time.time()
    for i in range(N):
        res = model(inputs)
        # Convert to numpy to force device synchronization
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
    """Compute summary statistics for an array of measurements.

    Args:
        data (np.ndarray): Array of numeric values.

    Returns:
        dict: Dictionary with keys ``mean``, ``std``, ``min``, ``max``.
    """
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def benchmark(model, dataset, output_dir):
    """Benchmark a model on a dataset and report latency and FPS.

    Collects up to ``total_step`` images from *dataset*, discards the
    first ``warmup_step`` measurements to eliminate cold-start effects,
    then reports latency statistics and frames-per-second.

    Args:
        model: Keras model to benchmark.
        dataset: Iterable yielding images or ``(image, target)`` tuples.
        output_dir (str | None): If given, results are appended to
            ``<output_dir>/benchmark/log.txt``.

    Returns:
        dict: Benchmark results including parameter count, latency
            statistics, and FPS.
    """
    print("Get model size and FPS")
    _outputs = {}
    
    # Count trainable parameters
    n_parameters = sum([np.prod(v.shape) for v in model.trainable_variables])
    _outputs.update({"nparam": int(n_parameters)})

    warmup_step = 5
    total_step = 20
    
    # Collect images from the dataset (up to total_step samples)
    images = []
    iterator = iter(dataset)
    for _ in range(total_step):
        try:
            data = next(iterator)
            if isinstance(data, (tuple, list)):
                images.append(data[0]) 
            else:
                images.append(data)
        except StopIteration:
            break
            
    if not images:
        print("No images found in dataset for benchmarking.")
        return _outputs

    latencies = []
    
    for img_id, img in enumerate(tqdm.tqdm(images)):
        # Add batch dimension for single images
        if len(img.shape) == 3:
            inputs = k.expand_dims(img, 0)
        else:
            inputs = img
            
        t = measure_time(model, inputs, N=1)
        
        # Skip early iterations treated as warm-up
        if img_id >= warmup_step:
            latencies.append(t)
            
    _outputs.update({"time": fmt_res(np.array(latencies))})
    
    mean_infer_time = float(fmt_res(np.array(latencies))["mean"])
    if mean_infer_time > 0:
        _outputs.update({"fps": 1 / mean_infer_time})
        
    res = _outputs
    
    # Persist results to a JSON log file
    if output_dir:
        os.makedirs(os.path.join(output_dir, "benchmark"), exist_ok=True)
        with open(os.path.join(output_dir, "benchmark", "log.txt"), "a") as f:
            f.write("Test benchmark on Val Dataset" + "\n")
            f.write(json.dumps(_outputs, indent=2) + "\n")
            
    return _outputs
