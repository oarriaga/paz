import time
import os
import json
import numpy as np
import tqdm
import keras.ops as k

WARMUP_STEPS = 5
TOTAL_STEPS = 20


def warmup(model, inputs, N=10):
    for _ in range(N):
        model(inputs)


def force_device_sync(result):
    if isinstance(result, (list, tuple)):
        value = result[0]
    elif isinstance(result, dict):
        value = list(result.values())[0]
    else:
        value = result
    return k.convert_to_numpy(value)


def measure_time(model, inputs, N=10):
    warmup(model, inputs, N=5)
    start_time = time.time()
    for _ in range(N):
        # Converting to numpy forces the device to finish the step.
        force_device_sync(model(inputs))
    return (time.time() - start_time) / N


def fmt_res(data):
    keys = ("mean", "std", "min", "max")
    values = (np.mean(data), np.std(data), np.min(data), np.max(data))
    return {key: float(value) for key, value in zip(keys, values)}


def collect_benchmark_images(dataset, total_steps):
    images = []
    iterator = iter(dataset)
    for _ in range(total_steps):
        try:
            data = next(iterator)
        except StopIteration:
            break
        images.append(data[0] if isinstance(data, (tuple, list)) else data)
    return images


def measure_latencies(model, images, warmup_steps):
    latencies = []
    for index, image in enumerate(tqdm.tqdm(images)):
        inputs = k.expand_dims(image, 0) if len(image.shape) == 3 else image
        elapsed = measure_time(model, inputs, N=1)
        # The first steps are treated as warm-up and dropped.
        if index >= warmup_steps:
            latencies.append(elapsed)
    return latencies


def write_benchmark_log(output_dir, outputs):
    if output_dir:
        directory = os.path.join(output_dir, "benchmark")
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "log.txt"), "a") as handle:
            handle.write("Test benchmark on Val Dataset" + "\n")
            handle.write(json.dumps(outputs, indent=2) + "\n")


def benchmark(model, dataset, output_dir):
    print("Get model size and FPS")
    num_parameters = sum(np.prod(v.shape) for v in model.trainable_variables)
    outputs = {"nparam": int(num_parameters)}
    images = collect_benchmark_images(dataset, TOTAL_STEPS)
    if not images:
        print("No images found in dataset for benchmarking.")
    else:
        latencies = np.array(measure_latencies(model, images, WARMUP_STEPS))
        outputs["time"] = fmt_res(latencies)
        mean_infer_time = float(outputs["time"]["mean"])
        if mean_infer_time > 0:
            outputs["fps"] = 1 / mean_infer_time
        write_benchmark_log(output_dir, outputs)
    return outputs
