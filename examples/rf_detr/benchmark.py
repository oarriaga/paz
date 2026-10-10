"""Times an RF-DETR forward pass, one row per variant.

Reports the median of --num_runs compiled calls at each variant's own
training resolution, so the numbers are per image at different input sizes.
"""
import argparse
import os
import time

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import jax

import paz

VARIANTS = {
    "nano": paz.models.RFDETRNano,
    "small": paz.models.RFDETRSmall,
    "medium": paz.models.RFDETRMedium,
    "base": paz.models.RFDETRBase,
    "large": paz.models.RFDETRLarge,
}


def measure_forward(model, num_runs):
    """Median seconds per call, after one call to compile the graph."""
    image = np.zeros((1, *model.input_shape[1:]), "float32")
    model.predict(image, verbose=0)
    durations = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model.predict(image, verbose=0)
        durations.append(time.perf_counter() - start)
    return float(np.median(durations))


def count_parameters(model):
    return sum(int(np.prod(weight.shape)) for weight in model.weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default=list(VARIANTS), nargs="+",
                        choices=list(VARIANTS))
    parser.add_argument("--num_runs", default=10, type=int)
    args = parser.parse_args()

    print("device", jax.default_backend())
    print(f"{'variant':<10}{'input':>8}{'parameters':>13}{'seconds':>10}")
    for variant in args.variants:
        model = VARIANTS[variant]()
        duration = measure_forward(model, args.num_runs)
        resolution = model.input_shape[1]
        parameters = count_parameters(model)
        print(f"{variant:<10}{resolution:>8}{parameters:>13}{duration:>10.3f}")
