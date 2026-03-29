import sys
import time

import jax
import jax.numpy as jp
import numpy as np


def start():
    return time.perf_counter()


def newline():
    def _newline(_):
        sys.stdout.write("\n")
        sys.stdout.flush()

    jax.debug.callback(_newline, jp.array(0), ordered=True)


def _print_bar(now_value, total_value, start_time, description, width):
    now_value = float(np.array(now_value))
    total_value = float(np.array(total_value))
    if total_value == 0.0:
        percent_complete = 0.0
    else:
        percent_complete = np.clip(now_value / total_value, 0.0, 1.0)
    filled_width = int(width * percent_complete)
    bar = "█" * filled_width + "-" * (width - filled_width)
    elapsed_time = time.perf_counter() - start_time
    if now_value > 0.0 and elapsed_time > 0.0:
        iterations_per_second = now_value / elapsed_time
        eta = (total_value - now_value) / iterations_per_second
    else:
        iterations_per_second = 0.0
        eta = float("inf")
    message = (
        f"\r{description}: |{bar}| {int(now_value)}/{int(total_value)} "
        f"({percent_complete:.0%}) "
        f"[{elapsed_time:.2f}s<{eta:.2f}s, {iterations_per_second:.2f}it/s]"
    )
    sys.stdout.write(message)
    sys.stdout.flush()


def print_bar(
    now_value, total_value, start_time, description="progress", width=50
):
    _print_bar(now_value, total_value, start_time, description, width)


def draw(now_value, total_value, start_time, description="progress", width=50):
    def callback(current, total):
        print_bar(current, total, start_time, description, width)

    jax.debug.callback(
        callback, jp.asarray(now_value), jp.asarray(total_value), ordered=True
    )


def show(total_value, description="progress", width=50):
    start_time = start()

    def _callback(now_value):
        draw(now_value, total_value, start_time, description, width)

    return _callback
