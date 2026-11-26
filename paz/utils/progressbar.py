import sys
import time
from collections import namedtuple

import jax
import jax.numpy as jp
import numpy as np

_LAST_LENGTH = 0
fields = ["description", "total_value", "width"]
Stage = namedtuple("Stage", fields + ["count", "start_time", "end_time"])


def start():
    return time.perf_counter()


def newline():
    def _newline(_):
        global _LAST_LENGTH
        _LAST_LENGTH = 0
        sys.stdout.write("\n")
        sys.stdout.flush()

    jax.debug.callback(_newline, jp.array(0), ordered=True)


def _build_message(now_value, total_value, start_time, description, width):
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
    return (
        f"\r{description}: |{bar}| {int(now_value)}/{int(total_value)} "
        f"({percent_complete:.0%}) "
        f"[{elapsed_time:.2f}s<{eta:.2f}s, {iterations_per_second:.2f}it/s]"
    )


def _append_suffix(message, suffix):
    if not suffix:
        return message
    return f"{message} | {suffix}"


def _pad_message(message):
    global _LAST_LENGTH
    num_spaces = max(_LAST_LENGTH - len(message), 0)
    _LAST_LENGTH = len(message)
    return f"{message}{' ' * num_spaces}"


def _print_bar(now_value, total_value, start_time, description, width, suffix):
    message = _build_message(
        now_value, total_value, start_time, description, width
    )
    message = _append_suffix(message, suffix)
    message = _pad_message(message)
    sys.stdout.write(message)
    sys.stdout.flush()


def print_bar(
    now_value,
    total_value,
    start_time,
    description="progress",
    width=50,
    suffix=None,
):
    _print_bar(now_value, total_value, start_time, description, width, suffix)


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


def skip(now_value):
    return None


def show_stages(stages):
    """Shows several stages of one process on a single line.

    # Arguments
        stages: List of `(description, total_value, width)`, in run order.

    # Returns
        List with one callback per stage, each taking the current count.
        They are safe to call under `jit`. End the line with `newline`.
    """
    states = [Stage(*stage, 0, None, None) for stage in stages]

    def build_report(index):
        def report(now_value):
            def callback(count):
                states[index] = _update_stage(states[index], int(count))
                _write_line("  ".join(map(_build_stage_text, states)))

            jax.debug.callback(callback, jp.asarray(now_value), ordered=True)

        return report

    return [build_report(index) for index in range(len(states))]


def _update_stage(stage, count):
    now = time.perf_counter()
    start_time = now if stage.start_time is None else stage.start_time
    end_time = now if count >= stage.total_value else None
    return stage._replace(count=count, start_time=start_time, end_time=end_time)


def _write_line(text):
    sys.stdout.write(_pad_message(f"\r{text}"))
    sys.stdout.flush()


def _build_stage_text(stage):
    filled_width = int(stage.width * stage.count / max(stage.total_value, 1))
    bar = "█" * filled_width + "-" * (stage.width - filled_width)
    text = f"{stage.description} |{bar}| {stage.count}/{stage.total_value}"
    if stage.end_time is not None:
        text = f"{text} {stage.end_time - stage.start_time:.2f}s"
    elif stage.start_time is not None:
        text = f"{text} {_build_timing_text(stage)}"
    return text


def _build_timing_text(stage):
    elapsed = time.perf_counter() - stage.start_time
    if stage.count == 0:
        remaining = float("inf")
    else:
        remaining = elapsed * (stage.total_value - stage.count) / stage.count
    percent_complete = stage.count / stage.total_value
    return f"{percent_complete:.0%} [{elapsed:.2f}s<{remaining:.2f}s]"
