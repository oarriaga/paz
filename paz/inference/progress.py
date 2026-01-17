import sys
import time


def now():
    return time.perf_counter()


def move_to_next_line():
    sys.stdout.write("\n")
    sys.stdout.flush()


def draw_bar(now_arg, total, start_time, description, width):
    percent_complete = _safe_divide(now_arg, total)
    bar = _make_bar(percent_complete, width)
    elapsed_time = time.perf_counter() - start_time
    iters_per_sec, eta = _compute_rates(now_arg, total, elapsed_time)
    message = (
        f"\r{description}: |{bar}| {int(now_arg)}/{int(total)} "
        f"({percent_complete:.0%}) "
        f"[{elapsed_time:.2f}s<{eta:.2f}s, {iters_per_sec:.2f}it/s]"
    )
    sys.stdout.write(message)
    sys.stdout.flush()


def _safe_divide(numerator, denominator):
    numerator = float(numerator)
    denominator = float(denominator)
    if denominator == 0:
        return 0.0
    return min(max(numerator / denominator, 0.0), 1.0)


def _make_bar(percent_complete, width):
    filled_width = int(width * percent_complete)
    return "#" * filled_width + "-" * (width - filled_width)


def _compute_rates(now_arg, total, elapsed_time):
    now_arg = float(now_arg)
    total = float(total)
    if now_arg > 0 and elapsed_time > 0:
        iters_per_sec = now_arg / elapsed_time
        eta = (total - now_arg) / iters_per_sec
        return iters_per_sec, eta
    return 0.0, float("inf")
