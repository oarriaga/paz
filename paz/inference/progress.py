import sys
import time


def now():
    return time.perf_counter()


def move_to_next_line():
    sys.stdout.write("\n")
    sys.stdout.flush()


def draw_bar(now_arg, total, start_time, description, width):
    total = float(total)
    now_arg = float(now_arg)
    if total == 0:
        percent_complete = 0.0
    else:
        percent_complete = min(max(now_arg / total, 0.0), 1.0)
    filled_width = int(width * percent_complete)
    bar = "#" * filled_width + "-" * (width - filled_width)
    elapsed_time = time.perf_counter() - start_time
    if now_arg > 0 and elapsed_time > 0:
        iters_per_sec = now_arg / elapsed_time
        eta = (total - now_arg) / iters_per_sec
    else:
        iters_per_sec, eta = 0.0, float("inf")
    message = (
        f"\r{description}: |{bar}| {int(now_arg)}/{int(total)} "
        f"({percent_complete:.0%}) "
        f"[{elapsed_time:.2f}s<{eta:.2f}s, {iters_per_sec:.2f}it/s]"
    )
    sys.stdout.write(message)
    sys.stdout.flush()


def build_bar_callback(total, start_time, description, width):
    return lambda now_arg: draw_bar(
        now_arg, total, start_time, description, width
    )
