import sys
import time


def _draw_bar(now_arg, total, start_time, description, width):

    def make_progress_bar_string(percent_complete, width):
        filled_width = int(width * percent_complete)
        return "█" * filled_width + "-" * (width - filled_width)

    def compute_ETA(now_arg, total, iterations_per_second):
        remaining_iterations = total - now_arg
        if iterations_per_second > 0:
            ETA = remaining_iterations / iterations_per_second
        else:
            ETA = 0
        return ETA

    def format_string(bar, percent_complete, elapsed_time, ETA, iters_per_sec):
        return (
            f"\r{description}: [{bar}] {now_arg}/{total} "
            f"({percent_complete:.0%}) "
            f"[{elapsed_time:.2f}s<{ETA:.2f}s, {iters_per_sec:.2f}it/s]"
        )

    def print_to_console(message):
        sys.stdout.write(message)
        sys.stdout.flush()

    percent_complete = now_arg / total
    bar = make_progress_bar_string(percent_complete, width)
    elapsed_time = time.time() - start_time
    if now_arg > 0:
        iterations_per_second = now_arg / elapsed_time
        ETA = compute_ETA(now_arg, iterations_per_second, total)
    else:
        iterations_per_second = 0
        ETA = float("inf")
    message_args = (percent_complete, elapsed_time, ETA, iterations_per_second)
    message = format_string(bar, *message_args)
    print_to_console(message)


def bar(iterable, total=None, description="progress", width=50):

    def move_to_the_next_line():
        sys.stdout.write("\n")
        sys.stdout.flush()

    total = len(iterable) if total is None else total
    start_time = time.time()
    _draw_bar(0, total, start_time, description, width)
    try:
        for now_arg, item in enumerate(iterable, 1):
            yield item
            _draw_bar(now_arg, total, start_time, description, width)
    finally:
        move_to_the_next_line()


if __name__ == "__main__":
    # Example 1: Looping over a range
    for i in bar(range(150), description="Downloading Data"):
        time.sleep(0.02)

    # Example 2: Looping over a list of files
    files = ["file1.txt", "file2.img", "file3.mov", "file4.zip"]
    for filename in bar(files, description="Processing Files   "):
        time.sleep(0.5)
