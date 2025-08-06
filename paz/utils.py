import os
import shutil
import functools
import time as pytime
import jax
import jax.numpy as jp


def time(function, jax_fn=True, name=None):

    if name is not None:
        name = name
    elif hasattr(function, "__name__"):
        name = function.__name__
    else:
        name = f"{function.__class__.__name__}.__call__"

    class Colors:
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        RESET = "\033[0m"

    def print_time(run_time):
        if run_time < 0.05:
            color = Colors.GREEN
        elif run_time < 0.2:
            color = Colors.YELLOW
        else:
            color = Colors.RED

        print(f"{color}'{name}': {run_time:.4f} s {Colors.RESET}")

    @functools.wraps(function)
    def jax_wrapper(*args, **kwargs):
        start_time = pytime.perf_counter()
        # value = function(*args, **kwargs).block_until_ready()
        value = function(*args, **kwargs)
        jax.block_until_ready(value)
        end_time = pytime.perf_counter()
        run_time = end_time - start_time
        print_time(run_time)
        return value

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        start_time = pytime.perf_counter()
        value = function(*args, **kwargs)
        end_time = pytime.perf_counter()
        run_time = end_time - start_time
        print_time(run_time)
        return value

    return jax_wrapper if jax_fn else wrapper


def on_device(device):
    """Decorator factory to specify default device for JAX operations."""

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with jax.default_device(device):
                print(f"--- INFO: Entering context for device: {device} ---")
                result = function(*args, **kwargs)
                print(f"--- INFO: Exiting context for device: {device} ---")
            return result

        return wrapper

    return decorator


def extract(filepath):
    output_path = os.path.dirname(filepath)
    shutil.unpack_archive(filepath, output_path)
    return output_path


def assert_snapshot(now_filedata, filepath, update=False, rtol=1e-07, atol=0):
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if os.path.isfile(filepath) and not update:
        old_filedata = jp.load(filepath)
        assert jp.allclose(old_filedata, now_filedata, rtol, atol)
        print_green(f"✅ Snapshot {os.path.basename(filepath)} matches.")
    else:
        if update:
            print_yellow(f"Updating golden file: {filepath}")
        else:
            print_yellow(f"File not found, creating snapshot: {filepath}")
        jp.save(filepath, now_filedata)


def print_with_color(text, ansi_code):
    """The base function that applies any ANSI color code and a reset."""
    reset_code = "\033[0m"
    print(f"{ansi_code}{text}{reset_code}")


def print_green(text):
    green_code = "\033[92m"
    print_with_color(text, green_code)


def print_yellow(text):
    yellow_code = "\033[93m"
    print_with_color(text, yellow_code)


def print_red(text):
    red_code = "\033[91m"
    print_with_color(text, red_code)
