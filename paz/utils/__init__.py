import os
import shutil
import functools

import jax
import jax.numpy as jp


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
    if update:
        save_snapshot(now_filedata, filepath)
    elif os.path.isfile(filepath):
        old_filedata = jp.load(filepath)
        assert jp.allclose(old_filedata, now_filedata, rtol, atol)
        print_green(f"✅ Snapshot {os.path.basename(filepath)} matches.")
    else:
        message = f"Missing snapshot {filepath}. Create it with update=True."
        raise FileNotFoundError(message)


def save_snapshot(now_filedata, filepath):
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    print_yellow(f"Writing golden file: {filepath}")
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


def __getattr__(name):
    if name == "time":
        from paz.utils.timing import time

        return time
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
