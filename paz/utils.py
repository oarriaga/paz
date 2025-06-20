import functools
import time as pytime
import jax


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
