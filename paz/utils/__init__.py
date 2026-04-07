import os
import shutil
import functools
import hashlib
from pathlib import Path

import jax
import jax.numpy as jp


# Cache configuration
CACHE_PATH = Path("/tmp/jax_aot_cache")


def _extract_shape(value):
    """Extract shapes from arbitrary pytrees."""

    def get_shape(leaf):
        return leaf.shape if hasattr(leaf, "shape") else leaf

    return jax.tree.map(get_shape, value)


def _cache_key(*args):
    """Build cache key from argument shapes and values."""
    shapes = tuple(_extract_shape(arg) for arg in args)
    digest = hashlib.sha256(str(shapes).encode()).hexdigest()
    return digest[:16]


def _load_cached(path):
    """Load exported function from disk."""
    with open(path, "rb") as filedata:
        return jax.export.deserialize(filedata.read()).call


def _save_cached(exported, path):
    """Save exported function to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as filedata:
        filedata.write(exported.serialize())


def _extract_non_static_args(args, static_argnums):
    """Extract non-static arguments for calling exported functions."""
    if static_argnums:
        return tuple(
            arg for i, arg in enumerate(args) if i not in static_argnums
        )
    return args


def clear_cache(cache_dir=None):
    """Delete all cached functions."""
    cache_dir = CACHE_PATH if cache_dir is None else Path(cache_dir)
    if cache_dir.exists():
        for cached_file in cache_dir.glob("*.bin"):
            cached_file.unlink()


def cache(func=None, static_argnums=(), cache_dir=None):
    """
    Caches a pre-jitted function to disk.

    Use this when you have already jitted your function and want to add
    persistent disk caching. The function must already be jitted with jax.jit.

    Usage:
        @jax.jit
        @cache
        def compute(x, y):
            return x + y

        @jax.jit(static_argnums=(1,))
        @cache(static_argnums=(1,))
        def process(data, config):
            return transform(data, config)
    """
    cache_dir = CACHE_PATH if cache_dir is None else Path(cache_dir)

    def decorator(fn):
        func_name = fn.__name__
        module_name = fn.__module__.replace(".", "_")

        def wrapper(*args):
            cache_key = _cache_key(*args)
            filename = f"{module_name}_{func_name}_{cache_key}.bin"
            cache_path = cache_dir / filename
            non_static_args = _extract_non_static_args(args, static_argnums)

            if cache_path.exists():
                cached_fn = _load_cached(cache_path)
                return cached_fn(*non_static_args)

            exported = jax.export.export(fn)(*args)
            _save_cached(exported, cache_path)
            return exported.call(*non_static_args)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def jit_and_cache(func=None, static_argnums=(), cache_dir=None, **jit_kwargs):
    """
    JIT compiles and caches a function to disk.

    Combines jax.jit with persistent disk caching for faster subsequent loads.

    Usage:
        @jit_and_cache
        def compute(x, y):
            return x + y

        @jit_and_cache(static_argnums=(1,))
        def process(data, config):
            return transform(data, config)
    """
    cache_dir = CACHE_PATH if cache_dir is None else Path(cache_dir)

    def decorator(fn):
        jitted = jax.jit(fn, static_argnums=static_argnums, **jit_kwargs)
        func_name = fn.__name__
        module_name = fn.__module__.replace(".", "_")

        def wrapper(*args):
            cache_key = _cache_key(*args)
            filename = f"{module_name}_{func_name}_{cache_key}.bin"
            cache_path = cache_dir / filename
            non_static_args = _extract_non_static_args(args, static_argnums)

            if cache_path.exists():
                cached_fn = _load_cached(cache_path)
                return cached_fn(*non_static_args)

            exported = jax.export.export(jitted)(*args)
            _save_cached(exported, cache_path)
            return exported.call(*non_static_args)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


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


def __getattr__(name):
    if name == "time":
        from paz.utils.timing import time

        return time
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
