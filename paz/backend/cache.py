"""
JIT compilation with persistent disk caching.

Usage:
    import cache

    @cache.jit
    def compute(x, y):
        return x + y

    @cache.jit(static_argnums=(1,))
    def process(data, config):
        return transform(data, config)

    @cache.jit(static_argnums=(2,), cache_dir="/custom/path")
    def advanced(x, y, mode):
        return x + y if mode == "forward" else x - y

    # Clear all cached functions
    cache.clear()

Edge cases and limitations:
    - Cache key is based on module name, function name, input shapes,
      and static argument values
    - Keyword arguments are not supported (JAX export limitation)
    - Cache invalidation is manual (use cache.clear() or delete .bin files)
"""

import hashlib
from pathlib import Path

import jax

PATH = Path("/tmp/jax_aot_cache")


def _extract_shape(value):
    """
    Extract shapes from arbitrary pytrees.

    For arrays: returns shape (e.g., (3, 4))
    For non-arrays (static args): returns value itself (e.g., 5, "mode")
    """

    def get_shape(leaf):
        return leaf.shape if hasattr(leaf, "shape") else leaf

    return jax.tree.map(get_shape, value)


def _key(*args):
    """
    Build cache key from argument shapes and values.

    Arrays contribute their shapes: jp.array([1,2,3]) -> (3,)
    Static args contribute their values: 5 -> 5, "mode" -> "mode"
    This ensures different static arg values get different cache entries.
    """
    shapes = tuple(_extract_shape(arg) for arg in args)
    digest = hashlib.sha256(str(shapes).encode()).hexdigest()
    return digest[:16]  # reasonable file length with neglibale collisions


def _load(path):
    """Load exported function from disk."""
    with open(path, "rb") as filedata:
        return jax.export.deserialize(filedata.read()).call


def _save(exported, path):
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


def clear(cache_dir=None):
    """
    Delete all cached functions.

    Args:
        cache_dir: Directory to clear (default: cache.PATH)
    """
    cache_dir = PATH if cache_dir is None else Path(cache_dir)
    if cache_dir.exists():
        for cached_file in cache_dir.glob("*.bin"):
            cached_file.unlink()


def jit(func=None, static_argnums=(), cache_dir=None, **jit_kwargs):
    """Loads cache or compiles and caches."""
    cache_dir = PATH if cache_dir is None else Path(cache_dir)

    def decorator(fn):
        jitted = jax.jit(fn, static_argnums=static_argnums, **jit_kwargs)
        func_name = fn.__name__
        module_name = fn.__module__.replace(".", "_")

        def wrapper(*args):
            cache_key = _key(*args)
            filename = f"{module_name}_{func_name}_{cache_key}.bin"
            cache_path = cache_dir / filename
            non_static_args = _extract_non_static_args(args, static_argnums)

            if cache_path.exists():
                cached_fn = _load(cache_path)
                return cached_fn(*non_static_args)

            exported = jax.export.export(jitted)(*args)
            _save(exported, cache_path)
            return exported.call(*non_static_args)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)
