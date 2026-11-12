import os
import re
import sys
import pickle
import inspect
import hashlib
import tempfile
import functools
import contextlib
import importlib.metadata
from pathlib import Path

import jax

CACHE_DIR = "~/.cache/paz/programs"


def cache(function, cache_dir=CACHE_DIR, static_argnums=(), static_argnames=(), key=""):  # fmt: skip
    """Jits `function` and keeps its exported program on disk.

    The first call for an argument signature loads the program from
    `cache_dir`, or traces `function` and writes the program there. A
    program is reused only when the source of the function's package (or
    script directory) and of paz, the Python, jax, jaxlib, optax and keras
    versions, the platform, the x64 and PRNG flags, the argument shapes,
    dtypes and pytree structure, the static values and `key` all match.

    Args:
        function: named function, optionally already wrapped by `jax.jit`.
        cache_dir: directory of the programs.
        static_argnums: positions of arguments fixed while tracing.
        static_argnames: names of arguments fixed while tracing.
        key: describes what the cache cannot see, such as weights loaded
            while tracing. Required when `function` is a closure.

    Returns:
        Function with the call signature and results of `jax.jit(function)`.

    Static values must be None, bool, int, float, str, bytes or tuples of
    them. `jax.vmap` and `jax.grad` cannot be applied to the result; apply
    them inside `function`. Host callbacks (`jax.debug.print`) cannot be
    exported. Files are read with pickle: only use a trusted `cache_dir`.
    """
    plain = unwrap_jit(function)
    validate(plain, key)
    signature = inspect.signature(plain)
    static_args = (signature, static_argnums, static_argnames)
    static_names = select_static_names(*static_args)
    programs = {}

    @functools.wraps(plain)
    def call(*args, **kwargs):
        split_args = (signature, static_names, args, kwargs)
        statics, dynamics = split_arguments(*split_args)
        leaves, tree = jax.tree.flatten(dynamics)
        types = tuple(jax.typeof(leaf) for leaf in leaves)
        memo = (tree, types, encode_statics(statics))
        if memo not in programs:
            flat_args = (function, signature, tree, statics)
            flat_function = build_flat_function(*flat_args)
            path = Path(cache_dir).expanduser() / build_name(plain, memo, key)
            programs[memo] = load(path, flat_function, leaves)
        program, output_tree = programs[memo]
        return jax.tree.unflatten(output_tree, program(*leaves))

    return call


def unwrap_jit(function):
    has_code = hasattr(function, "__code__")
    is_jitted = hasattr(function, "__wrapped__") and not has_code
    return function.__wrapped__ if is_jitted else function


def validate(function, key):
    if not inspect.isfunction(function):
        message = f"cache needs a named function, got {function!r}"
        raise TypeError(f"{message}; wrap it in a def")
    if function.__name__ == "<lambda>":
        raise ValueError("cache cannot tell lambdas apart; use a named def")
    if function.__closure__ and not key:
        names = ", ".join(function.__code__.co_freevars)
        message = f"{function.__qualname__} captures {names}"
        advice = "pass them as arguments or describe them in key"
        raise ValueError(f"{message}; {advice}")


def select_static_names(signature, static_argnums, static_argnames):
    names = list(signature.parameters)
    positional = [names[index] for index in static_argnums]
    return frozenset(positional) | frozenset(static_argnames)


def split_arguments(signature, static_names, args, kwargs):
    arguments = signature.bind(*args, **kwargs).arguments
    statics, dynamics = {}, {}
    for name, value in arguments.items():
        if name in static_names:
            statics[name] = value
        else:
            dynamics[name] = value
    return statics, dynamics


def encode_statics(statics):
    encoded = [(name, encode_static(value)) for name, value in statics.items()]
    return tuple(sorted(encoded))


def encode_static(value):
    # exact types: numpy scalars subclass float but print without their dtype
    if type(value) in (type(None), bool, int, float, str, bytes):
        encoded = f"{type(value).__name__}:{value!r}"
    elif isinstance(value, tuple):
        items = ",".join(encode_static(item) for item in value)
        encoded = f"{type(value).__qualname__}({items})"
    else:
        allowed = "None, bool, int, float, str, bytes or tuples of them"
        raise TypeError(f"static value {value!r} is not one of {allowed}")
    return encoded


def build_flat_function(function, signature, tree, statics):
    def flat_function(*leaves):
        arguments = {**jax.tree.unflatten(tree, leaves), **statics}
        names = [name for name in signature.parameters if name in arguments]
        bound = signature.bind_partial()
        bound.arguments.update((name, arguments[name]) for name in names)
        return function(*bound.args, **bound.kwargs)

    return flat_function


def build_name(function, memo, key):
    tree, types, encoded = memo
    identity = [function.__module__, str(function.__code__.co_firstlineno)]
    parts = [hash_code(function.__module__), *identity, key, str(tree)]
    parts.extend(str(leaf_type) for leaf_type in types)
    parts.extend([repr(encoded), jax.export.default_export_platform()])
    parts.extend(describe_flags())
    digest = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:24]
    name = f"{function.__module__}.{function.__qualname__}"
    return f"{re.sub(r'[^A-Za-z0-9_.]', '_', name)}-{digest}.program"


@functools.lru_cache
def hash_code(module_name):
    digest = hashlib.sha256(describe_environment().encode())
    for root, paths in [find_sources(module_name), find_sources("paz")]:
        for path in paths:
            digest.update(f"\0{path.relative_to(root)}\0".encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@functools.lru_cache
def describe_environment():
    names = ("jax", "jaxlib", "optax", "keras")
    versions = [f"{name}=={importlib.metadata.version(name)}" for name in names]
    return "\n".join([sys.version, *versions])


def find_sources(module_name):
    top = sys.modules[module_name.partition(".")[0]]
    if hasattr(top, "__path__"):
        root = Path(list(top.__path__)[0])
        paths = sorted(root.rglob("*.py"))
    elif getattr(top, "__file__", None):
        root = Path(top.__file__).parent
        paths = sorted(root.glob("*.py"))
    else:
        raise ValueError(f"{module_name} was defined interactively, no source")
    return root, paths


def describe_flags():
    flags = ("jax_enable_x64", "jax_default_prng_impl")
    flags = flags + ("jax_threefry_partitionable", "jax_numpy_dtype_promotion")
    flags = flags + ("jax_default_matmul_precision",)
    return [f"{flag}={getattr(jax.config, flag)}" for flag in flags]


def load(path, flat_function, leaves):
    try:
        content = read(path)
    except FileNotFoundError:
        content = trace(flat_function, leaves)
        write(path, content)
    return unpack(path, content)


def read(path):
    content = path.read_bytes()
    # a read-only cache directory still serves hits
    with contextlib.suppress(OSError):
        os.utime(path)
    return content


def trace(flat_function, leaves):
    output_trees = []

    def flat_outputs(*leaves):
        outputs, output_tree = jax.tree.flatten(flat_function(*leaves))
        output_trees.append(output_tree)
        return outputs

    exported = jax.export.export(jax.jit(flat_outputs))(*leaves)
    return pickle.dumps((exported.serialize(), output_trees[0]))


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".partial")
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    evict(path, 2 * 1024**3)


def evict(kept_path, max_bytes):
    stats = stat_programs(kept_path.parent, kept_path)
    stats.sort(key=lambda item: item[1].st_mtime, reverse=True)
    total = kept_path.stat().st_size
    for path, stat in stats:
        total = total + stat.st_size
        if total > max_bytes:
            path.unlink(missing_ok=True)


def stat_programs(directory, kept_path):
    stats = []
    for path in directory.glob("*.program"):
        # another process may evict the same file meanwhile
        with contextlib.suppress(FileNotFoundError):
            stats.append((path, path.stat()))
    return [item for item in stats if item[0] != kept_path]


def unpack(path, content):
    try:
        serialized, output_tree = pickle.loads(content)
        program = jax.export.deserialize(serialized)
    except Exception as error:
        message = f"cannot load {path}; delete it to trace again"
        raise RuntimeError(message) from error
    return jax.jit(program.call), output_tree


def clear_cache(cache_dir=CACHE_DIR):
    directory = Path(cache_dir).expanduser()
    for pattern in ("*.program", "*.partial"):
        for path in directory.glob(pattern):
            path.unlink(missing_ok=True)
