import os
import sys
import pickle
import functools
import subprocess
from collections import namedtuple

import jax
import jax.numpy as jp
import numpy as np
import pytest

import paz
from paz.utils import caching

Pair = namedtuple("Pair", ["first", "second"])
traces = []


def scale_pair(pair, factor):
    traces.append("scale_pair")
    return Pair(factor * pair.first, factor * pair.second)


def repeat(x, num, scale=1.0):
    traces.append("repeat")
    return jp.tile(x, num) * scale


def add_all(x, *rest, flag=None):
    return x + sum(rest)


def sum_squares(x):
    return jp.sum(x**2)


def build_scale(factor):
    def scale(x):
        return factor * x

    return scale


def wrap_in_local_pair(x):
    LocalPair = namedtuple("LocalPair", ["value"])
    return LocalPair(x)


def build_pair(size, dtype=jp.float32):
    return Pair(jp.ones(size, dtype), jp.array([0.0, 1.0], dtype))


def list_programs(directory):
    return sorted(directory.glob("*.program"))


def restart():
    jax.clear_caches()


@pytest.fixture(autouse=True)
def clear_traces():
    traces.clear()


def test_cache_returns_namedtuple(tmp_path):
    result = paz.cache(scale_pair, tmp_path)(build_pair(3), 2.0)
    assert isinstance(result, Pair)
    assert jp.allclose(result.first, jp.full(3, 2.0))
    assert jp.allclose(result.second, jp.array([0.0, 2.0]))


def test_cache_traces_once_and_writes_one_file(tmp_path):
    cached = paz.cache(scale_pair, tmp_path)
    cached(build_pair(3), 2.0)
    cached(build_pair(3), 3.0)
    cached(build_pair(3), factor=4.0)
    assert len(traces) == 1
    assert len(list_programs(tmp_path)) == 1


def test_cache_loads_in_new_process(tmp_path):
    paz.cache(scale_pair, tmp_path)(build_pair(3), 2.0)
    restart()
    result = paz.cache(scale_pair, tmp_path)(build_pair(3), 5.0)
    assert len(traces) == 1
    assert jp.allclose(result.first, jp.full(3, 5.0))


def test_cache_separates_shapes(tmp_path):
    cached = paz.cache(scale_pair, tmp_path)
    cached(build_pair(3), 2.0)
    cached(build_pair(4), 2.0)
    assert len(list_programs(tmp_path)) == 2


def test_cache_separates_weak_types(tmp_path):
    cached = paz.cache(scale_pair, tmp_path)
    cached(build_pair(3), 2.0)
    cached(build_pair(3), jp.float32(2.0))
    assert len(list_programs(tmp_path)) == 2


def test_cache_keeps_weak_type_of_python_scalar(tmp_path):
    result = paz.cache(scale_pair, tmp_path)(build_pair(3, jp.float16), 2.0)
    assert result.first.dtype == jp.float16


def test_cache_reuses_program_for_numpy_input(tmp_path):
    cached = paz.cache(scale_pair, tmp_path)
    cached(build_pair(3), 2.0)
    pair = Pair(np.ones(3, np.float32), np.array([0.0, 1.0], np.float32))
    result = cached(pair, 2.0)
    assert len(traces) == 1
    assert jp.allclose(result.second, jp.array([0.0, 2.0]))


def test_cache_static_by_name(tmp_path):
    cached = paz.cache(repeat, tmp_path, static_argnames=["num"])
    assert cached(jp.ones(2), 3).shape == (6,)
    assert jp.allclose(cached(jp.ones(2), num=3, scale=2.0), jp.full(6, 2.0))


def test_cache_static_by_position_reuses_file(tmp_path):
    paz.cache(repeat, tmp_path, static_argnames=["num"])(jp.ones(2), 3)
    restart()
    cached = paz.cache(repeat, tmp_path, static_argnums=(1,))
    cached(jp.ones(2), 3)
    assert len(traces) == 1
    assert cached(jp.ones(2), 2).shape == (4,)
    assert len(list_programs(tmp_path)) == 2


def test_encode_static_separates_types():
    values = [1, True, 1.0, "1"]
    encoded = {caching.encode_static(value) for value in values}
    assert len(encoded) == 4


def test_encode_static_equal_namedtuples():
    encoded = caching.encode_static(Pair(1, (2, "a")))
    assert encoded == caching.encode_static(Pair(1, (2, "a")))


@pytest.mark.parametrize("value", [[1], np.int64(1), sum_squares])
def test_encode_static_refuses_value(value):
    with pytest.raises(TypeError):
        caching.encode_static(value)


def test_cache_wraps_jitted_function(tmp_path):
    cached = paz.cache(jax.jit(scale_pair), tmp_path)
    cached(build_pair(3), 2.0)
    cached(build_pair(3), 3.0)
    assert len(traces) == 1


def test_cache_wraps_jitted_function_with_static(tmp_path):
    jitted = jax.jit(repeat, static_argnames=["num"])
    cached = paz.cache(jitted, tmp_path, static_argnames=["num"])
    assert cached(jp.ones(2), 3).shape == (6,)
    assert cached(jp.zeros(2), 3).shape == (6,)
    assert len(traces) == 1


def test_cache_varargs_and_keyword_only(tmp_path):
    cached = paz.cache(add_all, tmp_path)
    result = cached(jp.ones(2), 1.0, jp.ones(2), flag=None)
    assert jp.allclose(result, jp.full(2, 3.0))


def test_cache_refuses_lambda(tmp_path):
    with pytest.raises(ValueError):
        paz.cache(lambda x: x, tmp_path)


def test_cache_refuses_partial(tmp_path):
    with pytest.raises(TypeError):
        paz.cache(functools.partial(scale_pair, factor=2.0), tmp_path)


def test_cache_refuses_closure_without_key(tmp_path):
    with pytest.raises(ValueError, match="factor"):
        paz.cache(build_scale(2.0), tmp_path)


def test_cache_closures_with_keys(tmp_path):
    double = paz.cache(build_scale(2.0), tmp_path, key="factor=2")
    triple = paz.cache(build_scale(3.0), tmp_path, key="factor=3")
    assert jp.allclose(double(jp.ones(1)), 2.0)
    assert jp.allclose(triple(jp.ones(1)), 3.0)
    assert len(list_programs(tmp_path)) == 2


SCRIPT = """import sys
import jax.numpy as jp
import paz

traces = []


def double(x):
    traces.append(1)
    return 2 * x


paz.cache(double, sys.argv[1])(jp.ones(2))
print(len(traces))
"""


SCALE_SCRIPT = """import sys
import jax.numpy as jp
import paz


def scale(x):
    return {factor} * x


print(paz.cache(scale, sys.argv[1])(jp.ones(2)))
"""


def run_script(script, cache_dir):
    return int(read_script_output(script, cache_dir).split()[-1])


def read_script_output(script, cache_dir):
    environment = {**os.environ, "JAX_PLATFORMS": "cpu"}
    command = [sys.executable, str(script), str(cache_dir)]
    run = subprocess.run(command, env=environment, capture_output=True)
    assert run.returncode == 0, run.stderr.decode()
    return run.stdout.decode().strip()


def test_cache_source_edit_invalidates(tmp_path):
    script = tmp_path / "script" / "double.py"
    script.parent.mkdir()
    script.write_text(SCRIPT)
    cache_dir = tmp_path / "programs"
    counts = [run_script(script, cache_dir), run_script(script, cache_dir)]
    script.write_text(SCRIPT + "# edited\n")
    counts.append(run_script(script, cache_dir))
    assert counts == [1, 0, 1]
    assert len(list_programs(cache_dir)) == 2


def test_cache_separates_scripts_in_one_directory(tmp_path):
    cache_dir = tmp_path / "programs"
    double = tmp_path / "double.py"
    triple = tmp_path / "triple.py"
    double.write_text(SCALE_SCRIPT.format(factor=2))
    triple.write_text(SCALE_SCRIPT.format(factor=3))
    assert read_script_output(double, cache_dir) == "[2. 2.]"
    assert read_script_output(triple, cache_dir) == "[3. 3.]"


def test_build_name_depends_on_x64():
    leaves, tree = jax.tree.flatten({"x": jp.ones(2)})
    memo = (tree, tuple(jax.typeof(leaf) for leaf in leaves), ())
    name = caching.build_name(scale_pair, memo, "")
    try:
        jax.config.update("jax_enable_x64", True)
        name_x64 = caching.build_name(scale_pair, memo, "")
    finally:
        jax.config.update("jax_enable_x64", False)
    assert name != name_x64


def test_cache_inside_jit(tmp_path):
    cached = paz.cache(scale_pair, tmp_path)
    total = jax.jit(lambda x: cached(Pair(x, x), 2.0).first.sum())(jp.ones(3))
    assert jp.allclose(total, 6.0)


def test_cache_refuses_vmap(tmp_path):
    cached = paz.cache(sum_squares, tmp_path)
    with pytest.raises(NotImplementedError):
        jax.vmap(cached)(jp.ones((2, 3)))


def test_cache_refuses_grad(tmp_path):
    cached = paz.cache(sum_squares, tmp_path)
    with pytest.raises(ValueError):
        jax.grad(cached)(jp.ones(3))


def test_cache_raises_on_corrupt_file(tmp_path):
    paz.cache(sum_squares, tmp_path)(jp.ones(3))
    (path,) = list_programs(tmp_path)
    path.write_bytes(b"junk")
    restart()
    with pytest.raises(RuntimeError, match=str(path)):
        paz.cache(sum_squares, tmp_path)(jp.ones(3))
    assert path.read_bytes() == b"junk"


def test_cache_leaves_no_partial_file(tmp_path):
    paz.cache(sum_squares, tmp_path)(jp.ones(3))
    assert list(tmp_path.glob("*.partial")) == []


def test_cache_unpicklable_output_writes_nothing(tmp_path):
    with pytest.raises(pickle.PicklingError):
        paz.cache(wrap_in_local_pair, tmp_path)(jp.ones(3))
    assert list(tmp_path.iterdir()) == []


def test_cache_read_only_directory(tmp_path):
    paz.cache(sum_squares, tmp_path)(jp.ones(3))
    restart()
    tmp_path.chmod(0o555)
    try:
        cached = paz.cache(sum_squares, tmp_path)
        assert jp.allclose(cached(jp.ones(3)), 3.0)
        with pytest.raises(PermissionError):
            cached(jp.ones(4))
    finally:
        tmp_path.chmod(0o755)


def write_programs(directory, sizes):
    paths = []
    for index, size in enumerate(sizes):
        path = directory / f"program_{index}.program"
        path.write_bytes(bytes(size))
        os.utime(path, (index, index))
        paths.append(path)
    return paths


def test_evict_removes_oldest(tmp_path):
    oldest, middle, newest = write_programs(tmp_path, [10, 20, 30])
    caching.evict(newest, 30, 50)
    assert list_programs(tmp_path) == [middle, newest]


def test_evict_keeps_newest_over_budget(tmp_path):
    newest = write_programs(tmp_path, [10, 20, 30])[-1]
    caching.evict(newest, 30, 5)
    assert list_programs(tmp_path) == [newest]


def test_cache_hit_refreshes_mtime(tmp_path):
    paz.cache(sum_squares, tmp_path)(jp.ones(3))
    (path,) = list_programs(tmp_path)
    os.utime(path, (0, 0))
    restart()
    paz.cache(sum_squares, tmp_path)(jp.ones(3))
    assert path.stat().st_mtime > 0


def test_clear_cache_removes_programs_only(tmp_path):
    names = ["a.program", "b.partial", "notes.txt"]
    for name in names:
        (tmp_path / name).write_bytes(b"")
    paz.clear_cache(tmp_path)
    assert [path.name for path in tmp_path.iterdir()] == ["notes.txt"]


def test_clear_cache_missing_directory(tmp_path):
    paz.clear_cache(tmp_path / "missing")


def test_cache_wrapping_touches_no_disk(tmp_path):
    cache_dir = tmp_path / "programs"
    cached = paz.cache(sum_squares, cache_dir)
    assert not cache_dir.exists()
    cached(jp.ones(3))
    assert cache_dir.exists()
