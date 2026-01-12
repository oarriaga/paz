"""
Test suite for cache and jit_and_cache functions in utils module.

Tests cover:
- Shape extraction from various pytrees
- Cache key generation
- Function serialization (save/load)
- Cache clearing
- cache decorator (for pre-jitted functions)
- jit_and_cache decorator (combined jit + cache)
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jp
import pytest

import paz


# Fixtures


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_array():
    """Sample JAX array for testing."""
    return jp.array([1.0, 2.0, 3.0])


@pytest.fixture
def sample_2d_array():
    """Sample 2D JAX array for testing."""
    return jp.array([[1.0, 2.0], [3.0, 4.0]])


# Tests for _extract_shape


def test_extract_shape_array_returns_shape(sample_array):
    """Array input should return its shape."""
    from paz.utils import _extract_shape

    result = _extract_shape(sample_array)
    assert result == (3,)


def test_extract_shape_2d_array_returns_shape(sample_2d_array):
    """2D array input should return its shape."""
    from paz.utils import _extract_shape

    result = _extract_shape(sample_2d_array)
    assert result == (2, 2)


def test_extract_shape_scalar_int_returns_value():
    """Integer scalar should return the value itself."""
    from paz.utils import _extract_shape

    result = _extract_shape(5)
    assert result == 5


def test_extract_shape_scalar_float_returns_value():
    """Float scalar should return the value itself."""
    from paz.utils import _extract_shape

    result = _extract_shape(3.14)
    assert result == 3.14


def test_extract_shape_string_returns_value():
    """String should return the value itself."""
    from paz.utils import _extract_shape

    result = _extract_shape("mode")
    assert result == "mode"


def test_extract_shape_list_of_arrays(sample_array):
    """List of arrays should return list of shapes."""
    from paz.utils import _extract_shape

    arrays = [sample_array, sample_array * 2]
    result = _extract_shape(arrays)
    assert result == [(3,), (3,)]


def test_extract_shape_tuple_of_arrays(sample_array, sample_2d_array):
    """Tuple of arrays should return tuple of shapes."""
    from paz.utils import _extract_shape

    arrays = (sample_array, sample_2d_array)
    result = _extract_shape(arrays)
    assert result == ((3,), (2, 2))


def test_extract_shape_dict_of_arrays(sample_array, sample_2d_array):
    """Dict of arrays should return dict of shapes."""
    from paz.utils import _extract_shape

    arrays = {"x": sample_array, "y": sample_2d_array}
    result = _extract_shape(arrays)
    assert result == {"x": (3,), "y": (2, 2)}


def test_extract_shape_nested_pytree(sample_array):
    """Nested pytree should return nested structure of shapes."""
    from paz.utils import _extract_shape

    nested = {"a": [sample_array, sample_array], "b": {"c": sample_array}}
    result = _extract_shape(nested)
    assert result == {"a": [(3,), (3,)], "b": {"c": (3,)}}


def test_extract_shape_mixed_pytree(sample_array):
    """Pytree with arrays and scalars should preserve both."""
    from paz.utils import _extract_shape

    mixed = {"array": sample_array, "scalar": 42, "string": "test"}
    result = _extract_shape(mixed)
    assert result == {"array": (3,), "scalar": 42, "string": "test"}


# Tests for _cache_key


def test_cache_key_same_shapes_produce_same_key(sample_array):
    """Same array shapes should produce identical keys."""
    from paz.utils import _cache_key

    arr1 = jp.array([1.0, 2.0, 3.0])
    arr2 = jp.array([4.0, 5.0, 6.0])
    key1 = _cache_key(arr1)
    key2 = _cache_key(arr2)
    assert key1 == key2


def test_cache_key_different_shapes_produce_different_keys():
    """Different array shapes should produce different keys."""
    from paz.utils import _cache_key

    arr1 = jp.array([1.0, 2.0, 3.0])
    arr2 = jp.array([1.0, 2.0])
    key1 = _cache_key(arr1)
    key2 = _cache_key(arr2)
    assert key1 != key2


def test_cache_key_same_static_values_produce_same_key():
    """Same static argument values should produce same key."""
    from paz.utils import _cache_key

    key1 = _cache_key(5)
    key2 = _cache_key(5)
    assert key1 == key2


def test_cache_key_different_static_values_produce_different_keys():
    """Different static argument values should produce different keys."""
    from paz.utils import _cache_key

    key1 = _cache_key(5)
    key2 = _cache_key(10)
    assert key1 != key2


def test_cache_key_length_is_16_characters():
    """Cache key should be 16 characters long."""
    from paz.utils import _cache_key

    key = _cache_key(jp.array([1, 2, 3]))
    assert len(key) == 16


def test_cache_key_is_deterministic(sample_array):
    """Same inputs should always produce same key."""
    from paz.utils import _cache_key

    key1 = _cache_key(sample_array, 5, "mode")
    key2 = _cache_key(sample_array, 5, "mode")
    assert key1 == key2


def test_cache_key_multiple_args_order_matters(sample_array):
    """Argument order should affect cache key."""
    from paz.utils import _cache_key

    key1 = _cache_key(sample_array, 5)
    key2 = _cache_key(5, sample_array)
    assert key1 != key2


def test_cache_key_pytree_structure_matters(sample_array):
    """Different pytree structures with same shapes should differ."""
    from paz.utils import _cache_key

    dict_input = {"a": sample_array}
    list_input = [sample_array]
    key1 = _cache_key(dict_input)
    key2 = _cache_key(list_input)
    assert key1 != key2


# Tests for _save_cached and _load_cached


def test_save_creates_file(temp_cache_dir):
    """Saving should create a .bin file."""
    from paz.utils import _save_cached

    path = temp_cache_dir / "test.bin"

    @jax.jit
    def simple_func(x):
        return x * 2

    exported = jax.export.export(simple_func)(jp.array([1.0]))
    _save_cached(exported, path)
    assert path.exists()


def test_save_creates_parent_directory(temp_cache_dir):
    """Saving should create parent directories if missing."""
    from paz.utils import _save_cached

    path = temp_cache_dir / "subdir" / "test.bin"

    @jax.jit
    def simple_func(x):
        return x * 2

    exported = jax.export.export(simple_func)(jp.array([1.0]))
    _save_cached(exported, path)
    assert path.exists()
    assert path.parent.exists()


def test_load_returns_callable(temp_cache_dir):
    """Loading should return a callable function."""
    from paz.utils import _save_cached, _load_cached

    path = temp_cache_dir / "test.bin"

    @jax.jit
    def simple_func(x):
        return x * 2

    exported = jax.export.export(simple_func)(jp.array([1.0]))
    _save_cached(exported, path)
    loaded = _load_cached(path)
    assert callable(loaded)


def test_save_and_load_preserves_function_behavior(temp_cache_dir):
    """Loaded function should behave identically to original."""
    from paz.utils import _save_cached, _load_cached

    path = temp_cache_dir / "test.bin"

    @jax.jit
    def add_one(x):
        return x + 1

    x = jp.array([1.0, 2.0, 3.0])
    exported = jax.export.export(add_one)(x)
    _save_cached(exported, path)
    loaded = _load_cached(path)
    result = loaded(x)
    expected = jp.array([2.0, 3.0, 4.0])
    assert jp.allclose(result, expected)


# Tests for clear_cache


def test_clear_cache_deletes_bin_files(temp_cache_dir):
    """clear_cache should delete all .bin files in cache directory."""
    (temp_cache_dir / "file1.bin").touch()
    (temp_cache_dir / "file2.bin").touch()
    paz.clear_cache(temp_cache_dir)
    assert not (temp_cache_dir / "file1.bin").exists()
    assert not (temp_cache_dir / "file2.bin").exists()


def test_clear_cache_preserves_non_bin_files(temp_cache_dir):
    """clear_cache should not delete non-.bin files."""
    (temp_cache_dir / "file.txt").touch()
    (temp_cache_dir / "file.bin").touch()
    paz.clear_cache(temp_cache_dir)
    assert (temp_cache_dir / "file.txt").exists()
    assert not (temp_cache_dir / "file.bin").exists()


def test_clear_cache_handles_nonexistent_directory(temp_cache_dir):
    """clear_cache should not error on nonexistent directory."""
    nonexistent = temp_cache_dir / "does_not_exist"
    paz.clear_cache(nonexistent)


def test_clear_cache_uses_default_path_when_none():
    """clear_cache should use CACHE_PATH when cache_dir is None."""
    paz.clear_cache()


def test_clear_cache_empty_directory(temp_cache_dir):
    """clear_cache should handle empty directory without error."""
    paz.clear_cache(temp_cache_dir)


# Tests for cache decorator - Basic Functionality


def test_cache_decorator_without_args(temp_cache_dir):
    """@cache should work without parentheses."""

    @paz.cache
    @jax.jit
    def add(x, y):
        return x + y

    result = add(jp.array([1.0]), jp.array([2.0]))
    expected = jp.array([3.0])
    assert jp.allclose(result, expected)


def test_cache_decorator_with_empty_args(temp_cache_dir):
    """@cache() should work with empty parentheses."""

    @paz.cache(cache_dir=temp_cache_dir)
    @jax.jit
    def add(x, y):
        return x + y

    result = add(jp.array([1.0]), jp.array([2.0]))
    expected = jp.array([3.0])
    assert jp.allclose(result, expected)


def test_cache_returns_correct_result(temp_cache_dir):
    """Cached function should return correct result."""

    @paz.cache(cache_dir=temp_cache_dir)
    @jax.jit
    def multiply(x, scale):
        return x * scale

    result = multiply(jp.array([1.0, 2.0, 3.0]), 2.0)
    expected = jp.array([2.0, 4.0, 6.0])
    assert jp.allclose(result, expected)


# Tests for cache decorator - Caching Behavior


def test_cache_creates_cache_file_on_first_call(temp_cache_dir):
    """First call should create a cache file."""

    @paz.cache(cache_dir=temp_cache_dir)
    @jax.jit
    def add_one(x):
        return x + 1

    add_one(jp.array([1.0, 2.0, 3.0]))
    cache_files = list(temp_cache_dir.glob("*.bin"))
    assert len(cache_files) == 1


def test_cache_loads_from_cache_on_second_call(temp_cache_dir):
    """Second call with same shapes should load from cache."""

    @paz.cache(cache_dir=temp_cache_dir)
    @jax.jit
    def add_one(x):
        return x + 1

    add_one(jp.array([1.0, 2.0, 3.0]))
    cache_files_before = list(temp_cache_dir.glob("*.bin"))
    add_one(jp.array([4.0, 5.0, 6.0]))
    cache_files_after = list(temp_cache_dir.glob("*.bin"))
    assert len(cache_files_before) == len(cache_files_after)


def test_cache_different_shapes_create_different_cache(temp_cache_dir):
    """Different input shapes should create separate cache entries."""

    @paz.cache(cache_dir=temp_cache_dir)
    @jax.jit
    def add_one(x):
        return x + 1

    add_one(jp.array([1.0, 2.0, 3.0]))
    add_one(jp.array([1.0, 2.0]))
    cache_files = list(temp_cache_dir.glob("*.bin"))
    assert len(cache_files) == 2


def test_cache_with_static_argnums(temp_cache_dir):
    """static_argnums should mark arguments as static."""

    @paz.cache(static_argnums=(1,), cache_dir=temp_cache_dir)
    @jax.jit(static_argnums=(1,))
    def scale(x, factor):
        return x * factor

    result = scale(jp.array([1.0, 2.0]), 3.0)
    expected = jp.array([3.0, 6.0])
    assert jp.allclose(result, expected)


def test_cache_different_static_values_create_different_cache(temp_cache_dir):
    """Different static argument values should create separate caches."""

    @paz.cache(static_argnums=(1,), cache_dir=temp_cache_dir)
    @jax.jit(static_argnums=(1,))
    def scale(x, factor):
        return x * factor

    scale(jp.array([1.0, 2.0]), 2.0)
    scale(jp.array([1.0, 2.0]), 3.0)
    cache_files = list(temp_cache_dir.glob("*.bin"))
    assert len(cache_files) == 2


# Tests for jit_and_cache decorator - Basic Functionality


def test_jit_and_cache_decorator_without_args():
    """@jit_and_cache should work without parentheses."""

    @paz.jit_and_cache
    def add(x, y):
        return x + y

    result = add(jp.array([1.0]), jp.array([2.0]))
    expected = jp.array([3.0])
    assert jp.allclose(result, expected)


def test_jit_and_cache_decorator_with_empty_args(temp_cache_dir):
    """@jit_and_cache() should work with empty parentheses."""

    @paz.jit_and_cache(cache_dir=temp_cache_dir)
    def add(x, y):
        return x + y

    result = add(jp.array([1.0]), jp.array([2.0]))
    expected = jp.array([3.0])
    assert jp.allclose(result, expected)


def test_jit_and_cache_returns_correct_result(temp_cache_dir):
    """jit_and_cache function should return correct result."""

    @paz.jit_and_cache(cache_dir=temp_cache_dir)
    def multiply(x, scale):
        return x * scale

    result = multiply(jp.array([1.0, 2.0, 3.0]), 2.0)
    expected = jp.array([2.0, 4.0, 6.0])
    assert jp.allclose(result, expected)


# Tests for jit_and_cache decorator - Caching Behavior


def test_jit_and_cache_creates_cache_file_on_first_call(temp_cache_dir):
    """First call should create a cache file."""

    @paz.jit_and_cache(cache_dir=temp_cache_dir)
    def add_one(x):
        return x + 1

    add_one(jp.array([1.0, 2.0, 3.0]))
    cache_files = list(temp_cache_dir.glob("*.bin"))
    assert len(cache_files) == 1


def test_jit_and_cache_loads_from_cache_on_second_call(temp_cache_dir):
    """Second call with same shapes should load from cache."""

    @paz.jit_and_cache(cache_dir=temp_cache_dir)
    def add_one(x):
        return x + 1

    add_one(jp.array([1.0, 2.0, 3.0]))
    cache_files_before = list(temp_cache_dir.glob("*.bin"))
    add_one(jp.array([4.0, 5.0, 6.0]))
    cache_files_after = list(temp_cache_dir.glob("*.bin"))
    assert len(cache_files_before) == len(cache_files_after)


def test_jit_and_cache_different_shapes_create_different_cache(temp_cache_dir):
    """Different input shapes should create separate cache entries."""

    @paz.jit_and_cache(cache_dir=temp_cache_dir)
    def add_one(x):
        return x + 1

    add_one(jp.array([1.0, 2.0, 3.0]))
    add_one(jp.array([1.0, 2.0]))
    cache_files = list(temp_cache_dir.glob("*.bin"))
    assert len(cache_files) == 2


def test_jit_and_cache_filename_includes_module_name(temp_cache_dir):
    """Cache filename should include module name."""

    @paz.jit_and_cache(cache_dir=temp_cache_dir)
    def test_func(x):
        return x + 1

    test_func(jp.array([1.0]))
    cache_files = list(temp_cache_dir.glob("*.bin"))
    assert len(cache_files) == 1
    assert (
        "__main__" in cache_files[0].name or "utils_test" in cache_files[0].name
    )


def test_jit_and_cache_filename_includes_function_name(temp_cache_dir):
    """Cache filename should include function name."""

    @paz.jit_and_cache(cache_dir=temp_cache_dir)
    def my_function(x):
        return x + 1

    my_function(jp.array([1.0]))
    cache_files = list(temp_cache_dir.glob("*.bin"))
    assert len(cache_files) == 1
    assert "my_function" in cache_files[0].name


# Tests for jit_and_cache - static_argnums


def test_jit_and_cache_with_static_argnums(temp_cache_dir):
    """static_argnums should mark arguments as static."""

    @paz.jit_and_cache(static_argnums=(1,), cache_dir=temp_cache_dir)
    def scale(x, factor):
        return x * factor

    result = scale(jp.array([1.0, 2.0]), 3.0)
    expected = jp.array([3.0, 6.0])
    assert jp.allclose(result, expected)


def test_jit_and_cache_different_static_values_create_different_cache(
    temp_cache_dir,
):
    """Different static argument values should create separate caches."""

    @paz.jit_and_cache(static_argnums=(1,), cache_dir=temp_cache_dir)
    def scale(x, factor):
        return x * factor

    scale(jp.array([1.0, 2.0]), 2.0)
    scale(jp.array([1.0, 2.0]), 3.0)
    cache_files = list(temp_cache_dir.glob("*.bin"))
    assert len(cache_files) == 2


def test_jit_and_cache_same_static_values_reuse_cache(temp_cache_dir):
    """Same static argument values should reuse cache."""

    @paz.jit_and_cache(static_argnums=(1,), cache_dir=temp_cache_dir)
    def scale(x, factor):
        return x * factor

    scale(jp.array([1.0, 2.0]), 2.0)
    cache_files_before = list(temp_cache_dir.glob("*.bin"))
    scale(jp.array([3.0, 4.0]), 2.0)
    cache_files_after = list(temp_cache_dir.glob("*.bin"))
    assert len(cache_files_before) == len(cache_files_after) == 1


# Tests for jit_and_cache - Additional jit_kwargs


def test_jit_and_cache_passes_kwargs_to_jax_jit(temp_cache_dir):
    """Additional kwargs should be passed to jax.jit."""

    @paz.jit_and_cache(donate_argnums=(0,), cache_dir=temp_cache_dir)
    def add_one(x):
        return x + 1

    result = add_one(jp.array([1.0, 2.0, 3.0]))
    expected = jp.array([2.0, 3.0, 4.0])
    assert jp.allclose(result, expected)


# Tests for jit_and_cache - Edge Cases


def test_jit_and_cache_multiple_arrays_input(temp_cache_dir):
    """Function with multiple array inputs should work."""

    @paz.jit_and_cache(cache_dir=temp_cache_dir)
    def add_arrays(x, y, z):
        return x + y + z

    result = add_arrays(jp.array([1.0]), jp.array([2.0]), jp.array([3.0]))
    expected = jp.array([6.0])
    assert jp.allclose(result, expected)


def test_jit_and_cache_pytree_input(temp_cache_dir):
    """Function with pytree input should work."""

    @paz.jit_and_cache(cache_dir=temp_cache_dir)
    def sum_dict(data):
        return data["x"] + data["y"]

    result = sum_dict({"x": jp.array([1.0]), "y": jp.array([2.0])})
    expected = jp.array([3.0])
    assert jp.allclose(result, expected)


def test_jit_and_cache_nested_pytree_input(temp_cache_dir):
    """Function with nested pytree input should work."""

    @paz.jit_and_cache(cache_dir=temp_cache_dir)
    def process_nested(data):
        return data["outer"]["inner"] * 2

    result = process_nested({"outer": {"inner": jp.array([1.0, 2.0])}})
    expected = jp.array([2.0, 4.0])
    assert jp.allclose(result, expected)


def test_jit_and_cache_no_arguments_function(temp_cache_dir):
    """Function with no arguments should work."""

    @paz.jit_and_cache(cache_dir=temp_cache_dir)
    def constant():
        return jp.array([42.0])

    result = constant()
    expected = jp.array([42.0])
    assert jp.allclose(result, expected)


def test_jit_and_cache_returns_pytree(temp_cache_dir):
    """Function returning pytree should work."""

    @paz.jit_and_cache(cache_dir=temp_cache_dir)
    def split_result(x):
        return {"double": x * 2, "triple": x * 3}

    result = split_result(jp.array([1.0, 2.0]))
    assert jp.allclose(result["double"], jp.array([2.0, 4.0]))
    assert jp.allclose(result["triple"], jp.array([3.0, 6.0]))
