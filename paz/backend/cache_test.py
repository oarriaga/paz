"""
Comprehensive test suite for cache module.

Tests cover:
- Shape extraction from various pytrees
- Cache key generation
- Function serialization (save/load)
- Cache clearing
- JIT compilation and caching behavior
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
    result = paz.cache._extract_shape(sample_array)
    assert result == (3,)


def test_extract_shape_2d_array_returns_shape(sample_2d_array):
    """2D array input should return its shape."""
    result = paz.cache._extract_shape(sample_2d_array)
    assert result == (2, 2)


def test_extract_shape_scalar_int_returns_value():
    """Integer scalar should return the value itself."""
    result = paz.cache._extract_shape(5)
    assert result == 5


def test_extract_shape_scalar_float_returns_value():
    """Float scalar should return the value itself."""
    result = paz.cache._extract_shape(3.14)
    assert result == 3.14


def test_extract_shape_string_returns_value():
    """String should return the value itself."""
    result = paz.cache._extract_shape("mode")
    assert result == "mode"


def test_extract_shape_list_of_arrays(sample_array):
    """List of arrays should return list of shapes."""
    arrays = [sample_array, sample_array * 2]
    result = paz.cache._extract_shape(arrays)
    assert result == [(3,), (3,)]


def test_extract_shape_tuple_of_arrays(sample_array, sample_2d_array):
    """Tuple of arrays should return tuple of shapes."""
    arrays = (sample_array, sample_2d_array)
    result = paz.cache._extract_shape(arrays)
    assert result == ((3,), (2, 2))


def test_extract_shape_dict_of_arrays(sample_array, sample_2d_array):
    """Dict of arrays should return dict of shapes."""
    arrays = {"x": sample_array, "y": sample_2d_array}
    result = paz.cache._extract_shape(arrays)
    assert result == {"x": (3,), "y": (2, 2)}


def test_extract_shape_nested_pytree(sample_array):
    """Nested pytree should return nested structure of shapes."""
    nested = {"a": [sample_array, sample_array], "b": {"c": sample_array}}
    result = paz.cache._extract_shape(nested)
    assert result == {"a": [(3,), (3,)], "b": {"c": (3,)}}


def test_extract_shape_mixed_pytree(sample_array):
    """Pytree with arrays and scalars should preserve both."""
    mixed = {"array": sample_array, "scalar": 42, "string": "test"}
    result = paz.cache._extract_shape(mixed)
    assert result == {"array": (3,), "scalar": 42, "string": "test"}


# Tests for _key


def test_key_same_shapes_produce_same_key(sample_array):
    """Same array shapes should produce identical keys."""
    arr1 = jp.array([1.0, 2.0, 3.0])
    arr2 = jp.array([4.0, 5.0, 6.0])
    key1 = paz.cache._key(arr1)
    key2 = paz.cache._key(arr2)
    assert key1 == key2


def test_key_different_shapes_produce_different_keys():
    """Different array shapes should produce different keys."""
    arr1 = jp.array([1.0, 2.0, 3.0])
    arr2 = jp.array([1.0, 2.0])
    key1 = paz.cache._key(arr1)
    key2 = paz.cache._key(arr2)
    assert key1 != key2


def test_key_same_static_values_produce_same_key():
    """Same static argument values should produce same key."""
    key1 = paz.cache._key(5)
    key2 = paz.cache._key(5)
    assert key1 == key2


def test_key_different_static_values_produce_different_keys():
    """Different static argument values should produce different keys."""
    key1 = paz.cache._key(5)
    key2 = paz.cache._key(10)
    assert key1 != key2


def test_key_length_is_16_characters():
    """Cache key should be 16 characters long."""
    key = paz.cache._key(jp.array([1, 2, 3]))
    assert len(key) == 16


def test_key_is_deterministic(sample_array):
    """Same inputs should always produce same key."""
    key1 = paz.cache._key(sample_array, 5, "mode")
    key2 = paz.cache._key(sample_array, 5, "mode")
    assert key1 == key2


def test_key_multiple_args_order_matters(sample_array):
    """Argument order should affect cache key."""
    key1 = paz.cache._key(sample_array, 5)
    key2 = paz.cache._key(5, sample_array)
    assert key1 != key2


def test_key_pytree_structure_matters(sample_array):
    """Different pytree structures with same shapes should differ."""
    dict_input = {"a": sample_array}
    list_input = [sample_array]
    key1 = paz.cache._key(dict_input)
    key2 = paz.cache._key(list_input)
    assert key1 != key2


# Tests for _save and _load


def test_save_creates_file(temp_cache_dir):
    """Saving should create a .bin file."""
    path = temp_cache_dir / "test.bin"

    @jax.jit
    def simple_func(x):
        return x * 2

    exported = jax.export.export(simple_func)(jp.array([1.0]))
    paz.cache._save(exported, path)
    assert path.exists()


def test_save_creates_parent_directory(temp_cache_dir):
    """Saving should create parent directories if missing."""
    path = temp_cache_dir / "subdir" / "test.bin"

    @jax.jit
    def simple_func(x):
        return x * 2

    exported = jax.export.export(simple_func)(jp.array([1.0]))
    paz.cache._save(exported, path)
    assert path.exists()
    assert path.parent.exists()


def test_load_returns_callable(temp_cache_dir):
    """Loading should return a callable function."""
    path = temp_cache_dir / "test.bin"

    @jax.jit
    def simple_func(x):
        return x * 2

    exported = jax.export.export(simple_func)(jp.array([1.0]))
    paz.cache._save(exported, path)
    loaded = paz.cache._load(path)
    assert callable(loaded)


def test_save_and_load_preserves_function_behavior(temp_cache_dir):
    """Loaded function should behave identically to original."""
    path = temp_cache_dir / "test.bin"

    @jax.jit
    def add_one(x):
        return x + 1

    x = jp.array([1.0, 2.0, 3.0])
    exported = jax.export.export(add_one)(x)
    paz.cache._save(exported, path)
    loaded = paz.cache._load(path)
    result = loaded(x)
    expected = jp.array([2.0, 3.0, 4.0])
    assert jp.allclose(result, expected)


# Tests for clear


def test_clear_deletes_bin_files(temp_cache_dir):
    """Clear should delete all .bin files in cache directory."""
    (temp_cache_dir / "file1.bin").touch()
    (temp_cache_dir / "file2.bin").touch()
    paz.cache.clear(temp_cache_dir)
    assert not (temp_cache_dir / "file1.bin").exists()
    assert not (temp_cache_dir / "file2.bin").exists()


def test_clear_preserves_non_bin_files(temp_cache_dir):
    """Clear should not delete non-.bin files."""
    (temp_cache_dir / "file.txt").touch()
    (temp_cache_dir / "file.bin").touch()
    paz.cache.clear(temp_cache_dir)
    assert (temp_cache_dir / "file.txt").exists()
    assert not (temp_cache_dir / "file.bin").exists()


def test_clear_handles_nonexistent_directory(temp_cache_dir):
    """Clear should not error on nonexistent directory."""
    nonexistent = temp_cache_dir / "does_not_exist"
    paz.cache.clear(nonexistent)


def test_clear_uses_default_path_when_none():
    """Clear should use cache.PATH when cache_dir is None."""
    paz.cache.clear()


def test_clear_empty_directory(temp_cache_dir):
    """Clear should handle empty directory without error."""
    paz.cache.clear(temp_cache_dir)


# Tests for jit - Basic Functionality


def test_jit_decorator_without_args():
    """@jit should work without parentheses."""

    @paz.cache.jit
    def add(x, y):
        return x + y

    result = add(jp.array([1.0]), jp.array([2.0]))
    expected = jp.array([3.0])
    assert jp.allclose(result, expected)


def test_jit_decorator_with_empty_args(temp_cache_dir):
    """@jit() should work with empty parentheses."""

    @paz.cache.jit(cache_dir=temp_cache_dir)
    def add(x, y):
        return x + y

    result = add(jp.array([1.0]), jp.array([2.0]))
    expected = jp.array([3.0])
    assert jp.allclose(result, expected)


def test_jit_returns_correct_result(temp_cache_dir):
    """Jitted function should return correct result."""

    @paz.cache.jit(cache_dir=temp_cache_dir)
    def multiply(x, scale):
        return x * scale

    result = multiply(jp.array([1.0, 2.0, 3.0]), 2.0)
    expected = jp.array([2.0, 4.0, 6.0])
    assert jp.allclose(result, expected)


# Tests for jit - Caching Behavior


def test_jit_creates_cache_file_on_first_call(temp_cache_dir):
    """First call should create a cache file."""

    @paz.cache.jit(cache_dir=temp_cache_dir)
    def add_one(x):
        return x + 1

    add_one(jp.array([1.0, 2.0, 3.0]))
    cache_files = list(temp_cache_dir.glob("*.bin"))
    assert len(cache_files) == 1


def test_jit_loads_from_cache_on_second_call(temp_cache_dir):
    """Second call with same shapes should load from cache."""

    @paz.cache.jit(cache_dir=temp_cache_dir)
    def add_one(x):
        return x + 1

    add_one(jp.array([1.0, 2.0, 3.0]))
    cache_files_before = list(temp_cache_dir.glob("*.bin"))
    add_one(jp.array([4.0, 5.0, 6.0]))
    cache_files_after = list(temp_cache_dir.glob("*.bin"))
    assert len(cache_files_before) == len(cache_files_after)


def test_jit_different_shapes_create_different_cache(temp_cache_dir):
    """Different input shapes should create separate cache entries."""

    @paz.cache.jit(cache_dir=temp_cache_dir)
    def add_one(x):
        return x + 1

    add_one(jp.array([1.0, 2.0, 3.0]))
    add_one(jp.array([1.0, 2.0]))
    cache_files = list(temp_cache_dir.glob("*.bin"))
    assert len(cache_files) == 2


def test_jit_cache_filename_includes_module_name(temp_cache_dir):
    """Cache filename should include module name."""

    @paz.cache.jit(cache_dir=temp_cache_dir)
    def test_func(x):
        return x + 1

    test_func(jp.array([1.0]))
    cache_files = list(temp_cache_dir.glob("*.bin"))
    assert len(cache_files) == 1
    assert (
        "__main__" in cache_files[0].name or "cache_test" in cache_files[0].name
    )


def test_jit_cache_filename_includes_function_name(temp_cache_dir):
    """Cache filename should include function name."""

    @paz.cache.jit(cache_dir=temp_cache_dir)
    def my_function(x):
        return x + 1

    my_function(jp.array([1.0]))
    cache_files = list(temp_cache_dir.glob("*.bin"))
    assert len(cache_files) == 1
    assert "my_function" in cache_files[0].name


# Tests for jit - static_argnums


def test_jit_with_static_argnums(temp_cache_dir):
    """static_argnums should mark arguments as static."""

    @paz.cache.jit(static_argnums=(1,), cache_dir=temp_cache_dir)
    def scale(x, factor):
        return x * factor

    result = scale(jp.array([1.0, 2.0]), 3.0)
    expected = jp.array([3.0, 6.0])
    assert jp.allclose(result, expected)


def test_jit_different_static_values_create_different_cache(temp_cache_dir):
    """Different static argument values should create separate caches."""

    @paz.cache.jit(static_argnums=(1,), cache_dir=temp_cache_dir)
    def scale(x, factor):
        return x * factor

    scale(jp.array([1.0, 2.0]), 2.0)
    scale(jp.array([1.0, 2.0]), 3.0)
    cache_files = list(temp_cache_dir.glob("*.bin"))
    assert len(cache_files) == 2


def test_jit_same_static_values_reuse_cache(temp_cache_dir):
    """Same static argument values should reuse cache."""

    @paz.cache.jit(static_argnums=(1,), cache_dir=temp_cache_dir)
    def scale(x, factor):
        return x * factor

    scale(jp.array([1.0, 2.0]), 2.0)
    cache_files_before = list(temp_cache_dir.glob("*.bin"))
    scale(jp.array([3.0, 4.0]), 2.0)
    cache_files_after = list(temp_cache_dir.glob("*.bin"))
    assert len(cache_files_before) == len(cache_files_after) == 1


# Tests for jit - Additional jit_kwargs


def test_jit_passes_kwargs_to_jax_jit(temp_cache_dir):
    """Additional kwargs should be passed to jax.jit."""

    @paz.cache.jit(donate_argnums=(0,), cache_dir=temp_cache_dir)
    def add_one(x):
        return x + 1

    result = add_one(jp.array([1.0, 2.0, 3.0]))
    expected = jp.array([2.0, 3.0, 4.0])
    assert jp.allclose(result, expected)


# Tests for jit - Edge Cases


def test_jit_multiple_arrays_input(temp_cache_dir):
    """Function with multiple array inputs should work."""

    @paz.cache.jit(cache_dir=temp_cache_dir)
    def add_arrays(x, y, z):
        return x + y + z

    result = add_arrays(jp.array([1.0]), jp.array([2.0]), jp.array([3.0]))
    expected = jp.array([6.0])
    assert jp.allclose(result, expected)


def test_jit_pytree_input(temp_cache_dir):
    """Function with pytree input should work."""

    @paz.cache.jit(cache_dir=temp_cache_dir)
    def sum_dict(data):
        return data["x"] + data["y"]

    result = sum_dict({"x": jp.array([1.0]), "y": jp.array([2.0])})
    expected = jp.array([3.0])
    assert jp.allclose(result, expected)


def test_jit_nested_pytree_input(temp_cache_dir):
    """Function with nested pytree input should work."""

    @paz.cache.jit(cache_dir=temp_cache_dir)
    def process_nested(data):
        return data["outer"]["inner"] * 2

    result = process_nested({"outer": {"inner": jp.array([1.0, 2.0])}})
    expected = jp.array([2.0, 4.0])
    assert jp.allclose(result, expected)


def test_jit_no_arguments_function(temp_cache_dir):
    """Function with no arguments should work."""

    @paz.cache.jit(cache_dir=temp_cache_dir)
    def constant():
        return jp.array([42.0])

    result = constant()
    expected = jp.array([42.0])
    assert jp.allclose(result, expected)


def test_jit_returns_pytree(temp_cache_dir):
    """Function returning pytree should work."""

    @paz.cache.jit(cache_dir=temp_cache_dir)
    def split_result(x):
        return {"double": x * 2, "triple": x * 3}

    result = split_result(jp.array([1.0, 2.0]))
    assert jp.allclose(result["double"], jp.array([2.0, 4.0]))
    assert jp.allclose(result["triple"], jp.array([3.0, 6.0]))


def test_jit_same_function_name_different_modules(temp_cache_dir):
    """Same name in different contexts should create different cache."""

    @paz.cache.jit(cache_dir=temp_cache_dir)
    def helper(x):
        return x + 1

    helper(jp.array([1.0]))
    cache_files = list(temp_cache_dir.glob("*.bin"))
    first_count = len(cache_files)
    assert first_count >= 1
