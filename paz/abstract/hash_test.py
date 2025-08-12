import pytest
import jax
import jax.numpy as jp
import numpy as np
from numpy.testing import assert_array_equal

from .hash import (
    rotate_left,
    to_bytes,
    byterize,
    bytes_to_uint32,
    pytree_to_uint32,
    xxhash,
    array_uint32_to_hash,
    hash_to_arg,
    hash_pytree,
    pytree_to_hash_with_array_uint32,
)


def maybe_jit(func, jitted):
    return jax.jit(func) if jitted else func


def test_rotate_left_simple():
    """Tests a simple rotation with no wrapping."""
    inp = jp.uint32(0b1100)
    result = rotate_left(inp, 2)
    assert result == 0b110000


def test_rotate_left_wrap_around():
    """Tests a rotation that causes bits to wrap from the left to the right side."""
    inp = jp.uint32(0x80000000)  # MSB is 1, rest are 0
    result = rotate_left(inp, 1)
    assert result == 1  # The MSB should become the LSB


def test_rotate_left_by_zero():
    """Tests that rotating by zero steps changes nothing."""
    inp = jp.uint32(12345)
    result = rotate_left(inp, 0)
    assert result == inp


def test_to_bytes_with_uint8():
    """Tests casting a uint8 array, which should be a no-op."""
    inp = jp.array([1, 2, 3], dtype=jp.uint8)
    result = to_bytes(inp)
    assert_array_equal(result, np.array([1, 2, 3], dtype=np.uint8))


def test_to_bytes_with_bool():
    """Tests conversion of a boolean array to bytes."""
    inp = jp.array([True, False, True], dtype=jp.bool_)
    result = to_bytes(inp)
    assert_array_equal(result, np.array([1, 0, 1], dtype=np.uint8))


def test_to_bytes_with_int32():
    """Tests bit-casting a 32-bit integer to bytes (little-endian)."""
    inp = jp.array([256], dtype=jp.int32)  # 256 is 0x00000100
    result = to_bytes(inp)
    # Little-endian representation: [0, 1, 0, 0]
    assert_array_equal(result, np.array([0, 1, 0, 0], dtype=np.uint8))


def test_byterize_simple_dict():
    """Tests byterizing a simple PyTree (a dictionary)."""
    pytree = {"a": jp.array([1, 2], dtype=jp.uint8)}
    result = byterize(pytree)
    assert_array_equal(result, np.array([1, 2], dtype=np.uint8))


def test_byterize_nested_pytree():
    """Tests byterizing a nested PyTree with multiple types."""
    pytree = {
        "a": jp.array([True], dtype=jp.bool_),
        "b": [jp.array([255], dtype=jp.uint8), jp.array([3], dtype=jp.uint8)],
    }
    result = byterize(pytree)
    # The flattened array should be a concatenation of all leaves.
    assert_array_equal(result, np.array([1, 255, 3], dtype=np.uint8))


def test_byterize_empty_pytree():
    """Tests that an empty PyTree results in an empty byte array."""
    result = byterize({})
    assert result.shape == (0,)
    assert result.dtype == jp.uint8


@pytest.mark.parametrize("jitted", [True, False])
def test_bytes_to_uint32_perfect_fit(jitted):
    """Tests conversion when the byte array length is already a multiple of 4."""
    func = maybe_jit(bytes_to_uint32, jitted)
    inp = jp.array([1, 0, 0, 0, 0, 1, 0, 0], dtype=jp.uint8)
    result = func(inp)
    assert_array_equal(result, np.array([1, 256], dtype=np.uint32))


@pytest.mark.parametrize("jitted", [True, False])
def test_bytes_to_uint32_needs_padding(jitted):
    """Tests conversion when the byte array needs padding at the end."""
    func = maybe_jit(bytes_to_uint32, jitted)
    inp = jp.array(
        [1, 0, 0, 0, 2], dtype=jp.uint8
    )  # Length 5, needs 3 padding bytes
    result = func(inp)
    # Expected: [1,0,0,0] -> 1; [2,0,0,0] -> 2
    assert_array_equal(result, np.array([1, 2], dtype=np.uint32))


@pytest.mark.parametrize("jitted", [True, False])
def test_bytes_to_uint32_empty(jitted):
    """Tests that an empty byte array results in an empty uint32 array."""
    func = maybe_jit(bytes_to_uint32, jitted)
    result = func(jp.array([], dtype=jp.uint8))
    assert result.shape == (0,)


@pytest.mark.parametrize("jitted", [True, False])
def test_pytree_to_uint32_simple(jitted):
    """Tests the end-to-end conversion of a simple PyTree."""
    func = maybe_jit(pytree_to_uint32, jitted)
    pytree = {"data": jp.array([1, 0, 0, 0, 2, 0, 0, 0], dtype=jp.uint8)}
    result = func(pytree)
    assert_array_equal(result, np.array([1, 2], dtype=np.uint32))


@pytest.mark.parametrize("jitted", [True, False])
def test_array_uint32_to_hash_single_element(jitted):
    """Tests that hashing a single-element array is the same as a single hash step."""
    func = maybe_jit(array_uint32_to_hash, jitted)
    key, val = jp.uint32(0), jp.uint32(123)
    expected = xxhash(key, val)
    result = func(key, jp.array([val], dtype=jp.uint32))
    assert result == expected


@pytest.mark.parametrize("jitted", [True, False])
def test_array_uint32_to_hash_multiple_elements(jitted):
    """Tests the sequential nature of the scan-based hash."""
    func = maybe_jit(array_uint32_to_hash, jitted)
    key = jp.uint32(0)
    arr = jp.array([10, 20], dtype=jp.uint32)

    # Manually compute the two-step hash
    step1 = xxhash(key, arr[0])
    expected = xxhash(step1, arr[1])

    result = func(key, arr)
    assert result == expected


@pytest.mark.parametrize("jitted", [True, False])
def test_array_uint32_to_hash_empty_array(jitted):
    """Tests that hashing an empty array returns the initial key."""
    func = maybe_jit(array_uint32_to_hash, jitted)
    key = jp.uint32(12345)
    result = func(key, jp.array([], dtype=jp.uint32))
    assert result == key


## Tests for hash (top-level) ##


@pytest.mark.parametrize("jitted", [True, False])
def test_hash_simple_pytree(jitted):
    """Tests the top-level hash function with a simple tree."""
    func = maybe_jit(hash_pytree, jitted)
    key = 42
    pytree = {
        "x": jp.array([1, 0, 0, 0], dtype=jp.uint8)
    }  # This becomes one uint32(1)

    # Expected result: hash the array [1] with key 42
    expected = array_uint32_to_hash(
        jp.uint32(key), jp.array([1], dtype=jp.uint32)
    )
    result = func(pytree, key=jp.uint32(key))
    assert result == expected


## Tests for hash_to_arg ##


@pytest.mark.parametrize("jitted", [True, False])
def test_hash_to_arg_simple(jitted):
    """Tests the modulo logic of mapping a hash to an argument index."""
    # This test is a bit indirect, we mock the hashing part
    # Let's assume the hash of [1] is a known value.
    # We can run it once to find out.
    known_hash_val = array_uint32_to_hash(
        jp.uint32(0), jp.array([1], dtype=jp.uint32)
    )

    func = maybe_jit(hash_to_arg, jitted)
    capacity = jp.uint32(100)
    expected_arg = known_hash_val % capacity

    # We pass the key and array that produce the known hash
    result = func(jp.uint32(0), jp.array([1], dtype=jp.uint32), capacity)
    assert result == expected_arg


## Tests for pytree_to_hash_with_array_uint32 ##


@pytest.mark.parametrize("jitted", [True, False])
def test_pytree_to_hash_with_array_uint32_return_structure(jitted):
    """Tests that the function returns a tuple of (hash, array)."""
    func = maybe_jit(pytree_to_hash_with_array_uint32, jitted)
    pytree = {"x": jp.array([1, 0, 0, 0], dtype=jp.uint8)}

    result_hash, result_array = func(pytree, key=0)

    expected_array = jp.array([1], dtype=jp.uint32)
    expected_hash = array_uint32_to_hash(jp.uint32(0), expected_array)

    assert result_hash == expected_hash
    assert_array_equal(result_array, expected_array)
