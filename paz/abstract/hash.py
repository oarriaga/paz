import jax
import jax.numpy as jp


def rotate_left(x, num_steps):
    return (x << num_steps) | (x >> (32 - num_steps))


def xxhash(key, x):
    prime_1 = jp.uint32(0x9E3779B1)
    prime_2 = jp.uint32(0x85EBCA77)
    prime_3 = jp.uint32(0xC2B2AE3D)
    prime_5 = jp.uint32(0x165667B1)
    accumulator = jp.uint32(key) + prime_5
    for _ in range(4):
        lane = x & 255
        accumulator = accumulator + lane * prime_5
        accumulator = rotate_left(accumulator, 11) * prime_1
        x = x >> 8
    accumulator = accumulator ^ (accumulator >> 15)
    accumulator = accumulator * prime_2
    accumulator = accumulator ^ (accumulator >> 13)
    accumulator = accumulator * prime_3
    accumulator = accumulator ^ (accumulator >> 16)
    return accumulator


def array_uint32_to_hash(key, array_uint32):
    def apply_xxhash(key, x):
        result = xxhash(key, x)
        return result, result

    hash_value, _ = jax.lax.scan(apply_xxhash, key, array_uint32)
    return hash_value


def hash_to_arg(key, array_uint32, capacity):
    hash_value = array_uint32_to_hash(key, array_uint32)
    arg = hash_value % capacity
    return arg


def to_bytes(x):
    """Convert input to byte array."""
    if x.dtype == jp.bool_:  # if x is a boolean array and cast to uint8 if true
        x = x.astype(jp.uint8)
    return jax.lax.bitcast_convert_type(x, jp.uint8).reshape(-1)


def byterize(pytree):
    """Convert entire state tree to flattened byte array."""
    pytree = jax.tree_util.tree_map(to_bytes, pytree)
    pytree, _ = jax.tree_util.tree_flatten(pytree)
    if len(pytree) == 0:
        return jp.array([], dtype=jp.uint8)
    return jp.concatenate(pytree)


def bytes_to_uint32_old(x):
    # byte_array = byterize(x.default())  # TODO check
    byte_array = byterize(x)
    bytes_length = byte_array.shape[0]
    pad_length = (4 - bytes_length % 4) % 4
    byte_array = jp.pad(byte_array, (0, pad_length), mode="constant")
    byte_array = byte_array.reshape(-1, 4)
    byte_array = byte_array.astype(jp.uint32)
    # TODO chek if this is correct
    bitcast = jax.vmap(lambda x: jax.lax.bitcast_convert_type(x, jp.uint32))
    return bitcast(byte_array).reshape(-1)


def bytes_to_uint32(byte_array):
    bytes_length = byte_array.shape[0]
    pad_length = (4 - bytes_length % 4) % 4
    padded_array = jp.pad(byte_array, (0, pad_length), mode="constant")
    reshaped_array = padded_array.reshape(-1, 4)  # Shape (N, 4), dtype=uint8
    return jax.lax.bitcast_convert_type(reshaped_array, new_dtype=jp.uint32)


def pytree_to_uint32(pytree):
    byte_array = byterize(pytree)
    return bytes_to_uint32(byte_array)


def hash_pytree(pytree, key=0):
    array_uint32 = pytree_to_uint32(pytree)
    return array_uint32_to_hash(key, array_uint32)


def pytree_to_hash_with_array_uint32(pytree, key=0):
    array_uint32 = pytree_to_uint32(pytree)
    hash_value = array_uint32_to_hash(key, array_uint32)
    return hash_value, array_uint32
