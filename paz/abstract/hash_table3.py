# there is a predefined user size of the hash table.
# internally there are subtables build with extra capacity

from collections import namedtuple

import jax
import jax.numpy as jp

TABLE_SIZE_DTYPE = jp.uint32
TABLE_ARG_DTYPE = jp.uint8
Counter = namedtuple("Counter", ["slot_arg", "table_arg"])
Item = namedtuple("Item", ["arg", "data"])


def xxhash(key, x):

    def rotate_left(x, num_steps):
        return (x << num_steps) | (x >> (32 - num_steps))

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


def hash_to_slot_arg(key, array_uint32, capacity):
    hash_value = array_uint32_to_hash(key, array_uint32)
    arg = hash_value % capacity
    return arg


def byterize(pytree):
    """Convert entire state tree to flattened byte array."""

    def to_bytes(x):
        """Convert input to byte array."""
        if x.dtype == jp.bool_:  # if x is boolean array cast to uint8 if true
            x = x.astype(jp.uint8)
        return jax.lax.bitcast_convert_type(x, jp.uint8).reshape(-1)

    pytree = jax.tree_util.tree_map(to_bytes, pytree)
    pytree, _ = jax.tree_util.tree_flatten(pytree)
    if len(pytree) == 0:
        return jp.array([], dtype=jp.uint8)
    return jp.concatenate(pytree)


def bytes_to_uint32(byte_array):
    bytes_length = byte_array.shape[0]
    pad_length = (4 - bytes_length % 4) % 4
    padded_array = jp.pad(byte_array, (0, pad_length), mode="constant")
    reshaped_array = padded_array.reshape(-1, 4)  # Shape (N, 4), dtype=uint8
    return jax.lax.bitcast_convert_type(reshaped_array, new_dtype=jp.uint32)


def pytree_to_uint32(pytree):
    byte_array = byterize(pytree)
    return bytes_to_uint32(byte_array)


def pytree_to_uint32_with_slot_arg(table, pytree):
    item_uint32 = pytree_to_uint32(pytree)
    slot_arg = hash_to_slot_arg(table.key, item_uint32, table.num_slots)
    return slot_arg, item_uint32


def are_equal(item_A, item_B):
    args_match = jp.all(item_A.arg == item_B.arg)
    data_match = jp.all(item_A.data == item_B.data)
    return jp.logical_and(args_match, data_match)


def get_batch_size(pytree):
    first_leaf = jax.tree_util.tree_leaves(pytree)[0]
    return first_leaf.shape[0]


def compute_flat_arg(counter, table):
    return (counter.arg * table.num_tables) + counter.table_arg


# def unflatten_arg(flat_slot_arg, num_tables):
#     return (counter.arg * num_tables) + counter.table_arg


HashTable = namedtuple(
    "HashTable",
    [
        "key",
        "num_slots",
        "sub_table_size",
        "num_tables",
        "size",
        "args",
        "data",
        "table_occupancy",
    ],
)


def build(
    num_slots,
    item_data_shape,
    item_arg_dtype=jp.uint32,
    item_data_dtype=jp.uint32,
    num_tables=2,
    capacity_multiplier=2,
    key=777,
):
    sub_table_size = int(capacity_multiplier * num_slots / num_tables)
    internal_num_slots = sub_table_size * num_tables
    args = jp.zeros((internal_num_slots,), item_arg_dtype)
    data = jp.zeros((internal_num_slots,) + item_data_shape, item_data_dtype)
    table_occupancy = jp.zeros((sub_table_size), dtype=TABLE_ARG_DTYPE)
    size = TABLE_SIZE_DTYPE(0)
    return HashTable(
        key,
        num_slots,
        sub_table_size,
        num_tables,
        size,
        args,
        data,
        table_occupancy,
    )


def lookup(table, item):
    slot_arg, query_uint32 = pytree_to_uint32_with_slot_arg(table, item)

    def next_pointer(key, counter):

        def next_slot(key, counter):
            # This will only happen if reached last sub-table.
            # If the last sub-table would have been invalid it would have
            # exited in the continue_search function.
            new_key = key + 1
            slot_arg = hash_to_slot_arg(new_key, query_uint32, table.capacity)
            return new_key, Counter(slot_arg, TABLE_ARG_DTYPE(0))

        def next_table(key, counter):
            return key, Counter(counter.slot_arg, counter.table_arg + 1)

        # last_table_arg = counter.table_arg >= (table.num_tables - 1)
        # return jax.lax.cond(last_table_arg, next_slot, next_table, key, counter)
        not_last_table = counter.table_arg < (table.num_tables - 1)
        return jax.lax.cond(not_last_table, next_table, next_slot, key, counter)

    def move_pointer(state):
        key, counter, _ = state
        flat_arg = compute_flat_arg(counter, table)
        item_in_slot = Item(table.args[flat_arg], table.data[flat_arg])
        filled = counter.table_arg < table.table_occupancy[counter.slot_arg]
        found = jp.logical_and(filled, are_equal(item_in_slot, item))

        def identity(key, counter):
            return key, counter

        key, counter = jax.lax.cond(found, identity, next_pointer, key, counter)
        return key, counter, found

    def continue_search(state):
        _, counter, item_was_found = state
        num_tables_used = table.table_occupancy[counter.slot_arg]
        has_more_tables_to_check = counter.table_arg < num_tables_used
        # TODO this will not work unless removing items correctly moves items
        # has_more_tables_to_check = counter.table_arg < table.num_tables
        item_was_not_found = jp.logical_not(item_was_found)
        return jp.logical_and(item_was_not_found, has_more_tables_to_check)

    state = (table.key, Counter(slot_arg, TABLE_ARG_DTYPE(0)), False)
    _, counter, found = jax.lax.while_loop(continue_search, move_pointer, state)
    return compute_flat_arg(counter, table), found


def lookup_parallel(table, items):
    return jax.vmap(lookup, (None, 0))(table, items)


def insert(table, item):
    def do_nothing(table, cuckoo):
        return table

    def insert(table, cuckoo):
        flat_arg = compute_flat_arg(cuckoo, table)
        table_args = table.args.at[flat_arg].set(item.arg)
        table_data = table.data.at[flat_arg].set(item.data)
        bucket_occupancy = table.bucket_occupancy.at[cuckoo.arg].add(1)
        size = table.size + 1
        return table._replace(
            args=table_args,
            data=table_data,
            bucket_occupancy=bucket_occupancy,
            size=size,
        )

    slot_arg, item_uint32 = pytree_to_uint32_with_slot_arg(table, item)
    flat_slot_arg, found = lookup(table, item)
    table = jax.lax.cond(found, do_nothing, insert, table, flat_slot_arg)
    return table, jp.logical_not(found), flat_slot_arg
