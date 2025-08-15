from collections import namedtuple

import jax
import jax.numpy as jp

TABLE_SIZE_DTYPE = jp.uint32
TABLE_ARG_DTYPE = jp.uint8
Counter = namedtuple("Counter", ["slot_arg", "table_arg"])
Item = namedtuple("Item", ["arg", "data"])
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
    # TODO change name
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
    return (counter.slot_arg * table.num_tables) + counter.table_arg


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
    # TODO maybe just use an array of shape(sub_table_size, num_tables)
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


def _lookup(table, item):
    # 1. if 1st slot is empty, no search
    # 2. if 1st slot is not empty, check 1st table slot
    # 3. if 1st slot is not empty, check 2nd table slot
    # 4. if 1st slot is not empty, and table slot is full, check next slot
    # 5. if 2nd slot is empty, no search
    slot_arg, query_uint32 = pytree_to_uint32_with_slot_arg(table, item)

    def _next_probe(key, counter):

        def next_slot(key, counter):
            new_key, zero_table_arg = key + 1, TABLE_ARG_DTYPE(0)
            slot_arg = hash_to_slot_arg(new_key, query_uint32, table.num_slots)
            return new_key, Counter(slot_arg, zero_table_arg)

        def next_table(key, counter):
            return key, Counter(counter.slot_arg, counter.table_arg + 1)

        not_last_table = counter.table_arg < (table.num_tables - 1)
        return jax.lax.cond(not_last_table, next_table, next_slot, key, counter)

    def next_probe(state):
        key, counter, _ = state
        flat_arg = compute_flat_arg(counter, table)
        item_in_slot = Item(table.args[flat_arg], table.data[flat_arg])
        filled = counter.table_arg < table.table_occupancy[counter.slot_arg]
        found = jp.logical_and(filled, are_equal(item_in_slot, item))

        def identity(key, counter):
            return key, counter

        key, counter = jax.lax.cond(found, identity, _next_probe, key, counter)
        return key, counter, found

    def continue_search(state):
        _, counter, item_was_found = state
        num_occupants = table.table_occupancy[counter.slot_arg]
        is_in_empty_slot = counter.table_arg >= num_occupants
        return jp.logical_and(~item_was_found, ~is_in_empty_slot)

    state = (table.key, Counter(slot_arg, TABLE_ARG_DTYPE(0)), False)
    _, counter, found = jax.lax.while_loop(continue_search, next_probe, state)
    return counter, found


def lookup(table, item):
    counter, found = _lookup(table, item)
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
        table_occupancy = table.table_occupancy.at[cuckoo.slot_arg].add(1)
        size = table.size + 1
        return table._replace(
            args=table_args,
            data=table_data,
            table_occupancy=table_occupancy,
            size=size,
        )

    counter, found = _lookup(table, item)
    table = jax.lax.cond(found, do_nothing, insert, table, counter)
    return table, jp.logical_not(found), compute_flat_arg(counter, table)


def unique_mask(pytrees, key=None):
    items_uint32 = jax.vmap(pytree_to_uint32)(pytrees)
    batch_size = get_batch_size(items_uint32)
    _, unique_args, inverse_args = jp.unique(
        items_uint32,
        axis=0,
        size=batch_size,
        return_index=True,
        return_inverse=True,
    )
    mask = jp.zeros(batch_size, dtype=jp.bool_).at[unique_args].set(True)
    return mask, unique_args, inverse_args


def insert_parallel(table, items):
    get_hash_and_slots = jax.vmap(pytree_to_uint32_with_slot_arg, (None, 0))
    slot_arg, uint32eds = get_hash_and_slots(table, items)
    batch_size = get_batch_size(items)
    unique, unique_items_args, inverse_indices = unique_mask(items)
    counter = Counter(slot_arg, jp.zeros((batch_size,), dtype=TABLE_ARG_DTYPE))
    keys = jp.full((batch_size,), table.seed, dtype=jp.uint32)

    # TODO should we add unique mask and keys?
    counter, found = jax.vmap(_lookup, (None, 0))(table, items)
    updatable = jp.logical_and(jp.logical_not(found), unique)
    # Perform parallel insertion
    table, inserted_idx = _insert_parallel(
        table, items, uint32eds, keys, counter, updatable
    )

    # Provisional index: found -> counter, inserted -> inserted_idx
    provisional_index = jp.where(found, counter.slot_arg, inserted_idx.slot_arg)
    provisional_table_index = jp.where(
        found, counter.table_arg, inserted_idx.table_arg
    )
    provisional_idx = Counter(provisional_index, provisional_table_index)

    # Only keep indices for unique elements
    correct_indices_for_uniques = Counter(
        provisional_idx.slot_arg[unique_items_args],
        table_arg=provisional_idx.table_arg[unique_items_args],
    )

    # Broadcast to all batch elements using inverse_indices
    final_idx = Counter(
        correct_indices_for_uniques.slot_arg[inverse_indices],
        correct_indices_for_uniques.table_arg[inverse_indices],
    )
    return table, updatable, unique, compute_flat_arg(final_idx, table)
