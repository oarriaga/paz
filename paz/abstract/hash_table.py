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


def _batch_update_table(table, items_to_insert, counters_for_insert):
    """
    Performs the final batch update on the table's arrays.
    This simplified version assumes it only receives items that need to be inserted.
    """
    num_items_to_insert = get_batch_size(items_to_insert)
    # If there's nothing to insert, return the table as is.
    if num_items_to_insert == 0:
        return table

    insert_args = counters_for_insert.slot_arg
    insert_flat_args = compute_flat_arg(counters_for_insert, table)

    # Atomically update all table fields
    new_args = table.args.at[insert_flat_args].set(items_to_insert.arg)
    new_data = table.data.at[insert_flat_args].set(items_to_insert.data)

    # Increment the occupancy count for each slot where an item was inserted
    occupancy_increments = (
        jp.zeros_like(table.table_occupancy).at[insert_args].add(1)
    )
    new_occupancy = table.table_occupancy + occupancy_increments

    # The new size is simply the old size plus the number of items we received
    new_size = table.size + num_items_to_insert

    return table._replace(
        args=new_args,
        data=new_data,
        table_occupancy=new_occupancy,
        size=new_size,
    )


def _resolve_insert_conflicts(
    table, initial_keys, initial_counters, items_uint32, inserted_mask
):
    """
    Iteratively finds conflict-free insertion slots for a batch of new items.
    This version uses a JIT-compatible tie-breaking rule to prevent errors.
    """

    # This is the main body function for the while_loop.
    def find_valid_slots(loop_state):
        keys, counters, needs_new_slot_mask = loop_state
        num_items = get_batch_size(counters)

        def get_next_probe(key, counter, item_uint32):
            next_ctr_same_slot = Counter(
                counter.slot_arg, counter.table_arg + 1
            )
            new_key = key + 1
            new_slot_arg = hash_to_slot_arg(
                new_key, item_uint32, table.num_slots
            )
            new_table_arg = table.table_occupancy[new_slot_arg]
            next_ctr_new_slot = Counter(new_slot_arg, new_table_arg)
            is_last_table = counter.table_arg >= (table.num_tables - 1)

            def _jump_to_new_slot():
                return new_key, next_ctr_new_slot

            def _advance_in_current_slot():
                return key, next_ctr_same_slot

            return jax.lax.cond(
                is_last_table, _jump_to_new_slot, _advance_in_current_slot
            )

        def _get_next_probe_if_needed(mask, k, c, u32):
            def _get_next():
                return get_next_probe(k, c, u32)

            def _do_nothing():
                return k, c

            return jax.lax.cond(mask, _get_next, _do_nothing)

        next_keys, next_counters = jax.vmap(_get_next_probe_if_needed)(
            needs_new_slot_mask, keys, counters, items_uint32
        )

        # --- CORRECTED JIT-SAFE TIE-BREAKING ---
        proposed_flat_args = compute_flat_arg(next_counters, table)

        # Get the first-occurrence index for each value.
        _vals, first_indices, inverse_indices = jp.unique(
            proposed_flat_args,
            return_index=True,
            return_inverse=True,
            size=num_items,
        )

        # An item is the first occurrence if its own index in the array equals the
        # canonical "first_index" for its value. This avoids boolean masking.
        is_first_occurrence_mask = (
            jp.arange(num_items) == first_indices[inverse_indices]
        )

        has_conflict = jp.logical_not(is_first_occurrence_mask)
        overflowed = next_counters.table_arg >= table.num_tables

        new_needs_new_slot_mask = jp.logical_and(
            inserted_mask, jp.logical_or(overflowed, has_conflict)
        )
        return next_keys, next_counters, new_needs_new_slot_mask

    # Initial conflict check before the loop starts
    initial_flat_args = compute_flat_arg(initial_counters, table)
    num_initial_items = get_batch_size(initial_counters)

    # CORRECTED JIT-SAFE initial conflict check
    _i_vals, i_first_indices, i_inverse_indices = jp.unique(
        initial_flat_args,
        return_index=True,
        return_inverse=True,
        size=num_initial_items,
    )
    initial_is_first_mask = (
        jp.arange(num_initial_items) == i_first_indices[i_inverse_indices]
    )
    initial_has_conflict = jp.logical_not(initial_is_first_mask)
    initial_needs_new_slot = jp.logical_and(inserted_mask, initial_has_conflict)

    def _continue_if_conflicts(state):
        _keys, _counters, needs_new_slot_mask = state
        return jp.any(needs_new_slot_mask)

    # Run the loop
    _final_keys, final_counters, _ = jax.lax.while_loop(
        cond_fun=_continue_if_conflicts,
        body_fun=find_valid_slots,
        init_val=(initial_keys, initial_counters, initial_needs_new_slot),
    )
    return final_counters


def insert_parallel(table, items):
    """
    Inserts a batch of items into the hash table in parallel.
    This version filters data before the final update to correctly handle duplicates.
    """
    # 1. Isolate unique items to work with a smaller, duplicate-free set.
    unique_item_mask, unique_indices, inverse_indices = unique_mask(items)
    unique_items = jax.tree_util.tree_map(lambda x: x[unique_indices], items)
    num_unique_items = get_batch_size(unique_items)

    # 2. Look up only the unique items.
    unique_existing_args, unique_found_mask = lookup_parallel(
        table, unique_items
    )

    # 3. Determine which of the unique items need to be inserted.
    unique_inserted_mask = jp.logical_not(unique_found_mask)

    # 4. Prepare initial state FOR UNIQUE ITEMS ONLY.
    unique_items_uint32 = jax.vmap(pytree_to_uint32)(unique_items)
    initial_keys_for_uniques = jp.full(
        (num_unique_items,), table.key, dtype=jp.uint32
    )
    initial_slot_args_for_uniques = jax.vmap(
        hash_to_slot_arg, in_axes=(None, 0, None)
    )(table.key, unique_items_uint32, table.num_slots)
    initial_table_args_for_uniques = table.table_occupancy[
        initial_slot_args_for_uniques
    ]
    initial_counters_for_uniques = Counter(
        initial_slot_args_for_uniques, initial_table_args_for_uniques
    )

    # 5. Resolve conflicts for the unique items that need inserting.
    final_counters_for_uniques = _resolve_insert_conflicts(
        table,
        initial_keys_for_uniques,
        initial_counters_for_uniques,
        unique_items_uint32,
        unique_inserted_mask,
    )

    # 6. Update the table with ONLY the items that were actually inserted.
    # THIS IS THE KEY FIX: We filter the items and counters *before* the update call.
    items_to_insert = jax.tree_util.tree_map(
        lambda x: x[unique_inserted_mask], unique_items
    )
    counters_for_insert = jax.tree_util.tree_map(
        lambda x: x[unique_inserted_mask], final_counters_for_uniques
    )
    new_table = _batch_update_table(table, items_to_insert, counters_for_insert)

    # 7. Construct the final results for the original full batch by broadcasting.
    inserted_mask = jp.logical_and(
        unique_item_mask, unique_inserted_mask[inverse_indices]
    )

    insert_flat_args_for_uniques = compute_flat_arg(
        final_counters_for_uniques, new_table
    )
    final_args_for_uniques = jp.where(
        unique_found_mask, unique_existing_args, insert_flat_args_for_uniques
    )
    final_args = final_args_for_uniques[inverse_indices]

    return new_table, inserted_mask, final_args
