from collections import namedtuple
import jax
import jax.numpy as jp


def hash_fn(item, key):
    hash_value = jp.uint32(key)
    hash_value = (hash_value << 5) ^ item.id
    if item.data.ndim == 2:
        hash_value = (hash_value << 5) ^ item.data[:, 0]
        hash_value = (hash_value << 5) ^ item.data[:, 1]
        item_as_uint32 = jp.concatenate([item.id[:, None], item.data], axis=1)
    else:
        hash_value = (hash_value << 5) ^ item.data[0]
        hash_value = (hash_value << 5) ^ item.data[1]
        item_as_uint32 = jp.concatenate([jp.array([item.id]), item.data])
    return hash_value, item_as_uint32


def equals_fn(item1, item2):
    id_match = jp.all(item1.id == item2.id)
    data_match = jp.all(item1.data == item2.data)
    return jp.logical_and(id_match, data_match)


SIZE_DTYPE = jp.uint32
TABLE_ARG_DTYPE = jp.uint8
CuckooArg = namedtuple("CuckooArg", ["arg", "table_arg"])
HashArg = namedtuple("HashArg", ["arg"])
Item = namedtuple("Item", ["id", "data"])
HashTable = namedtuple(
    "HashTable",
    [
        "key",
        "capacity",
        "hidden_capacity",
        "cuckoo_table_n",
        "size",
        "item_ids",
        "item_datas",
        "table_arg",
        # "hash_fn",
        # "equals_fn",
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


def hash_array(key, array_uint32):
    def scan_body(key, x):
        result = xxhash(key, x)
        return result, result

    hash_value, _ = jax.lax.scan(scan_body, key, array_uint32)
    return hash_value


def hash_to_arg(key, array_uint32, capacity):
    hash_value = hash_array(key, array_uint32)
    arg = hash_value % capacity
    return arg


def item_to_uint32_with_arg(table, item):
    hash_value, item_uint32 = hash_fn(item, table.key)
    return hash_value % table.capacity, item_uint32


def get_batch_size(pytree):
    first_leaf = jax.tree_util.tree_leaves(pytree)[0]
    return first_leaf.shape[0]


def compute_flat_arg(cuckoo_arg, table):
    return (cuckoo_arg.arg * table.cuckoo_table_n) + cuckoo_arg.table_arg


def unique_mask(cuckoo_args):
    batch_size = get_batch_size(cuckoo_args)
    combined_args = (cuckoo_args.arg << 8) + cuckoo_args.table_arg
    _, first_occurrence_indices = jp.unique(
        combined_args, axis=0, size=batch_size, return_index=True
    )
    is_first_occurrence_mask = (
        jp.zeros((batch_size,), dtype=jp.bool_)
        .at[first_occurrence_indices]
        .set(True)
    )
    is_padding_index = first_occurrence_indices >= batch_size
    final_mask = jp.where(is_padding_index, False, is_first_occurrence_mask)
    return final_mask


def build(
    key,
    capacity,
    example_item,
    cuckoo_table_n=2,
    hash_size_multiplier=2,
):
    hidden_capacity = int(hash_size_multiplier * capacity / cuckoo_table_n)
    table_size = (hidden_capacity + 1) * cuckoo_table_n

    size = SIZE_DTYPE(0)
    item_ids = jp.zeros((table_size,), dtype=example_item.id.dtype)
    item_datas = jp.zeros(
        (table_size,) + example_item.data.shape, dtype=example_item.data.dtype
    )

    table_arg = jp.zeros((hidden_capacity + 1,), dtype=TABLE_ARG_DTYPE)
    return HashTable(
        key=key,
        capacity=capacity,
        hidden_capacity=hidden_capacity,
        cuckoo_table_n=cuckoo_table_n,
        size=size,
        item_ids=item_ids,
        item_datas=item_datas,
        table_arg=table_arg,
        # hash_fn=hash_function,
        # equals_fn=equality_function,
    )


def _lookup(key, table, item, item_uint32, cuckoo_arg, is_item_found):
    def calculate_next_probing_slot(now_key, now_cuckoo_arg):
        def move_to_next_hash_function(key, cuckoo_index):
            new_key = key + 1
            hash_arg = hash_to_arg(new_key, item_uint32, table.capacity)
            return new_key, CuckooArg(hash_arg, TABLE_ARG_DTYPE(0))

        def move_to_next_cuckoo_slot(key, cuckoo_arg):
            return key, CuckooArg(cuckoo_arg.arg, cuckoo_arg.table_arg + 1)

        last_hash_slot = now_cuckoo_arg.table_arg >= (table.cuckoo_table_n - 1)
        return jax.lax.cond(
            last_hash_slot,
            move_to_next_hash_function,
            move_to_next_cuckoo_slot,
            now_key,
            now_cuckoo_arg,
        )

    def check_next_slot(state):
        key, cuckoo_arg, item_was_found = state
        arg = compute_flat_arg(cuckoo_arg, table)
        item_in_slot = Item(id=table.item_ids[arg], data=table.item_datas[arg])
        slot_filled = cuckoo_arg.table_arg < table.table_arg[cuckoo_arg.arg]
        is_item_found = jp.logical_and(
            slot_filled, equals_fn(item_in_slot, item)
        )

        def do_nothing(key, cuckoo_index):
            return key, cuckoo_index

        new_key, new_cuckoo_arg = jax.lax.cond(
            is_item_found,
            do_nothing,
            calculate_next_probing_slot,
            key,
            cuckoo_arg,
        )
        return new_key, new_cuckoo_arg, is_item_found

    def do_continue(state):
        _, cuckoo_arg, item_was_found = state
        is_in_bounds = cuckoo_arg.table_arg < table.table_arg[cuckoo_arg.arg]
        return jp.logical_and(~item_was_found, is_in_bounds)

    arg = compute_flat_arg(cuckoo_arg, table)
    item_in_first_slot = Item(
        id=table.item_ids[arg], data=table.item_datas[arg]
    )
    is_first_slot_filled = (
        cuckoo_arg.table_arg < table.table_arg[cuckoo_arg.arg]
    )
    is_found_immediately = jp.logical_and(
        is_first_slot_filled, equals_fn(item_in_first_slot, item)
    )
    is_item_found = jp.logical_or(is_item_found, is_found_immediately)
    state = (key, cuckoo_arg, is_item_found)
    return jax.lax.while_loop(do_continue, check_next_slot, state)


def lookup(table, item):
    arg, item_uint32 = item_to_uint32_with_arg(table, item)
    cuckoo_arg = CuckooArg(arg, TABLE_ARG_DTYPE(0))
    _, final_cuckoo_arg, found = _lookup(
        table.key, table, item, item_uint32, cuckoo_arg, False
    )
    return HashArg(compute_flat_arg(final_cuckoo_arg, table)), found


def _lookup_parallel(table, items):
    vectorized_item_to_arg = jax.vmap(
        item_to_uint32_with_arg, in_axes=(None, 0)
    )
    initial_args, items_as_uint32 = vectorized_item_to_arg(table, items)

    batch_size = get_batch_size(items)
    cuckoo_args = CuckooArg(initial_args, jp.zeros(batch_size, TABLE_ARG_DTYPE))
    keys = jp.full((batch_size,), table.key)
    status = jp.zeros(batch_size, dtype=jp.bool)
    vectorized_lookup = jax.vmap(_lookup, (0, None, 0, 0, 0, 0))
    _, cuckoo_args, status = vectorized_lookup(
        keys, table, items, items_as_uint32, cuckoo_args, status
    )
    return cuckoo_args, status
    # flat_args = compute_flat_arg(cuckoo_args, table)
    # return HashArg(flat_args), status


def lookup_parallel(table, items):
    vectorized_item_to_arg = jax.vmap(
        item_to_uint32_with_arg, in_axes=(None, 0)
    )
    initial_args, items_as_uint32 = vectorized_item_to_arg(table, items)

    batch_size = get_batch_size(items)
    cuckoo_args = CuckooArg(initial_args, jp.zeros(batch_size, TABLE_ARG_DTYPE))
    keys = jp.full((batch_size,), table.key)
    status = jp.zeros(batch_size, dtype=jp.bool)
    vectorized_lookup = jax.vmap(_lookup, (0, None, 0, 0, 0, 0))
    _, cuckoo_args, status = vectorized_lookup(
        keys, table, items, items_as_uint32, cuckoo_args, status
    )
    flat_args = compute_flat_arg(cuckoo_args, table)
    return HashArg(flat_args), status


def insert(table, item):
    def _do_nothing(table, cuckoo_arg):
        return table

    def _insert(table, cuckoo_arg):
        flat_arg = compute_flat_arg(cuckoo_arg, table)
        updated_ids = table.item_ids.at[flat_arg].set(item.id)
        updated_datas = table.item_datas.at[flat_arg].set(item.data)
        table_arg = table.table_arg.at[cuckoo_arg.arg].add(1)
        size = table.size + 1
        return table._replace(
            item_ids=updated_ids,
            item_datas=updated_datas,
            table_arg=table_arg,
            size=size,
        )

    arg, uint32_item = item_to_uint32_with_arg(table, item)
    cuckoo_arg = CuckooArg(arg, TABLE_ARG_DTYPE(0))
    _, cuckoo_arg, is_item_found = _lookup(
        table.key, table, item, uint32_item, cuckoo_arg, False
    )
    table = jax.lax.cond(is_item_found, _do_nothing, _insert, table, cuckoo_arg)
    is_item_inserted = ~is_item_found
    return table, is_item_inserted, HashArg(compute_flat_arg(cuckoo_arg, table))


def resolve_collisions(state, table):
    keys, cuckoo_args, needs_placement, items_as_uint32, should_be_inserted = (
        state
    )

    def get_next_slot(key, cuckoo_arg, item_as_uint32):
        def next_hash(key, _):
            next_key = key + 1
            hash_arg = hash_to_arg(next_key, item_as_uint32, table.capacity)
            return next_key, CuckooArg(hash_arg, TABLE_ARG_DTYPE(0))

        def next_cuckoo(key, current_cuckoo_arg):
            table_arg = current_cuckoo_arg.table_arg + 1
            next_cuckoo_arg = CuckooArg(current_cuckoo_arg.arg, table_arg)
            return key, next_cuckoo_arg

        last_slot = cuckoo_arg.table_arg >= (table.cuckoo_table_n - 1)
        return jax.lax.cond(last_slot, next_hash, next_cuckoo, key, cuckoo_arg)

    get_slots = jax.vmap(get_next_slot, in_axes=(0, 0, 0))
    new_keys, new_cuckoo_args = get_slots(keys, cuckoo_args, items_as_uint32)
    keys = jp.where(needs_placement, new_keys, keys)

    def update(new_value, value):
        return jp.where(needs_placement, new_value, value)

    cuckoo_args = jax.tree_util.tree_map(update, new_cuckoo_args, cuckoo_args)
    is_slot_unique_mask = unique_mask(cuckoo_args)
    needs_placement = jp.logical_and(should_be_inserted, ~is_slot_unique_mask)
    return (
        keys,
        cuckoo_args,
        needs_placement,
        items_as_uint32,
        should_be_inserted,
    )


def insert_at_resolved_slots(table, items, to_be_inserted, cuckoo_args):
    def _filter_leaf(leaf):
        return leaf[to_be_inserted]

    items_to_insert = jax.tree_util.tree_map(_filter_leaf, items)
    cuckoo_args_to_insert = jax.tree_util.tree_map(_filter_leaf, cuckoo_args)
    flat_args = compute_flat_arg(cuckoo_args_to_insert, table)
    updated_ids = table.item_ids.at[flat_args].set(items_to_insert.id)
    updated_datas = table.item_datas.at[flat_args].set(items_to_insert.data)
    num_inserts_per_bucket = (
        jp.zeros_like(table.table_arg).at[cuckoo_args_to_insert.arg].add(1)
    )
    new_table_arg = table.table_arg + num_inserts_per_bucket
    num_items_inserted = jp.sum(to_be_inserted, dtype=SIZE_DTYPE)
    size = table.size + num_items_inserted
    return table._replace(
        item_ids=updated_ids,
        item_datas=updated_datas,
        table_arg=new_table_arg,
        size=size,
    )


def parallel_insert(table, items):
    def do_nothing(table):
        return table

    def resolve_and_insert_batch(table):

        def _get_item_as_uint32(item):
            return item_to_uint32_with_arg(table, item)[1]

        batch_size = get_batch_size(items)
        keys = jp.full((batch_size,), table.key)
        items_as_uint32 = jax.vmap(_get_item_as_uint32)(items)
        needs_placement = jp.logical_and(
            should_be_inserted, ~unique_mask(cuckoo_args)
        )
        loop_state = (
            keys,
            cuckoo_args,
            needs_placement,
            items_as_uint32,
            should_be_inserted,
        )

        def loop_condition(current_state):
            _, _, needs_placement_mask, _, _ = current_state
            return jp.any(needs_placement_mask)

        def loop_body(current_state):
            return resolve_collisions(
                current_state,
                table,
                # items_as_uint32,
                # should_be_inserted,
            )

        _, final_cuckoo_args, _, _, _ = jax.lax.while_loop(
            loop_condition, loop_body, loop_state
        )
        return insert_at_resolved_slots(
            table, items, should_be_inserted, final_cuckoo_args
        )

    cuckoo_args, are_items_found = _lookup_parallel(table, items)
    should_be_inserted = ~are_items_found
    table = jax.lax.cond(
        jp.any(should_be_inserted),
        resolve_and_insert_batch,
        do_nothing,
        table,
    )
    hash_arg = HashArg(compute_flat_arg(cuckoo_args, table))
    return table, should_be_inserted, hash_arg
