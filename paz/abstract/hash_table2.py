from collections import namedtuple

import jax
import jax.numpy as jp

SIZE_DTYPE = jp.uint32
TABLE_ARG_DTYPE = jp.uint8
Cuckoo = namedtuple("Cuckoo", ["arg", "probe_arg"])
Item = namedtuple("Item", ["arg", "data"])

from .hash import pytree_to_uint32_with_hash_arg, hash_to_arg

HashTable = namedtuple(
    "HashTable",
    [
        "key",
        "capacity",
        "num_buckets",
        "cuckoo_depth",
        "size",
        "args",
        "data",
        "bucket_occupancy",
    ],
)


def are_equal(item_A, item_B):
    args_match = jp.all(item_A.arg == item_B.arg)
    data_match = jp.all(item_A.data == item_B.data)
    return jp.logical_and(args_match, data_match)


def compute_flat_arg(cuckoo, table):
    return (cuckoo.arg * table.cuckoo_depth) + cuckoo.probe_arg


def build(key, num_slots, item, cuckoo_depth=2, capacity_multiplier=2):
    num_buckets = int(capacity_multiplier * num_slots / cuckoo_depth)
    table_size = (num_buckets + 1) * cuckoo_depth
    size = SIZE_DTYPE(0)
    args = jp.zeros((table_size,), dtype=item.arg.dtype)
    data = jp.zeros((table_size,) + item.data.shape, dtype=item.data.dtype)
    bucket_occupancy = jp.zeros((num_buckets + 1,), dtype=TABLE_ARG_DTYPE)
    return HashTable(
        key,
        num_slots,
        num_buckets,
        cuckoo_depth,
        size,
        args,
        data,
        bucket_occupancy,
    )


def _lookup(key, table, item, item_uint32, cuckoo, is_item_found):

    def next_probe(key, cuckoo):

        def next_hash(key, cuckoo):
            new_key = key + 1
            hash_arg = hash_to_arg(new_key, item_uint32, table.capacity)
            return new_key, Cuckoo(hash_arg, TABLE_ARG_DTYPE(0))

        def next_slot(key, cuckoo):
            return key, Cuckoo(cuckoo.arg, cuckoo.probe_arg + 1)

        is_last_probe = cuckoo.probe_arg >= (table.cuckoo_depth - 1)
        return jax.lax.cond(is_last_probe, next_hash, next_slot, key, cuckoo)

    def check_next_slot(state):
        key, cuckoo, item_was_found = state
        flat_arg = compute_flat_arg(cuckoo, table)
        item_in_slot = Item(table.args[flat_arg], table.data[flat_arg])
        filled = cuckoo.probe_arg < table.bucket_occupancy[cuckoo.arg]
        found = jp.logical_and(filled, are_equal(item_in_slot, item))

        def do_nothing(key, cuckoo):
            return key, cuckoo

        key, cuckoo = jax.lax.cond(found, do_nothing, next_probe, key, cuckoo)
        return key, cuckoo, found

    def _continue(state):
        _, cuckoo, item_was_found = state
        is_in_bounds = cuckoo.probe_arg < table.bucket_occupancy[cuckoo.arg]
        return jp.logical_and(~item_was_found, is_in_bounds)

    flat_arg = compute_flat_arg(cuckoo, table)
    item_in_slot = Item(table.args[flat_arg], table.data[flat_arg])
    filled = cuckoo.probe_arg < table.bucket_occupancy[cuckoo.arg]
    found = jp.logical_and(filled, are_equal(item_in_slot, item))
    found = jp.logical_or(is_item_found, found)
    return jax.lax.while_loop(_continue, check_next_slot, (key, cuckoo, found))


def lookup(table, item):
    arg, item_uint32 = pytree_to_uint32_with_hash_arg(table, item)
    cuckoo, key = Cuckoo(arg, TABLE_ARG_DTYPE(0)), table.key
    _, cuckoo, found = _lookup(key, table, item, item_uint32, cuckoo, False)
    return compute_flat_arg(cuckoo, table), found


def insert(table, item):
    def _do_nothing(table, cuckoo_arg):
        return table

    def _insert(table, cuckoo):
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

    hash_arg, item_uint32 = pytree_to_uint32_with_hash_arg(table, item)
    cuckoo = Cuckoo(hash_arg, TABLE_ARG_DTYPE(0))
    _, cuckoo, found = _lookup(
        table.key, table, item, item_uint32, cuckoo, False
    )
    table = jax.lax.cond(found, _do_nothing, _insert, table, cuckoo)
    is_item_inserted = ~found
    hash_arg = compute_flat_arg(cuckoo, table)
    return table, is_item_inserted, hash_arg
