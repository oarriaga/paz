import pytest
import jax.numpy as jp

from .hash_table import (
    build,
    lookup,
    lookup_parallel,
    insert,
    parallel_insert,
    compute_flat_arg,
    Item,
)


def batch_items(items):
    ids = jp.array([item.id for item in items])
    datas = jp.stack([item.data for item in items])
    return Item(id=ids, data=datas)


@pytest.fixture
def item_a():
    return Item(
        id=jp.array(1, dtype=jp.uint8),
        data=jp.array([10, 11], dtype=jp.uint32),
    )


@pytest.fixture
def item_b():
    return Item(
        id=jp.array(2, dtype=jp.uint8), data=jp.array([20, 21], dtype=jp.uint32)
    )


@pytest.fixture
def item_c():
    return Item(
        id=jp.array(3, dtype=jp.uint8), data=jp.array([30, 31], dtype=jp.uint32)
    )


@pytest.fixture
def empty_table(item_a):
    return build(
        key=42,
        capacity=100,
        example_item=item_a,
    )


@pytest.fixture
def table_with_one_item(empty_table, item_a):
    table, _, _ = insert(empty_table, item_a)
    return table, item_a


def test_build_initializes_capacity(empty_table):
    assert empty_table.capacity == 100


def test_build_initializes_size_to_zero(empty_table):
    assert empty_table.size == 0


def test_lookup_finds_existing_item(table_with_one_item):
    table, item = table_with_one_item
    _, found = lookup(table, item)
    assert found


def test_lookup_fails_on_non_existent_item(table_with_one_item, item_b):
    table, _ = table_with_one_item
    _, found = lookup(table, item_b)
    assert not found


def test_insert_new_item_increases_size(empty_table, item_a):
    new_table, _, _ = insert(empty_table, item_a)
    assert new_table.size == 1


def test_insert_new_item_reports_insertion(empty_table, item_a):
    _, was_inserted, _ = insert(empty_table, item_a)
    assert was_inserted


def test_insert_duplicate_item_does_not_increase_size(table_with_one_item):
    table, item = table_with_one_item
    new_table, _, _ = insert(table, item)
    assert new_table.size == 1


def test_insert_duplicate_item_reports_no_insertion(table_with_one_item):
    table, item = table_with_one_item
    _, was_inserted, _ = insert(table, item)
    assert not was_inserted


def test_lookup_parallel_finds_all_existing_items(empty_table, item_a, item_b):
    table, _, _ = insert(empty_table, item_a)
    table, _, _ = insert(table, item_b)
    items_to_find = batch_items([item_a, item_b])
    _, found_mask = lookup_parallel(table, items_to_find)
    assert jp.all(found_mask)


def test_lookup_parallel_finds_mixed_batch_correctly(
    empty_table, item_a, item_b, item_c
):
    table, _, _ = insert(empty_table, item_a)
    table, _, _ = insert(table, item_c)
    items_to_find = batch_items([item_a, item_b, item_c])
    _, found_mask = lookup_parallel(table, items_to_find)
    expected_mask = jp.array([True, False, True])
    assert jp.array_equal(found_mask, expected_mask)


def test_parallel_insert_new_items_increases_size(empty_table, item_a, item_b):
    items_to_insert = batch_items([item_a, item_b])
    new_table, _, _ = parallel_insert(empty_table, items_to_insert)
    assert new_table.size == 2


def test_parallel_insert_reports_correct_insertion_mask(
    table_with_one_item, item_b
):
    table, item_a = table_with_one_item
    items_to_insert = batch_items([item_a, item_b])
    _, was_inserted_mask, _ = parallel_insert(table, items_to_insert)
    expected_mask = jp.array([False, True])
    assert jp.array_equal(was_inserted_mask, expected_mask)


def test_parallel_insert_with_in_batch_duplicates_inserts_once(
    empty_table, item_a, item_b
):
    items_to_insert = batch_items([item_a, item_b, item_a])
    new_table, _, _ = parallel_insert(empty_table, items_to_insert)
    assert new_table.size == 2


def hash_item_colliding(item, key):
    hash_value = jp.uint32(12345)  # Force the same hash value
    if item.data.ndim == 2:
        item_as_uint32 = jp.concatenate([item.id[:, None], item.data], axis=1)
    else:
        item_as_uint32 = jp.concatenate([jp.array([item.id]), item.data])
    return hash_value, item_as_uint32


def test_parallel_insert_handles_in_batch_hash_collision(item_a):
    table = build(
        key=42,
        capacity=100,
        example_item=item_a,
    )
    item_d = Item(id=4, data=jp.array([40, 41], dtype=jp.uint32))
    item_e = Item(id=5, data=jp.array([50, 51], dtype=jp.uint32))
    items_to_insert = batch_items([item_d, item_e])

    table, was_inserted_mask, hash_args = parallel_insert(
        table, items_to_insert
    )

    assert table.size == 2
    assert jp.all(was_inserted_mask)
    assert hash_args.arg[0] != hash_args.arg[1]
