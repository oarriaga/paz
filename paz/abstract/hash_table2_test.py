import pytest
import jax
import jax.numpy as jp
from collections import namedtuple

from .hash_table2 import (
    build,
    lookup,
    insert,
    Item,
    are_equal,
)
from . import hash as hash_helpers


@pytest.fixture
def item_a():
    return Item(arg=jp.uint32(1), data=jp.array([10, 11], dtype=jp.uint32))


@pytest.fixture
def item_b():
    return Item(arg=jp.uint32(2), data=jp.array([20, 21], dtype=jp.uint32))


@pytest.fixture
def item_c():
    return Item(arg=jp.uint32(3), data=jp.array([30, 31], dtype=jp.uint32))


@pytest.fixture
def empty_table(item_a):
    return build(key=42, num_slots=100, item=item_a)


@pytest.fixture
def table_with_one_item(empty_table, item_a):
    table, _, returned_arg = insert(empty_table, item_a)
    return table, item_a, returned_arg


@pytest.fixture
def table_after_collision(monkeypatch, empty_table, item_a, item_b):
    def colliding_hash_function(table, pytree):
        return 42, jp.array([0, 0, 0])

    monkeypatch.setattr(
        hash_helpers, "pytree_to_uint32_with_hash_arg", colliding_hash_function
    )

    table_after_a, was_inserted_a, arg_a = insert(empty_table, item_a)
    table_after_b, was_inserted_b, arg_b = insert(table_after_a, item_b)

    return {
        "final_table": table_after_b,
        "arg_a": arg_a,
        "arg_b": arg_b,
    }


# --- Unit Tests ---


### build()
def test_build_initializes_capacity(empty_table):
    assert empty_table.capacity == 100


def test_build_initializes_size_to_zero(empty_table):
    assert empty_table.size == 0


def test_build_initializes_args_array_with_correct_shape(empty_table):
    assert empty_table.args.shape == (202,)


def test_build_initializes_data_array_with_correct_shape(empty_table, item_a):
    assert empty_table.data.shape == (202,) + item_a.data.shape


def test_build_initializes_bucket_occupancy_array_with_correct_shape(
    empty_table,
):
    assert empty_table.bucket_occupancy.shape == (101,)


### lookup()
def test_lookup_finds_existing_item(table_with_one_item):
    table, item, _ = table_with_one_item
    _, is_found = lookup(table, item)
    assert is_found


def test_lookup_fails_for_non_existent_item(empty_table, item_a):
    _, is_found = lookup(empty_table, item_a)
    assert not is_found


def test_lookup_returns_correct_arg(table_with_one_item):
    table, item, inserted_arg = table_with_one_item
    looked_up_arg, _ = lookup(table, item)
    assert looked_up_arg == inserted_arg


### insert()
def test_insert_new_item_increases_size(empty_table, item_a):
    new_table, _, _ = insert(empty_table, item_a)
    assert new_table.size == 1


def test_insert_new_item_reports_true(empty_table, item_a):
    _, was_inserted, _ = insert(empty_table, item_a)
    assert was_inserted


def test_insert_new_item_can_be_found(empty_table, item_a):
    new_table, _, _ = insert(empty_table, item_a)
    _, is_found = lookup(new_table, item_a)
    assert is_found


def test_insert_duplicate_item_does_not_increase_size(table_with_one_item):
    table, item, _ = table_with_one_item
    new_table, _, _ = insert(table, item)
    assert new_table.size == 1


def test_insert_duplicate_item_reports_false(table_with_one_item):
    table, item, _ = table_with_one_item
    _, was_inserted, _ = insert(table, item)
    assert not was_inserted


def test_insert_is_immutable(empty_table, item_a):
    original_args_copy = empty_table.args.copy()
    insert(empty_table, item_a)
    assert jp.array_equal(empty_table.args, original_args_copy)


### Cuckoo Collision Handling
def test_collision_insert_increases_size_correctly(table_after_collision):
    assert table_after_collision["final_table"].size == 2


def test_collision_insert_results_in_different_final_args(
    table_after_collision,
):
    assert table_after_collision["arg_a"] != table_after_collision["arg_b"]


def test_collision_insert_allows_finding_first_item(
    table_after_collision, item_a
):
    final_table = table_after_collision["final_table"]
    _, is_found = lookup(final_table, item_a)
    assert is_found


def test_collision_insert_allows_finding_second_item(
    table_after_collision, item_b
):
    final_table = table_after_collision["final_table"]
    _, is_found = lookup(final_table, item_b)
    assert is_found
