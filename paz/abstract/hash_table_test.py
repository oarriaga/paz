import pytest
import jax
import jax.numpy as jp

from .hash_table import (
    build,
    lookup,
    Item,
    pytree_to_uint32_with_slot_arg,
    compute_flat_arg,
    lookup_parallel,
    insert,
    insert_parallel,
    Counter,
)


# ### Helper Functions for Advanced Scenarios ###


def find_colliding_items(table, num_colliders=2):
    """
    A helper to find multiple different items that produce the same initial hash.
    """
    hashes_found = {}
    # Use a wider range to increase chances of finding collisions
    for i in range(100000):
        item = Item(
            arg=jp.uint32(i), data=jp.array([i * 2, i * 3], dtype=jp.uint32)
        )
        slot_arg, _ = pytree_to_uint32_with_slot_arg(table, item)
        slot_arg_int = int(slot_arg)

        if slot_arg_int not in hashes_found:
            hashes_found[slot_arg_int] = []

        hashes_found[slot_arg_int].append(item)

        if len(hashes_found[slot_arg_int]) == num_colliders:
            return hashes_found[slot_arg_int]

    raise RuntimeError(
        f"Could not find {num_colliders} items with a hash collision."
    )


def stack_items(items: list[Item]) -> Item:
    """Stacks a list of Item namedtuples into a single batched Item."""
    if not items:
        # Create an empty Item with correctly shaped fields for an empty batch
        first_item_prototype = Item(
            arg=jp.uint32(0), data=jp.array([0, 0], dtype=jp.uint32)
        )
        return Item(
            arg=jp.array([], dtype=first_item_prototype.arg.dtype),
            data=jp.array([], dtype=first_item_prototype.data.dtype).reshape(
                0, *first_item_prototype.data.shape
            ),
        )

    all_args = jp.stack([item.arg for item in items])
    all_data = jp.stack([item.data for item in items])
    return Item(arg=all_args, data=all_data)


# ### Fixtures ###


@pytest.fixture
def item_a():
    return Item(arg=jp.uint32(1), data=jp.array([10, 11], dtype=jp.uint32))


@pytest.fixture
def item_b():
    return Item(arg=jp.uint32(2), data=jp.array([20, 21], dtype=jp.uint32))


@pytest.fixture
def empty_table(item_a):
    return build(
        key=42, num_slots=100, item_data_shape=item_a.data.shape, num_tables=2
    )


@pytest.fixture
def table_with_one_item(empty_table, item_a):
    """Manually builds a table with one item, decoupling test from insert()."""
    table = empty_table
    slot_arg, _ = pytree_to_uint32_with_slot_arg(table, item_a)
    counter = Counter(slot_arg=slot_arg, table_arg=jp.uint8(0))
    flat_arg = compute_flat_arg(counter, table)

    # Manually set the item data and update occupancy
    args = table.args.at[flat_arg].set(item_a.arg)
    data = table.data.at[flat_arg].set(item_a.data)
    occupancy = table.table_occupancy.at[slot_arg].set(1)

    table = table._replace(
        args=args, data=data, table_occupancy=occupancy, size=table.size + 1
    )

    return {"table": table, "item": item_a, "flat_arg": flat_arg}


@pytest.fixture
def table_with_collision(empty_table):
    """Manually builds a table with a hash collision."""
    table = empty_table
    item_1, item_2 = find_colliding_items(table, num_colliders=2)

    # Place item_1 in the first cuckoo slot
    slot_arg, _ = pytree_to_uint32_with_slot_arg(table, item_1)
    counter1 = Counter(slot_arg=slot_arg, table_arg=jp.uint8(0))
    arg_1 = compute_flat_arg(counter1, table)

    # Place item_2 in the second cuckoo slot
    counter2 = Counter(slot_arg=slot_arg, table_arg=jp.uint8(1))
    arg_2 = compute_flat_arg(counter2, table)

    # Update table state
    args = table.args.at[arg_1].set(item_1.arg).at[arg_2].set(item_2.arg)
    data = table.data.at[arg_1].set(item_1.data).at[arg_2].set(item_2.data)
    occupancy = table.table_occupancy.at[slot_arg].set(2)

    table = table._replace(
        args=args, data=data, table_occupancy=occupancy, size=table.size + 2
    )

    return {
        "table": table,
        "item_1": item_1,
        "item_2": item_2,
        "arg_1": arg_1,
        "arg_2": arg_2,
    }


@pytest.fixture
def table_with_full_slot(empty_table):
    """Manually builds a table where a slot is full."""
    item_1, item_2, item_3 = find_colliding_items(empty_table, num_colliders=3)

    # Use the same logic as the collision fixture to fill a slot
    slot_arg, _ = pytree_to_uint32_with_slot_arg(empty_table, item_1)
    counter1 = Counter(slot_arg=slot_arg, table_arg=jp.uint8(0))
    counter2 = Counter(slot_arg=slot_arg, table_arg=jp.uint8(1))
    arg_1 = compute_flat_arg(counter1, empty_table)
    arg_2 = compute_flat_arg(counter2, empty_table)

    args = empty_table.args.at[arg_1].set(item_1.arg).at[arg_2].set(item_2.arg)
    data = (
        empty_table.data.at[arg_1].set(item_1.data).at[arg_2].set(item_2.data)
    )
    occupancy = empty_table.table_occupancy.at[slot_arg].set(2)

    table = empty_table._replace(
        args=args, data=data, table_occupancy=occupancy, size=2
    )

    return {"table": table, "non_inserted_item": item_3}


@pytest.fixture
def table_for_parallel_tests(table_with_collision, item_a):
    """Provides a table with multiple items for batch lookups."""
    # This table contains two colliding items and one simple item.
    table = table_with_collision["table"]
    item_1 = table_with_collision["item_1"]
    item_2 = table_with_collision["item_2"]

    # Insert a third, non-colliding item
    final_table, _, arg_a = insert(table, item_a)

    return {
        "table": final_table,
        "items": [item_1, item_2, item_a],
        "args": [
            table_with_collision["arg_1"],
            table_with_collision["arg_2"],
            arg_a,
        ],
        "non_existent_item": Item(arg=jp.uint32(999), data=jp.array([99, 99])),
    }


### Basic Tests ###


def test_build_initializes_capacity(empty_table):
    assert empty_table.num_slots == 100


def test_build_initializes_size_to_zero(empty_table):
    assert empty_table.size == 0


# ### Granular Tests for the `lookup` Function ###

# #### Scenario: Basic Lookups ####


def test_lookup_in_empty_table_returns_found_false(empty_table, item_a):
    """Verifies that looking for an item in an empty table fails."""
    _, found = lookup(empty_table, item_a)
    assert not found


def test_lookup_existing_item_returns_found_true(table_with_one_item):
    """Verifies that an inserted item can be found."""
    table = table_with_one_item["table"]
    item = table_with_one_item["item"]
    _, found = lookup(table, item)
    assert found


def test_lookup_existing_item_returns_correct_arg(table_with_one_item):
    """Verifies that finding an item returns its correct storage index."""
    table = table_with_one_item["table"]
    item = table_with_one_item["item"]
    expected_arg = table_with_one_item["flat_arg"]
    returned_arg, _ = lookup(table, item)
    assert returned_arg == expected_arg


def test_lookup_nonexistent_item_in_filled_table_returns_found_false(
    table_with_one_item, item_b
):
    """Verifies lookup fails for an item not in a non-empty table."""
    table = table_with_one_item["table"]
    _, found = lookup(table, item_b)
    assert not found


# #### Scenario: Lookups with Hash Collisions ####


def test_lookup_finds_first_colliding_item(table_with_collision):
    """After a collision, verifies the first inserted item is still findable."""
    table = table_with_collision["table"]
    item_1 = table_with_collision["item_1"]
    _, found = lookup(table, item_1)
    assert found


def test_lookup_returns_correct_arg_for_first_colliding_item(
    table_with_collision,
):
    """Verifies the correct index for the first of two colliding items."""
    table = table_with_collision["table"]
    item_1 = table_with_collision["item_1"]
    expected_arg = table_with_collision["arg_1"]
    returned_arg, _ = lookup(table, item_1)
    assert returned_arg == expected_arg


def test_lookup_finds_second_colliding_item(table_with_collision):
    """After a collision, verifies the second item is found in its cuckoo slot."""
    table = table_with_collision["table"]
    item_2 = table_with_collision["item_2"]
    _, found = lookup(table, item_2)
    assert found


def test_lookup_returns_correct_arg_for_second_colliding_item(
    table_with_collision,
):
    """Verifies the correct index for the second of two colliding items."""
    table = table_with_collision["table"]
    item_2 = table_with_collision["item_2"]
    expected_arg = table_with_collision["arg_2"]
    returned_arg, _ = lookup(table, item_2)
    assert returned_arg == expected_arg


# #### Scenario: Advanced Cuckoo Path Lookup ####


def test_lookup_nonexistent_third_collider_returns_found_false(
    table_with_full_slot,
):
    """
    Tests that a lookup for an item whose primary slot is full correctly
    follows the cuckoo path to a new slot and terminates, returning False.
    """
    table = table_with_full_slot["table"]
    non_inserted_item = table_with_full_slot["non_inserted_item"]
    _, found = lookup(table, non_inserted_item)
    assert not found


def test_lookup_parallel_with_empty_batch(empty_table):
    """Tests that passing an empty batch of items returns empty results."""
    empty_batch = stack_items([])

    returned_args, found_flags = lookup_parallel(empty_table, empty_batch)

    assert returned_args.shape[0] == 0
    assert found_flags.shape[0] == 0


def test_lookup_parallel_finds_all_existing_items(table_for_parallel_tests):
    """Verifies a batch lookup finds all items that are present in the table."""
    table = table_for_parallel_tests["table"]
    items_to_find = stack_items(table_for_parallel_tests["items"])

    _, found_flags = lookup_parallel(table, items_to_find)

    assert jp.all(found_flags)


def test_lookup_parallel_returns_correct_args(table_for_parallel_tests):
    """Verifies a batch lookup returns the correct flat arguments for all found items."""
    table = table_for_parallel_tests["table"]
    items_to_find = stack_items(table_for_parallel_tests["items"])
    expected_args = jp.array(table_for_parallel_tests["args"])

    returned_args, _ = lookup_parallel(table, items_to_find)

    assert jp.all(returned_args == expected_args)


def test_lookup_parallel_mixed_found_and_not_found(table_for_parallel_tests):
    """Tests a batch containing both existing and non-existing items."""
    table = table_for_parallel_tests["table"]
    existing_item = table_for_parallel_tests["items"][0]
    non_existent_item = table_for_parallel_tests["non_existent_item"]

    items_to_find = stack_items([existing_item, non_existent_item])
    expected_found = jp.array([True, False])

    _, found_flags = lookup_parallel(table, items_to_find)

    assert jp.all(found_flags == expected_found)


# ### Tests for the `insert` Function ###

# # #### Scenario: Simple Insertion ####


def test_insert_into_empty_table_returns_inserted_true(empty_table, item_a):
    """Verifies that inserting a new item returns an 'inserted' flag of True."""
    _table, inserted, _arg = insert(empty_table, item_a)
    assert inserted


def test_insert_into_empty_table_increments_size(empty_table, item_a):
    """Checks that a successful insert increases the table's size to 1."""
    new_table, _, _ = insert(empty_table, item_a)
    assert new_table.size == 1


def test_insert_stores_item_arg_correctly(empty_table, item_a):
    """Ensures the item's 'arg' field is placed at the correct index."""
    new_table, _, flat_arg = insert(empty_table, item_a)
    assert new_table.args[flat_arg] == item_a.arg


def test_insert_stores_item_data_correctly(empty_table, item_a):
    """Ensures the item's 'data' field is placed at the correct index."""
    new_table, _, flat_arg = insert(empty_table, item_a)
    assert jp.all(new_table.data[flat_arg] == item_a.data)


def test_insert_updates_table_occupancy(empty_table, item_a):
    """Verifies that the occupancy for the item's slot is incremented to 1."""
    slot_arg, _ = pytree_to_uint32_with_slot_arg(empty_table, item_a)
    new_table, _, _ = insert(empty_table, item_a)
    assert new_table.table_occupancy[slot_arg] == 1


# #### Scenario: Duplicate Insertion ####


def test_insert_duplicate_item_returns_inserted_false(empty_table, item_a):
    """Verifies inserting a duplicate item returns an 'inserted' flag of False."""
    table_after_1, _, _ = insert(empty_table, item_a)
    _table_after_2, inserted, _arg = insert(table_after_1, item_a)
    assert not inserted


def test_insert_duplicate_item_does_not_change_size(empty_table, item_a):
    """Checks that inserting a duplicate item does not change the table's size."""
    table_after_1, _, _ = insert(empty_table, item_a)
    table_after_2, _, _ = insert(table_after_1, item_a)
    assert table_after_2.size == table_after_1.size


def test_insert_duplicate_item_returns_same_arg(empty_table, item_a):
    """Ensures that inserting a duplicate returns the original item's index."""
    _, _, flat_arg_1 = insert(empty_table, item_a)
    table_after_1, _, _ = insert(empty_table, item_a)
    _, _, flat_arg_2 = insert(table_after_1, item_a)
    assert flat_arg_1 == flat_arg_2


# # #### Scenario: Collision Insertion ####


def test_insert_colliding_item_returns_inserted_true(empty_table):
    """
    Verifies that inserting a different item with the same hash still returns
    an 'inserted' flag of True.
    """
    item_1, item_2 = find_colliding_items(empty_table, num_colliders=2)
    table_after_1, _, _ = insert(empty_table, item_1)
    _table_after_2, inserted, _ = insert(table_after_1, item_2)
    assert inserted


def test_insert_colliding_item_increments_size(empty_table):
    """Checks that inserting a colliding item increases the total size to 2."""
    item_1, item_2 = find_colliding_items(empty_table, num_colliders=2)
    table_after_1, _, _ = insert(empty_table, item_1)
    table_after_2, _, _ = insert(table_after_1, item_2)
    assert table_after_2.size == 2


def test_insert_colliding_item_updates_occupancy_to_two(empty_table):
    """Verifies that a collision updates the slot's occupancy count to 2."""
    item_1, item_2 = find_colliding_items(empty_table, num_colliders=2)
    slot_arg, _ = pytree_to_uint32_with_slot_arg(empty_table, item_1)

    table_after_1, _, _ = insert(empty_table, item_1)
    table_after_2, _, _ = insert(table_after_1, item_2)

    assert table_after_2.table_occupancy[slot_arg] == 2


def test_insert_colliding_items_are_stored_at_different_args(empty_table):
    """Ensures two colliding items are stored at different flat indices."""
    item_1, item_2 = find_colliding_items(empty_table, num_colliders=2)
    table_after_1, _, flat_arg_1 = insert(empty_table, item_1)
    _table_after_2, _, flat_arg_2 = insert(table_after_1, item_2)
    assert flat_arg_1 != flat_arg_2


### Tests for the `insert_parallel` Function ###

# #### Scenario: Basic Batch Insertion ####


def test_insert_parallel_new_items_returns_all_true_in_mask(
    empty_table, item_a, item_b
):
    """Verifies inserting a batch of new items results in an all-True 'inserted_mask'."""
    items_to_insert = stack_items([item_a, item_b])
    _table, inserted_mask, _args = insert_parallel(empty_table, items_to_insert)
    assert jp.all(inserted_mask)


def test_insert_parallel_new_items_updates_size_correctly(
    empty_table, item_a, item_b
):
    """Checks that the table size is correctly updated after a batch insert."""
    items_to_insert = stack_items([item_a, item_b])
    new_table, _, _ = insert_parallel(empty_table, items_to_insert)
    assert new_table.size == 2


def test_insert_parallel_new_items_are_findable(empty_table, item_a, item_b):
    """Ensures that all items from a batch insert can be found afterward."""
    items_to_insert = stack_items([item_a, item_b])
    new_table, _, _ = insert_parallel(empty_table, items_to_insert)
    _, found_flags = lookup_parallel(new_table, items_to_insert)
    assert jp.all(found_flags)


# #### Scenario: Mixed Batch (Existing and New Items) ####


def test_insert_parallel_mixed_batch_returns_correct_mask(
    table_with_one_item, item_b
):
    """
    Tests that the 'inserted_mask' is False for pre-existing items and
    True for new items in a mixed batch.
    """
    # table_with_one_item fixture already contains item_a
    item_a = table_with_one_item["item"]
    table = table_with_one_item["table"]

    items_to_insert = stack_items(
        [item_a, item_b]
    )  # item_a exists, item_b is new
    _, inserted_mask, _ = insert_parallel(table, items_to_insert)

    assert jp.all(inserted_mask == jp.array([False, True]))


def test_insert_parallel_mixed_batch_updates_size_correctly(
    table_with_one_item, item_b
):
    """Verifies the size is only incremented by the number of new items."""
    item_a = table_with_one_item["item"]
    table = table_with_one_item["table"]

    items_to_insert = stack_items([item_a, item_b])
    new_table, _, _ = insert_parallel(table, items_to_insert)

    assert new_table.size == 2  # Started with 1, added 1 new item


# # #### Scenario: Batch with Internal Collisions ####


def test_insert_parallel_with_collision_batch_inserts_all(empty_table):
    """
    Checks that when a batch contains items that collide, they are all
    successfully inserted.
    """
    item_c1, item_c2 = find_colliding_items(empty_table, num_colliders=2)
    items_to_insert = stack_items([item_c1, item_c2])

    _, inserted_mask, _ = insert_parallel(empty_table, items_to_insert)

    assert jp.all(inserted_mask)


def test_insert_parallel_with_collision_batch_returns_unique_args(empty_table):
    """
    Ensures the conflict resolution loop assigns unique flat arguments to
    colliding items within a batch.
    """
    item_c1, item_c2 = find_colliding_items(empty_table, num_colliders=2)
    items_to_insert = stack_items([item_c1, item_c2])

    _, _, final_args = insert_parallel(empty_table, items_to_insert)

    assert final_args[0] != final_args[1]


def test_insert_parallel_with_collision_batch_updates_occupancy(empty_table):
    """
    Verifies that a batch of colliding items correctly updates the
    shared slot's occupancy count.
    """
    item_c1, item_c2 = find_colliding_items(empty_table, num_colliders=2)
    slot_arg, _ = pytree_to_uint32_with_slot_arg(empty_table, item_c1)
    items_to_insert = stack_items([item_c1, item_c2])

    new_table, _, _ = insert_parallel(empty_table, items_to_insert)

    assert new_table.table_occupancy[slot_arg] == 2


# #### Scenario: Batch with Internal Duplicates ####


def test_insert_parallel_with_duplicate_in_batch_returns_consistent_args(
    empty_table, item_a, item_b
):
    """
    Ensures that duplicate items in a batch are assigned the same final flat argument.
    """
    items_to_insert = stack_items([item_a, item_b, item_a])
    _, _, final_args = insert_parallel(empty_table, items_to_insert)
    assert final_args[0] == final_args[2]


def test_insert_parallel_with_duplicate_in_batch_updates_size_correctly(
    empty_table, item_a, item_b
):
    """
    Checks that if a batch contains duplicates, the final size reflects
    only the unique items being inserted.
    """
    items_to_insert = stack_items(
        [item_a, item_b, item_a]
    )  # Contains a duplicate
    new_table, _, _ = insert_parallel(empty_table, items_to_insert)
    assert new_table.size == 2
