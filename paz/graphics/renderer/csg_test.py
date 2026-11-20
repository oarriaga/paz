import jax.numpy as jp
import pytest

from paz.graphics import CSG, UNION, INTERSECTION, DIFFERENCE, FARAWAY
from paz.graphics.renderer import csg

PRESENT = {0: jp.array(True), 1: jp.array(True)}


def evaluate_pair(operation, is_present=PRESENT):
    is_inside = {0: jp.array([False, False, True, True])}
    is_inside[1] = jp.array([False, True, False, True])
    return csg.evaluate(CSG(0, 1, operation), is_inside, is_present)[0]


def test_collect_leaves_follows_tree_order():
    tree = CSG(CSG(3, 0, UNION), CSG(2, CSG(5, 1, UNION), UNION), UNION)
    assert csg.collect_leaves(tree) == [3, 0, 2, 5, 1]


def test_evaluate_union():
    expected = jp.array([False, True, True, True])
    assert jp.array_equal(evaluate_pair(UNION), expected)


def test_evaluate_intersection():
    expected = jp.array([False, False, False, True])
    assert jp.array_equal(evaluate_pair(INTERSECTION), expected)


def test_evaluate_difference():
    expected = jp.array([False, False, True, False])
    assert jp.array_equal(evaluate_pair(DIFFERENCE), expected)


@pytest.mark.parametrize("operation", [UNION, INTERSECTION, DIFFERENCE])
def test_masked_right_operand_is_skipped(operation):
    is_present = {0: jp.array(True), 1: jp.array(False)}
    expected = jp.array([False, False, True, True])
    assert jp.array_equal(evaluate_pair(operation, is_present), expected)


@pytest.mark.parametrize("operation", [UNION, INTERSECTION])
def test_masked_left_operand_is_skipped(operation):
    is_present = {0: jp.array(False), 1: jp.array(True)}
    expected = jp.array([False, True, False, True])
    assert jp.array_equal(evaluate_pair(operation, is_present), expected)


def test_difference_without_left_operand_is_empty():
    is_present = {0: jp.array(False), 1: jp.array(True)}
    assert not bool(jp.any(evaluate_pair(DIFFERENCE, is_present)))


def test_node_without_operands_is_skipped_by_its_parent():
    is_inside = {leaf: jp.array([True]) for leaf in range(3)}
    is_present = {0: jp.array(True), 1: jp.array(False), 2: jp.array(False)}
    tree = CSG(0, CSG(1, 2, UNION), INTERSECTION)
    assert bool(csg.evaluate(tree, is_inside, is_present)[0][0])


def select_roots(operation, is_present=PRESENT):
    enters = jp.array([[2.0], [1.0]])
    exits = jp.array([[6.0], [4.0]])
    args = CSG(0, 1, operation), enters, exits, is_present
    return csg.select_surface_roots(*args)[..., 0]


def test_union_keeps_outer_roots():
    expected = jp.array([[FARAWAY, 1.0], [6.0, FARAWAY]])
    assert jp.array_equal(select_roots(UNION), expected)


def test_intersection_keeps_inner_roots():
    expected = jp.array([[2.0, FARAWAY], [FARAWAY, 4.0]])
    assert jp.array_equal(select_roots(INTERSECTION), expected)


def test_difference_keeps_cut_and_far_roots():
    expected = jp.array([[FARAWAY, FARAWAY], [6.0, 4.0]])
    assert jp.array_equal(select_roots(DIFFERENCE), expected)


def test_masked_leaf_has_no_roots():
    is_present = {0: jp.array(True), 1: jp.array(False)}
    expected = jp.array([[2.0, FARAWAY], [6.0, FARAWAY]])
    assert jp.array_equal(select_roots(DIFFERENCE, is_present), expected)


def test_roots_behind_the_origin_are_dropped():
    enters, exits = jp.array([[-3.0]]), jp.array([[2.0]])
    is_present = {0: jp.array(True)}
    roots = csg.select_surface_roots(0, enters, exits, is_present)
    assert jp.array_equal(roots[..., 0], jp.array([[FARAWAY], [2.0]]))


def test_shadow_ray_drops_only_the_root_it_starts_on():
    enters, exits = jp.array([[1e-4]]), jp.array([[3.0]])
    roots = jp.stack([enters, exits])
    args = roots, enters, exits
    dropped = csg.drop_starting_roots(*args, jp.array([[True]]))
    assert jp.array_equal(dropped[..., 0], jp.array([[FARAWAY], [3.0]]))
    kept = csg.drop_starting_roots(*args, jp.array([[False]]))
    assert jp.array_equal(kept, roots)
