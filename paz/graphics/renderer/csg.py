import jax.numpy as jp

import paz
from paz.graphics.constants import DIFFERENCE, INTERSECTION, UNION
from paz.graphics.constants import FARAWAY
from paz.graphics.geometry import is_ahead


def intersect(tree, roots, mask):
    """Closest surface depth of a CSG tree, as one row per leaf shape.

    `roots` holds every root of every leaf, shaped `(L, 4, N)`. A root is a
    surface of the tree only where crossing it changes whether the ray is
    inside the tree, so each leaf row shades and shadows like any shape.
    """
    is_present = map_leaf_masks(tree, mask)
    return jp.min(select_surface_roots(tree, roots, is_present), axis=1)


def intersect_shadow(tree, roots, mask, receivers):
    """Closest surface crossing of a CSG tree, as one blocker per leaf.

    `receivers` holds the shape index each shadow ray starts on. The root
    it starts on is dropped by identity, since float error can put it on
    either side of the origin.
    """
    is_present = map_leaf_masks(tree, mask)
    surface_roots = select_surface_roots(tree, roots, is_present)
    is_receiver = jp.array(collect_leaves(tree))[:, None] == receivers
    surface_roots = drop_starting_roots(surface_roots, roots, is_receiver)
    depths = jp.min(surface_roots, axis=1)
    return depths < FARAWAY, depths


def collect_leaves(tree):
    if isinstance(tree, paz.graphics.CSG):
        leaves_A = collect_leaves(tree.shape_A)
        leaves = leaves_A + collect_leaves(tree.shape_B)
    else:
        leaves = [tree]
    return leaves


def map_leaf_masks(tree, mask):
    leaves = collect_leaves(tree)
    return dict(zip(leaves, mask[jp.array(leaves)]))


def drop_starting_roots(surface_roots, roots, is_receiver):
    distances = jp.where(is_root(roots), jp.abs(roots), jp.inf)
    is_start = distances == jp.min(distances, axis=1, keepdims=True)
    is_start = is_start & jp.expand_dims(is_receiver, 1)
    return jp.where(is_start, FARAWAY, surface_roots)


# Keeps the roots where the tree changes state. Comparing just before with
# just after a root lets leaves with flush faces agree on it.
def select_surface_roots(tree, roots, is_present):
    depths = roots.reshape(-1, 1, 1, roots.shape[-1])
    is_inside_before = count_is_odd((roots >= depths) & is_root(roots))
    is_inside_after = count_is_odd((roots > depths) & is_root(roots))
    before = evaluate_leaves(tree, is_inside_before, is_present)
    after = evaluate_leaves(tree, is_inside_after, is_present)
    depths = depths[:, 0, 0]
    is_surface = (before != after) & is_ahead(depths)
    return jp.where(is_surface, depths, FARAWAY).reshape(roots.shape)


# `FARAWAY` marks a row without a root. A half-space ends at infinity,
# which is a root like any other.
def is_root(roots):
    return roots != FARAWAY


# A ray is inside a shape while an odd number of its roots are still ahead.
def count_is_odd(is_beyond):
    return jp.sum(is_beyond, axis=2) % 2 == 1


def evaluate_leaves(tree, is_inside, is_present):
    leaves = collect_leaves(tree)
    is_inside = dict(zip(leaves, jp.moveaxis(is_inside, 1, 0)))
    return evaluate(tree, is_inside, is_present)[0]


# Returns whether rays are inside `tree`, and whether `tree` is present.
def evaluate(tree, is_inside, is_present):
    if isinstance(tree, paz.graphics.CSG):
        state_A = evaluate(tree.shape_A, is_inside, is_present)
        state_B = evaluate(tree.shape_B, is_inside, is_present)
        state = combine(tree.operation, state_A, state_B)
    else:
        state = is_inside[tree] & is_present[tree], is_present[tree]
    return state


# A masked operand is skipped rather than treated as empty, so masking a
# leaf renders the same tree with that leaf removed.
def combine(operation, state_A, state_B):
    (inside_A, has_A), (inside_B, has_B) = state_A, state_B
    union = inside_A | inside_B
    intersection = (inside_A | ~has_A) & (inside_B | ~has_B)
    intersection = intersection & (has_A | has_B)
    cases = [union, intersection, inside_A & ~inside_B]
    operations = UNION, INTERSECTION, DIFFERENCE
    is_case = [operation == case for case in operations]
    is_inside = jp.select(is_case, cases, False)
    is_present = jp.where(operation == DIFFERENCE, has_A, has_A | has_B)
    return is_inside, is_present
