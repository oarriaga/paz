import jax
import jax.numpy as jp

import paz
from paz.graphics.constants import DIFFERENCE, INTERSECTION, UNION
from paz.graphics.constants import EPSILON, FARAWAY
from paz.graphics.shapes import solid


def intersect(shapes, tree, rays, mask):
    """Closest surface of a CSG tree, as one row per leaf shape.

    A leaf boundary is a surface of the tree only where crossing it changes
    whether the ray is inside the tree. Each leaf reports the nearest such
    crossing, so the rows shade and shadow like any other shape row.
    """
    origins, directions = rays
    merged = merge_leaves(shapes, tree)
    enters, exits = compute_intervals(merged, rays)
    is_present = map_leaf_masks(tree, mask)
    roots = select_surface_roots(tree, enters, exits, is_present)
    depths = jp.expand_dims(jp.min(roots, axis=0), -1)
    points = origins + depths * directions
    eyes = jp.broadcast_to(-directions, points.shape)
    normals = jax.vmap(solid.compute_normals)(merged, points, eyes)
    return depths[..., 0] < FARAWAY, depths, points, normals, eyes


def intersect_shadow(shapes, tree, rays, mask, receivers):
    """Closest surface crossing of a CSG tree, as one blocker per leaf.

    `receivers` holds the shape index each shadow ray starts on. The root
    it starts on is dropped by identity, since float error can put it on
    either side of the origin.
    """
    merged = merge_leaves(shapes, tree)
    enters, exits = compute_intervals(merged, rays)
    is_present = map_leaf_masks(tree, mask)
    roots = select_surface_roots(tree, enters, exits, is_present)
    is_receiver = jp.array(collect_leaves(tree))[:, None] == receivers
    roots = drop_starting_roots(roots, enters, exits, is_receiver)
    depths = jp.min(roots, axis=0)
    return depths < FARAWAY, depths


def collect_leaves(tree):
    if isinstance(tree, paz.graphics.CSG):
        leaves_A = collect_leaves(tree.shape_A)
        leaves = leaves_A + collect_leaves(tree.shape_B)
    else:
        leaves = [tree]
    return leaves


def merge_leaves(shapes, tree):
    leaves = [shapes[leaf] for leaf in collect_leaves(tree)]
    return paz.graphics.shapes.field_merge(leaves, ["transform", "type"])


def compute_intervals(merged, rays):
    compute_interval = paz.lock(solid.compute_interval, *rays)
    return jax.vmap(compute_interval)(merged)


def map_leaf_masks(tree, mask):
    leaves = collect_leaves(tree)
    return dict(zip(leaves, mask[jp.array(leaves)]))


def drop_starting_roots(roots, enters, exits, is_receiver):
    starts_on_enter = jp.abs(enters) < jp.abs(exits)
    is_start = jp.stack([starts_on_enter, ~starts_on_enter]) & is_receiver
    return jp.where(is_start, FARAWAY, roots)


# Keeps the leaf roots, shaped `(2, L, N)`, where the tree changes state.
# Comparing just before with just after a root lets flush leaves agree.
def select_surface_roots(tree, enters, exits, is_present):
    num_leaves = len(enters)
    roots = jp.concatenate([enters, exits])[:, None]
    is_inside_before = (enters < roots) & (roots <= exits)
    is_inside_after = (enters <= roots) & (roots < exits)
    before = evaluate_leaves(tree, is_inside_before, is_present)
    after = evaluate_leaves(tree, is_inside_after, is_present)
    roots = roots[:, 0]
    is_surface = (before != after) & (roots > EPSILON) & (roots < FARAWAY)
    roots = jp.where(is_surface, roots, FARAWAY)
    return roots.reshape(2, num_leaves, -1)


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
