from functools import partial
import jax
import jax.numpy as jp


@partial(jax.vmap, in_axes=(None, None, 0))
def interpolate_zero(query_times, knot_times, knots):
    """Zero-order spline interpolation. Pick the most recent knot at or before
    the query time. So a query at 0.4 with knots at [0.0, 0.5, 1.0] outputs
    knot 0's value (10.0), even though 0.5 is geometrically closer. Always
    look-back, never look-ahead. Thus, zero-order hold is causal i.e. the
    control at time t never depends on a knot in the future.
    """
    return knots[jp.searchsorted(knot_times, query_times, side="right") - 1]


@partial(jax.vmap, in_axes=(None, None, 0))
def interpolate_linear(query_times, knot_times, knots):
    """Linear spline interpolation."""
    right = find_right_knot(query_times, knot_times)
    width = (knot_times[right] - knot_times[right - 1])[:, None]
    position = (query_times - knot_times[right - 1])[:, None] / width
    return knots[right - 1] + position * (knots[right] - knots[right - 1])


@partial(jax.vmap, in_axes=(None, None, 0))
def interpolate_cubic(query_times, knot_times, knots):
    """Natural C2 cubic spline with not-a-knot boundary conditions."""
    knot_slopes = compute_natural_cubic_slopes(knot_times, knots)
    return evaluate_cubic_segment(query_times, knot_times, knots, knot_slopes)


def find_right_knot(query_times, knot_times):
    index = jp.searchsorted(knot_times, query_times, side="right")
    return jp.clip(index, 1, len(knot_times) - 1)


def evaluate_cubic_segment(query_times, knot_times, knots, knot_slopes):
    right = find_right_knot(query_times, knot_times)
    width = (knot_times[right] - knot_times[right - 1])[:, None]
    position = (query_times - knot_times[right - 1])[:, None] / width
    weight_start_value = 2 * position**3 - 3 * position**2 + 1
    weight_end_value = -2 * position**3 + 3 * position**2
    weight_start_slope = position**3 - 2 * position**2 + position
    weight_end_slope = position**3 - position**2
    start_value, end_value = knots[right - 1], knots[right]
    start_slope, end_slope = knot_slopes[right - 1], knot_slopes[right]
    values = weight_start_value * start_value + weight_end_value * end_value
    slopes = weight_start_slope * start_slope + weight_end_slope * end_slope
    return values + width * slopes


def compute_natural_cubic_slopes(knot_times, knots):
    widths = jp.diff(knot_times)
    segment_slopes = jp.diff(knots, axis=0) / widths[:, None]
    matrix = build_natural_cubic_matrix(widths)
    rhs = build_natural_cubic_rhs(widths, segment_slopes)
    return jp.linalg.solve(matrix, rhs)


def build_natural_cubic_matrix(widths):
    start_span = widths[0:1] + widths[1:2]
    end_span = widths[-1:] + widths[-2:-1]
    interior = 2 * (widths[:-1] + widths[1:])
    main = jp.concatenate([widths[1:2], interior, widths[-2:-1]])
    upper = jp.concatenate([start_span, widths[:-1]])
    lower = jp.concatenate([widths[1:], end_span])
    return jp.diag(main) + jp.diag(upper, k=1) + jp.diag(lower, k=-1)


def build_natural_cubic_rhs(widths, segment_slopes):
    start_span, end_span = widths[0] + widths[1], widths[-1] + widths[-2]
    cross_left = widths[1:, None] * segment_slopes[:-1]
    cross_right = widths[:-1, None] * segment_slopes[1:]
    interior = 3 * (cross_left + cross_right)
    start = not_a_knot_start_rhs(widths, segment_slopes, start_span)
    end = not_a_knot_end_rhs(widths, segment_slopes, end_span)
    return jp.concatenate([start[None], interior, end[None]])


def not_a_knot_start_rhs(widths, segment_slopes, span):
    inner = widths[0] ** 2 * segment_slopes[1]
    outer = (widths[0] + 2 * span) * widths[1] * segment_slopes[0]
    return (inner + outer) / span


def not_a_knot_end_rhs(widths, segment_slopes, span):
    inner = widths[-1] ** 2 * segment_slopes[-2]
    outer = (2 * span + widths[-1]) * widths[-2] * segment_slopes[-1]
    return (inner + outer) / span
