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
    indices = jp.searchsorted(knot_times, query_times, side="right") - 1
    indices = jp.clip(indices, 0, len(knot_times) - 1)
    return knots[indices]


@partial(jax.vmap, in_axes=(None, None, 0))
def interpolate_linear(query_times, knot_times, knots):
    """Linear spline interpolation."""
    right = find_right_knot(query_times, knot_times)
    width = (knot_times[right] - knot_times[right - 1])[:, None]
    position = (query_times - knot_times[right - 1])[:, None] / width
    values = knots[right - 1] + position * (knots[right] - knots[right - 1])
    return clamp_interpolation(query_times, knot_times, knots, values)


@partial(jax.vmap, in_axes=(None, None, 0))
def interpolate_cubic(query_times, knot_times, knots):
    """Cubic Hermite interpolation with finite-difference knot slopes."""
    if len(knot_times) == 1:
        return jp.broadcast_to(knots[0], (len(query_times), knots.shape[-1]))
    knot_slopes = compute_cubic_slopes(knot_times, knots)
    values = evaluate_cubic_segment(query_times, knot_times, knots, knot_slopes)
    return clamp_interpolation(query_times, knot_times, knots, values)


def find_right_knot(query_times, knot_times):
    index = jp.searchsorted(knot_times, query_times, side="right")
    return jp.clip(index, 1, len(knot_times) - 1)


def clamp_interpolation(query_times, knot_times, knots, values):
    before = (query_times <= knot_times[0])[:, None]
    after = (query_times >= knot_times[-1])[:, None]
    values = jp.where(before, knots[0], values)
    return jp.where(after, knots[-1], values)


def evaluate_cubic_segment(query_times, knot_times, knots, knot_slopes):
    """Evaluate the Hermite cubic segment containing each query time."""
    right = find_right_knot(query_times, knot_times)
    position, width = compute_segment_position(query_times, knot_times, right)
    weights = compute_hermite_weights(position)
    values = compute_hermite_values(knots, right, weights)
    slopes = compute_hermite_slopes(knot_slopes, right, weights)
    return values + width * slopes


def compute_segment_position(query_times, knot_times, right):
    width = (knot_times[right] - knot_times[right - 1])[:, None]
    offset = (query_times - knot_times[right - 1])[:, None]
    return offset / width, width


def compute_hermite_weights(position):
    start_value = 2 * position**3 - 3 * position**2 + 1
    end_value = -2 * position**3 + 3 * position**2
    start_slope = position**3 - 2 * position**2 + position
    end_slope = position**3 - position**2
    return start_value, end_value, start_slope, end_slope


def compute_hermite_values(knots, right, weights):
    start_weight, end_weight = weights[:2]
    start_value, end_value = knots[right - 1], knots[right]
    return start_weight * start_value + end_weight * end_value


def compute_hermite_slopes(knot_slopes, right, weights):
    start_weight, end_weight = weights[2:]
    start_slope, end_slope = knot_slopes[right - 1], knot_slopes[right]
    return start_weight * start_slope + end_weight * end_slope


def compute_cubic_slopes(knot_times, knots):
    """Compute one-sided endpoint slopes and averaged interior slopes."""
    widths = jp.diff(knot_times)
    segment_slopes = jp.diff(knots, axis=0) / widths[:, None]
    interior = 0.5 * (segment_slopes[:-1] + segment_slopes[1:])
    start = segment_slopes[0][None]
    end = segment_slopes[-1][None]
    return jp.concatenate([start, interior, end], axis=0)
