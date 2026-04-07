from functools import partial
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jp


class Trace(NamedTuple):
    losses: object
    metrics: object
    stop_step: object


class Metrics(NamedTuple):
    trace: object
    steps: object
    arg: int
    period: int
    default: object
    values: object
    step: int


def _validate_metrics(metrics, metrics_every):
    if not isinstance(metrics_every, Integral):
        raise TypeError("`metrics_every` must be an integer.")
    if metrics_every < 1:
        raise ValueError("`metrics_every` must be >= 1.")
    if (metrics is None) and (metrics_every != 1):
        raise ValueError("`metrics_every` requires `metrics`.")


def MetricsState(metrics, params, max_steps, metrics_every):
    size = _num_metric_slots(metrics, max_steps, metrics_every)
    steps = jp.zeros((size,), dtype=jp.int32)
    trace = _build_metrics_trace(metrics, params, size)
    default = _build_metric_values(metrics, params)
    values = default
    return Metrics(trace, steps, 0, metrics_every, default, values, 0)


def _num_metric_slots(metrics, max_steps, metrics_every):
    if metrics is None:
        num_metric_slots = 0
    else:
        num_metric_slots = (max_steps + metrics_every - 1) // metrics_every
    return num_metric_slots


def _build_metrics_trace(metrics, params, size):
    if metrics is None:
        return {}
    metric_values = _build_metric_values(metrics, params)
    initialize_leaf = partial(_initialize_metric_trace_leaf, size=size)
    return jax.tree.map(initialize_leaf, metric_values)


def _build_metric_values(metrics, params):
    if metrics is None:
        return {}
    return metrics(params)


def _initialize_metric_trace_leaf(leaf, size):
    return jp.zeros((size, *leaf.shape), dtype=leaf.dtype)


def _update_metrics(metrics_state, metrics, params, step_arg):
    if metrics is None:
        return metrics_state

    def compute_metrics():
        metric_values = metrics(params)
        update_leaf = partial(_update_trace_leaf, metric_arg=metrics_state.arg)
        trace = jax.tree.map(update_leaf, metrics_state.trace, metric_values)
        steps = metrics_state.steps.at[metrics_state.arg].set(step_arg + 1)
        arg = metrics_state.arg + 1
        step = step_arg + 1
        return metrics_state._replace(
            trace=trace, steps=steps, arg=arg, values=metric_values, step=step
        )

    def keep_metrics():
        return metrics_state

    do_compute_metrics = _do_compute_metrics(step_arg, metrics_state.period)
    return jax.lax.cond(do_compute_metrics, compute_metrics, keep_metrics)


def _update_trace_leaf(trace_leaf, value_leaf, metric_arg):
    return trace_leaf.at[metric_arg].set(value_leaf)


def _do_compute_metrics(step_arg, metrics_every):
    return jp.mod(step_arg, metrics_every) == 0


def trim_trace(trace):
    num_steps = int(jax.device_get(trace.stop_step))
    losses = _trim_losses(trace.losses, num_steps)
    num_metric_steps = int(jax.device_get(_count_metric_steps(trace)))
    trim_sparse = partial(_trim_sparse_trace_leaf, num_steps=num_metric_steps)
    metrics_trace = jax.tree.map(trim_sparse, trace.metrics.trace)
    metric_steps = trace.metrics.steps[:num_metric_steps]
    args = (
        metrics_trace,
        metric_steps,
        trace.metrics.arg,
        trace.metrics.period,
        trace.metrics.default,
        trace.metrics.values,
        trace.metrics.step,
    )
    metrics_state = Metrics(*args)
    return Trace(losses, metrics_state, trace.stop_step)


def _count_metric_steps(trace):
    is_initialized = trace.metrics.steps > 0
    is_before_stop = trace.metrics.steps <= trace.stop_step
    return jp.sum(is_initialized & is_before_stop)


def _trim_losses(losses, num_steps):
    return losses[:num_steps]


def _trim_sparse_trace_leaf(leaf, num_steps):
    return leaf[:num_steps]
