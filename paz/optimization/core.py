from functools import partial
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jp
import optax
import optax.tree_utils as otu

def minimize(parameters, loss_fn, optimizer, max_steps, tolerance, metrics=None, metrics_every=1, trace=False):  # fmt: skip
    _validate_metrics(metrics, metrics_every)
    optimizer = optax.with_extra_args_support(optimizer)
    state = optimizer.init(parameters)
    value_grad_fn = _build_value_grad_fn(loss_fn, state)
    args = (parameters, state, loss_fn, optimizer, max_steps, tolerance)
    if trace:
        results = _minimize_with_trace(*args, value_grad_fn, metrics, metrics_every)  # fmt: skip
    else:
        results = _minimize(*args, value_grad_fn, metrics, metrics_every)
    return results


def _minimize(params, state, loss_fn, optimizer, max_steps, tolerance, value_grad_fn, metrics, metrics_every):  # fmt: skip
    size = _num_metric_slots(metrics, max_steps, metrics_every)
    metrics_state = _build_metrics_state(metrics, params, size)

    def step(carry):
        params, state, step_arg, has_met_criteria, metrics_state = carry
        args = (params, state, loss_fn, optimizer, tolerance, value_grad_fn)
        _, params, state, has_met_criteria = _gradient_step(*args)
        args = (metrics_state, metrics, params, step_arg, metrics_every)
        metrics_state = _update_metrics_trace(*args)
        carry = (params, state, step_arg + 1, has_met_criteria, metrics_state)
        return carry

    def cond(carry):
        _, _, step_arg, has_met_criteria, _ = carry
        has_steps_remaining = step_arg < max_steps
        return has_steps_remaining & jp.logical_not(has_met_criteria)

    carry = (params, state, 0, False, metrics_state)
    carry = jax.lax.while_loop(cond, step, carry)
    params, _, step_arg, has_met_criteria, metrics_state = carry
    history = _build_metrics_history(metrics, metrics_state, step_arg)
    return has_met_criteria, params, history


def _minimize_with_trace(params, state, loss_fn, optimizer, max_steps, tolerance, value_grad_fn, metrics, metrics_every):  # fmt: skip
    size = _num_metric_slots(metrics, max_steps, metrics_every)
    metrics_state = _build_metrics_state(metrics, params, size)

    def dummy_step(carry, _):
        params, _, _, _, _ = carry
        return carry, (-1.0, params)

    def gradient_step(carry, step_arg):
        params, state, _, stop_step, metrics_state = carry
        args = (params, state, loss_fn, optimizer, tolerance, value_grad_fn)
        loss, params, state, has_met_criteria = _gradient_step(*args)
        args = (stop_step, max_steps, has_met_criteria, step_arg)
        stop_step = _update_stop_step(*args)
        args = (metrics_state, metrics, params, step_arg, metrics_every)
        metrics_state = _update_metrics_trace(*args)
        carry = (params, state, has_met_criteria, stop_step, metrics_state)
        return carry, (loss, params)

    def step(carry, step_arg):
        _, _, has_met_criteria, _, _ = carry
        args = (has_met_criteria, dummy_step, gradient_step, carry, step_arg)
        return jax.lax.cond(*args)

    carry = (params, state, False, max_steps, metrics_state)
    steps = jp.arange(max_steps)
    carry, trace = jax.lax.scan(step, carry, steps)
    params, _, has_met_criteria, stop_step, metrics_state = carry
    losses, params_trace = trace
    trace = Trace(losses, params_trace, metrics_state, stop_step)
    return has_met_criteria, params, trace


class Trace(NamedTuple):
    losses: object
    parameters: object
    metrics: object
    stop_step: object


class Metrics(NamedTuple):
    trace: object
    steps: object
    arg: int


def _update_stop_step(stop_step, max_steps, has_met_criteria, step_arg):
    has_not_stopped_yet = stop_step == max_steps  # stop_step is still default
    should_record_stop_step = has_not_stopped_yet & has_met_criteria
    completed_steps = step_arg + 1
    return jp.where(should_record_stop_step, completed_steps, stop_step)


def _gradient_step(params, state, loss_fn, optimizer, tolerance, value_grad_fn):
    loss, gradients = value_grad_fn(params, state)
    kwargs = {"value": loss, "grad": gradients, "value_fn": loss_fn}
    delta, state = optimizer.update(gradients, state, params, **kwargs)
    params = optax.apply_updates(params, delta)
    has_met_criteria = jp.logical_not(_is_gradient_norm_high(gradients, tolerance))  # fmt: skip
    return loss, params, state, has_met_criteria


def _is_gradient_norm_high(gradients, tolerance):
    gradient_norm = otu.tree_norm(gradients)
    return gradient_norm >= tolerance


def _build_value_grad_fn(loss_fn, state):
    if _stores_value_and_grad(state):
        value_grad = optax.value_and_grad_from_state(loss_fn)
        return lambda params, state: value_grad(params, state=state)
    value_grad = jax.value_and_grad(loss_fn)
    return lambda params, _: value_grad(params)


def _stores_value_and_grad(state):
    value = otu.tree_get(state, "value")
    gradient = otu.tree_get(state, "grad")
    return (value is not None) and (gradient is not None)


def _validate_metrics(metrics, metrics_every):
    if not isinstance(metrics_every, Integral):
        raise TypeError("`metrics_every` must be an integer.")
    if metrics_every < 1:
        raise ValueError("`metrics_every` must be >= 1.")
    if (metrics is None) and (metrics_every != 1):
        raise ValueError("`metrics_every` requires `metrics`.")


def _num_metric_slots(metrics, max_steps, metrics_every):
    if metrics is None:
        num_metric_slots = 0
    else:
        num_metric_slots = (max_steps + metrics_every - 1) // metrics_every
    return num_metric_slots


def _build_metrics_state(metrics, params, size):
    steps = jp.zeros((size,), dtype=jp.int32)
    trace = _build_metrics_trace(metrics, params, size)
    return Metrics(trace, steps, 0)


def _build_metrics_history(metrics, metrics_state, stop_step):
    if metrics is None:
        return None
    return Trace(None, None, metrics_state, stop_step)


def _build_metrics_trace(metrics, params, size):
    if metrics is None:
        return {}
    metric_values = metrics(params)
    initialize_leaf = partial(_initialize_metric_trace_leaf, size=size)
    return jax.tree.map(initialize_leaf, metric_values)


def _initialize_metric_trace_leaf(leaf, size):
    return jp.zeros((size, *leaf.shape), dtype=leaf.dtype)


def _update_metrics_trace(metrics_state, metrics, params, step_arg, metrics_every):  # fmt: skip
    if metrics is None:
        return metrics_state

    def compute_metrics():
        metric_values = metrics(params)
        update_leaf = partial(_update_metric_trace_leaf, metric_arg=metrics_state.arg)  # fmt: skip
        trace = jax.tree.map(update_leaf, metrics_state.trace, metric_values)
        steps = metrics_state.steps.at[metrics_state.arg].set(step_arg + 1)
        return Metrics(trace, steps, metrics_state.arg + 1)

    def keep_metrics():
        return metrics_state

    do_compute_metrics = _do_compute_metrics(step_arg, metrics_every)
    return jax.lax.cond(do_compute_metrics, compute_metrics, keep_metrics)


def _update_metric_trace_leaf(trace_leaf, value_leaf, metric_arg):
    return trace_leaf.at[metric_arg].set(value_leaf)


def _do_compute_metrics(step_arg, metrics_every):
    return jp.mod(step_arg, metrics_every) == 0


def trim_trace(trace):
    num_steps = int(jax.device_get(trace.stop_step))
    losses = _trim_losses(trace.losses, num_steps)
    params_trace = _trim_parameters(trace.parameters, num_steps)
    num_metric_steps = int(jax.device_get(_count_metric_steps(trace)))
    trim_sparse = partial(_trim_sparse_trace_leaf, num_steps=num_metric_steps)
    metrics_trace = jax.tree.map(trim_sparse, trace.metrics.trace)
    metric_steps = trace.metrics.steps[:num_metric_steps]
    metrics_state = Metrics(metrics_trace, metric_steps, trace.metrics.arg)
    return Trace(losses, params_trace, metrics_state, trace.stop_step)


def _count_metric_steps(trace):
    is_initialized = trace.metrics.steps > 0
    is_before_stop = trace.metrics.steps <= trace.stop_step
    return jp.sum(is_initialized & is_before_stop)


def _trim_dense_trace_leaf(leaf, num_steps):
    return leaf[:num_steps]


def _trim_losses(losses, num_steps):
    if losses is None:
        return None
    return losses[:num_steps]


def _trim_parameters(parameters, num_steps):
    if parameters is None:
        return None
    trim_dense = partial(_trim_dense_trace_leaf, num_steps=num_steps)
    return jax.tree.map(trim_dense, parameters)


def _trim_sparse_trace_leaf(leaf, num_steps):
    return leaf[:num_steps]
