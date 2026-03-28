from functools import partial
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jp
import optax
import optax.tree_utils as otu

MAX_STEPS_REACHED = 0
STOP_FN_MET = 1


def grad_norm_stop(tolerance):
    def stop_fn(_step_arg, _params, _loss, gradients):
        return otu.tree_norm(gradients) < tolerance

    return stop_fn


def loss_stop(tolerance):
    def stop_fn(_step_arg, _params, loss, _gradients):
        return loss < tolerance

    return stop_fn


def minimize(parameters, loss_fn, optimizer, max_steps, stop_fn=None, metrics=None, metrics_every=1, callbacks=None):  # fmt: skip
    _validate_metrics(metrics, metrics_every)
    callbacks = () if callbacks is None else tuple(callbacks)
    stop_fn = _never_stop if stop_fn is None else stop_fn
    optimizer = optax.with_extra_args_support(optimizer)
    state = optimizer.init(parameters)
    value_grad_fn = _build_value_grad_fn(loss_fn, state)
    args = (parameters, state, loss_fn, optimizer, max_steps, stop_fn)
    args += (value_grad_fn, metrics, metrics_every, callbacks)
    return _minimize(*args)


def _minimize(params, state, loss_fn, optimizer, max_steps, stop_fn, value_grad_fn, metrics, metrics_every, callbacks):  # fmt: skip
    losses = jp.zeros((max_steps,))
    size = _num_metric_slots(metrics, max_steps, metrics_every)
    metrics_state = _build_metrics_state(metrics, params, size)
    metric_defaults = _build_metric_values(metrics, params)

    def step(carry):
        params, state, step_arg, has_met_criteria, losses, metrics_state = carry
        params_before = params
        args = (params, state, loss_fn, optimizer, value_grad_fn)
        loss, gradients, next_params, next_state = _gradient_step(*args)
        args = (metrics_state, metrics, params_before, step_arg, metrics_every, metric_defaults)  # fmt: skip
        metrics_state, metric_values, has_metrics = _update_metrics_trace(*args)
        _run_callbacks(callbacks, step_arg + 1, params_before, loss, metric_values, has_metrics)  # fmt: skip
        losses = losses.at[step_arg].set(loss)
        args = (step_arg + 1, params_before, loss, gradients)
        has_met_criteria = stop_fn(*args)
        args = (has_met_criteria, params_before, state, next_params, next_state)
        params, state = _select_step_state(*args)
        carry = (params, state, step_arg + 1, has_met_criteria, losses, metrics_state)  # fmt: skip
        return carry

    def cond(carry):
        _, _, step_arg, has_met_criteria, _, _ = carry
        has_steps_remaining = step_arg < max_steps
        return has_steps_remaining & jp.logical_not(has_met_criteria)

    carry = (params, state, 0, False, losses, metrics_state)
    carry = jax.lax.while_loop(cond, step, carry)
    params, _, step_arg, has_met_criteria, losses, metrics_state = carry
    status = _build_status(has_met_criteria)
    history = Trace(losses, metrics_state, step_arg)
    return status, params, history


class Trace(NamedTuple):
    losses: object
    metrics: object
    stop_step: object


class Metrics(NamedTuple):
    trace: object
    steps: object
    arg: int


def _gradient_step(params, state, loss_fn, optimizer, value_grad_fn):
    loss, gradients = value_grad_fn(params, state)
    kwargs = {"value": loss, "grad": gradients, "value_fn": loss_fn}
    delta, state = optimizer.update(gradients, state, params, **kwargs)
    params = optax.apply_updates(params, delta)
    return loss, gradients, params, state


def _select_step_state(has_met_criteria, params, state, next_params, next_state):  # fmt: skip

    def keep_state():
        return params, state

    def update_state():
        return next_params, next_state

    return jax.lax.cond(has_met_criteria, keep_state, update_state)


def _never_stop(_step_arg, _params, _loss, _gradients):
    return jp.array(False)


def _build_status(has_met_criteria):
    return jp.where(has_met_criteria, STOP_FN_MET, MAX_STEPS_REACHED)


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


def _update_metrics_trace(metrics_state, metrics, params, step_arg, metrics_every, metric_defaults):  # fmt: skip
    if metrics is None:
        return metrics_state, metric_defaults, False

    def compute_metrics():
        metric_values = metrics(params)
        update_leaf = partial(_update_metric_trace_leaf, metric_arg=metrics_state.arg)  # fmt: skip
        trace = jax.tree.map(update_leaf, metrics_state.trace, metric_values)
        steps = metrics_state.steps.at[metrics_state.arg].set(step_arg + 1)
        metrics_state_ = Metrics(trace, steps, metrics_state.arg + 1)
        return metrics_state_, metric_values, True

    def keep_metrics():
        return metrics_state, metric_defaults, False

    do_compute_metrics = _do_compute_metrics(step_arg, metrics_every)
    return jax.lax.cond(do_compute_metrics, compute_metrics, keep_metrics)


def _update_metric_trace_leaf(trace_leaf, value_leaf, metric_arg):
    return trace_leaf.at[metric_arg].set(value_leaf)


def _do_compute_metrics(step_arg, metrics_every):
    return jp.mod(step_arg, metrics_every) == 0


def _run_callbacks(callbacks, step_arg, params, loss, metrics, has_metrics):
    if len(callbacks) == 0:
        return

    def observe(step_arg, params, loss, metrics, has_metrics):
        metrics = metrics if bool(has_metrics) else {}
        for callback in callbacks:
            callback(step_arg, params, loss, metrics)

    args = (step_arg, params, loss, metrics, has_metrics)
    jax.debug.callback(observe, *args, ordered=True)


def trim_trace(trace):
    num_steps = int(jax.device_get(trace.stop_step))
    losses = _trim_losses(trace.losses, num_steps)
    num_metric_steps = int(jax.device_get(_count_metric_steps(trace)))
    trim_sparse = partial(_trim_sparse_trace_leaf, num_steps=num_metric_steps)
    metrics_trace = jax.tree.map(trim_sparse, trace.metrics.trace)
    metric_steps = trace.metrics.steps[:num_metric_steps]
    metrics_state = Metrics(metrics_trace, metric_steps, trace.metrics.arg)
    return Trace(losses, metrics_state, trace.stop_step)


def _count_metric_steps(trace):
    is_initialized = trace.metrics.steps > 0
    is_before_stop = trace.metrics.steps <= trace.stop_step
    return jp.sum(is_initialized & is_before_stop)


def _trim_losses(losses, num_steps):
    return losses[:num_steps]


def _trim_sparse_trace_leaf(leaf, num_steps):
    return leaf[:num_steps]
