from typing import NamedTuple

import jax
import jax.numpy as jp
import optax
import optax.tree_utils as otu

import paz.utils.progressbar as progressbar
from paz.optimization.callbacks import _build_callbacks
from paz.optimization.callbacks import _run_callbacks
from paz.optimization.history import MetricsState
from paz.optimization.history import Trace
from paz.optimization.history import _update_metrics
from paz.optimization.history import _validate_metrics
from paz.optimization.stopping import _build_status
from paz.optimization.stopping import _never_stop


def minimize(parameters, loss_fn, optimizer, max_steps, stop_fn=None, metrics=None, metrics_every=1, callbacks=None, verbose=False):  # fmt: skip
    _validate_metrics(metrics, metrics_every)
    callbacks = _build_callbacks(callbacks, max_steps, verbose)
    stop_fn = _never_stop if stop_fn is None else stop_fn
    optimizer = optax.with_extra_args_support(optimizer)
    state = optimizer.init(parameters)
    value_grad_fn = _build_value_grad_fn(loss_fn, state)
    minimizer = _Minimizer(optimizer, loss_fn, value_grad_fn, stop_fn)
    metrics_state = MetricsState(metrics, parameters, max_steps, metrics_every)
    args = (parameters, state, metrics_state, minimizer, max_steps)
    args += (metrics, callbacks)
    result = _minimize(*args)
    if verbose and max_steps > 0:
        progressbar.newline()
    return result


def _minimize(
    params, state, metrics_state, minimizer, max_steps, metrics, callbacks
):

    def update_metrics(metrics_state, params_now, step_arg):
        args = (metrics_state, metrics, params_now, step_arg)
        return _update_metrics(*args)

    def step(carry):
        params_now, state, step_arg, has_to_stop, losses, metrics_state = carry
        args = (params_now, state, minimizer)
        loss, grads, params_new, state_new = _gradient_step(*args)
        metrics_state = update_metrics(metrics_state, params_now, step_arg)
        step_num = step_arg + 1
        _run_callbacks(callbacks, step_num, params_now, loss, metrics_state)
        losses = losses.at[step_arg].set(loss)
        has_to_stop = minimizer.stop_fn(step_num, params_now, loss, grads)
        args = (has_to_stop, params_now, state, params_new, state_new)
        params, state = _select_state(*args)
        return (params, state, step_num, has_to_stop, losses, metrics_state)

    def cond(carry):
        _, _, step_arg, has_to_stop, _, _ = carry
        has_steps_remaining = step_arg < max_steps
        return has_steps_remaining & jp.logical_not(has_to_stop)

    losses = jp.zeros((max_steps,))
    carry = (params, state, 0, False, losses, metrics_state)
    carry = jax.lax.while_loop(cond, step, carry)
    params, _, step_arg, has_to_stop, losses, metrics_state = carry
    status = _build_status(has_to_stop)
    history = Trace(losses, metrics_state, step_arg)
    return status, params, history


class _Minimizer(NamedTuple):
    optimizer: object
    loss_fn: object
    value_grad_fn: object
    stop_fn: object


def _gradient_step(params, state, minimizer):
    optimizer = minimizer.optimizer
    loss, gradients = minimizer.value_grad_fn(params, state)
    kwargs = {"value": loss, "grad": gradients, "value_fn": minimizer.loss_fn}
    delta, state = optimizer.update(gradients, state, params, **kwargs)
    params = optax.apply_updates(params, delta)
    return loss, gradients, params, state


def _select_state(has_to_stop, params, state, next_params, next_state):
    def keep_state():
        return params, state

    def update_state():
        return next_params, next_state

    return jax.lax.cond(has_to_stop, keep_state, update_state)


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
