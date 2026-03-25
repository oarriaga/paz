from collections import namedtuple

import jax
import jax.numpy as jp
import optax
import optax.tree_utils as otu
from optax import scale_by_zoom_linesearch
from optax import scale_by_backtracking_linesearch

import paz.utils.progressbar as progressbar

OptimizeResult = namedtuple(
    "OptimizeResult",
    ["success", "parameters", "losses", "parameters_history"],
)


def optimize(parameters, loss_fn, optimizer, max_steps, tolerance):
    value_and_grad_fn = optax.value_and_grad_from_state(loss_fn)
    progress_callback = progressbar.show(max_steps, "optimizing", width=30)

    def is_gradient_norm_high(state):
        step_arg = otu.tree_get(state, "count")
        gradient = otu.tree_get(state, "grad")
        gradient_norm = otu.tree_l2_norm(gradient)
        is_step_zero = step_arg == 0
        is_less_than_max_steps = step_arg < max_steps
        gradient_is_still_high = gradient_norm >= tolerance
        return is_step_zero | (is_less_than_max_steps & gradient_is_still_high)

    def dummy_step(carry, step_arg):
        parameters, state, has_met_criteria = carry
        return carry, (-1.0, has_met_criteria, parameters)

    def gradient_step(carry, step_arg):
        parameters, state, has_met_criteria = carry
        loss, gradients = value_and_grad_fn(parameters, state=state)
        kwargs = {"value": loss, "grad": gradients, "value_fn": loss_fn}
        delta, state = optimizer.update(gradients, state, parameters, **kwargs)
        parameters = optax.apply_updates(parameters, delta)
        has_met_criteria = jp.logical_not(is_gradient_norm_high(state))
        carry = (parameters, state, has_met_criteria)
        return carry, (loss, has_met_criteria, parameters)

    def step(carry, step_arg):
        progress_callback(step_arg + 1)
        parameters, state, has_met_criteria = carry
        return jax.lax.cond(
            has_met_criteria, dummy_step, gradient_step, carry, step_arg
        )

    carry = (parameters, optimizer.init(parameters), False)
    steps = jp.arange(max_steps)
    carry, history = jax.lax.scan(step, carry, steps)
    progressbar.newline()
    parameters, _, has_met_criteria = carry
    losses, criteria_history, parameters_history = history
    where_criteria_met = jp.argmax(criteria_history)
    losses = losses[:where_criteria_met]
    return OptimizeResult(
        has_met_criteria, parameters, losses, parameters_history
    )


def LBFGS(learning_rate, memory_size, linesearch):
    return optax.lbfgs(learning_rate, memory_size, True, linesearch)


def LineSearch(max_line_steps, wolfe_criterion=False):
    if wolfe_criterion:
        return scale_by_zoom_linesearch(
            max_line_steps, initial_guess_strategy="one"
        )
    return scale_by_backtracking_linesearch(
        max_line_steps, store_grad=True, slope_rtol=1e-5
    )
