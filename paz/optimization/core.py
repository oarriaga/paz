import jax
import jax.numpy as jp
import optax
import optax.tree_utils as otu

def minimize(parameters, loss_fn, optimizer, max_steps, tolerance, metrics=None, trace=False):  # fmt: skip
    if (metrics is not None) and not trace:
        raise ValueError("`metrics` requires `trace=True`.")
    value_grad_fn = optax.value_and_grad_from_state(loss_fn)
    args = (parameters, loss_fn, optimizer, max_steps, tolerance, value_grad_fn)
    return _minimize_with_trace(*args, metrics) if trace else _minimize(*args)


def _minimize(params, loss_fn, optimizer, max_steps, tolerance, value_grad_fn):

    def step(carry):
        params, state, step_arg, has_met_criteria = carry
        args = (params, state, loss_fn, optimizer, tolerance, value_grad_fn)
        _, params, state, has_met_criteria = _gradient_step(*args)
        return (params, state, step_arg + 1, has_met_criteria)

    def cond(carry):
        _, _, step_arg, has_met_criteria = carry
        has_steps_remaining = step_arg < max_steps
        return has_steps_remaining & jp.logical_not(has_met_criteria)

    carry = (params, optimizer.init(params), 0, False)
    params, _, _, has_met_criteria = jax.lax.while_loop(cond, step, carry)
    return has_met_criteria, params, None


def _minimize_with_trace(params, loss_fn, optimizer, max_steps, tolerance, value_grad_fn, metrics):  # fmt: skip
    metric_zeros = _build_metric_zeros(metrics, params)

    def dummy_step(carry, _):
        params, _, _, _ = carry
        return carry, (-1.0, params, metric_zeros)

    def gradient_step(carry, step_arg):
        params, state, _, stop_step = carry
        args = (params, state, loss_fn, optimizer, tolerance, value_grad_fn)
        loss, params, state, has_met_criteria = _gradient_step(*args)
        args = (stop_step, max_steps, has_met_criteria, step_arg)
        stop_step = _update_stop_step(*args)
        metric_values = {} if metrics is None else metrics(params)
        carry = (params, state, has_met_criteria, stop_step)
        return carry, (loss, params, metric_values)

    def step(carry, step_arg):
        _, _, has_met_criteria, _ = carry
        args = (has_met_criteria, dummy_step, gradient_step, carry, step_arg)
        return jax.lax.cond(*args)

    carry = (params, optimizer.init(params), False, max_steps)
    steps = jp.arange(max_steps)
    carry, trace = jax.lax.scan(step, carry, steps)
    params, _, has_met_criteria, stop_step = carry
    trace = _trim_trace(trace, stop_step)
    return has_met_criteria, params, trace


def _update_stop_step(stop_step, max_steps, has_met_criteria, step_arg):
    has_not_stopped_yet = stop_step == max_steps
    should_record_stop_step = has_not_stopped_yet & has_met_criteria
    completed_steps = step_arg + 1
    return jp.where(should_record_stop_step, completed_steps, stop_step)


def _gradient_step(params, state, loss_fn, optimizer, tolerance, value_grad_fn):
    loss, gradients = value_grad_fn(params, state=state)
    kwargs = {"value": loss, "grad": gradients, "value_fn": loss_fn}
    delta, state = optimizer.update(gradients, state, params, **kwargs)
    params = optax.apply_updates(params, delta)
    has_met_criteria = jp.logical_not(_is_gradient_norm_high(state, tolerance))
    return loss, params, state, has_met_criteria


def _is_gradient_norm_high(state, tolerance):
    gradient = otu.tree_get(state, "grad")
    gradient_norm = otu.tree_norm(gradient)
    return gradient_norm >= tolerance


def _build_metric_zeros(metrics, params):
    if metrics is None:
        return {}
    metric_values = metrics(params)
    return jax.tree.map(jp.zeros_like, metric_values)


def _trim_trace(trace, stop_step):
    losses, params_trace, metrics_trace = trace
    num_steps = int(jax.device_get(stop_step))
    losses = losses[:num_steps]
    params_trace = jax.tree.map(lambda leaf: leaf[:num_steps], params_trace)
    metrics_trace = jax.tree.map(lambda leaf: leaf[:num_steps], metrics_trace)
    return losses, params_trace, metrics_trace
