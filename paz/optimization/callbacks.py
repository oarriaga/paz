import jax

import paz.utils.progressbar as progressbar


def TraceParameters(parameters_trace):
    def callback(_step_arg, parameters, _loss, _metrics):
        parameters_trace.append(parameters)

    return callback


def _build_callbacks(callbacks, max_steps, verbose):
    callbacks = () if callbacks is None else tuple(callbacks)
    if verbose:
        callbacks += (_build_progress_callback(max_steps),)
    return callbacks


def _build_progress_callback(max_steps):
    start_time = None

    def callback(step_arg, _params, loss, metrics):
        nonlocal start_time
        if step_arg == 1:
            start_time = progressbar.start()
        description = _build_progress_description(loss, metrics)
        progressbar.print_bar(step_arg, max_steps, start_time, description)

    return callback


def _build_progress_description(loss, metrics):
    parts = [_format_progress_value("loss", loss)]
    for name, value in metrics.values.items():
        part = _format_progress_value(name, value)
        if part is not None:
            parts.append(part)
    return "minimize " + " ".join(parts)


def _format_progress_value(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return f"{name}={value:.4g}"


def _run_callbacks(callbacks, step_arg, params, loss, metrics):
    if len(callbacks) == 0:
        return

    def observe(step_arg, params, loss, metrics):
        for callback in callbacks:
            callback(step_arg, params, loss, metrics)

    args = (step_arg, params, loss, metrics)
    jax.debug.callback(observe, *args, ordered=True)
