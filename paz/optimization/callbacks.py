import jax

import paz.utils.progressbar as progressbar


def TraceParameters(parameters_trace):
    def callback(_step_arg, parameters, _loss, _metrics):
        parameters_trace.append(parameters)

    return callback


def _build_callbacks(callbacks):
    return () if callbacks is None else tuple(callbacks)


def _build_progress_callback(max_steps, stop_message):
    start_time = None

    def callback(step_arg, loss, metrics, learning_rate, info, has_to_stop):
        nonlocal start_time
        if step_arg == 1:
            start_time = progressbar.start()
        description = _build_progress_description(loss, metrics)
        suffix = _build_progress_suffix(learning_rate, info, has_to_stop, stop_message)  # fmt: skip
        progressbar.print_bar(step_arg, max_steps, start_time, description, width=30, suffix=suffix)  # fmt: skip

    return callback


def _build_progress_description(loss, metrics):
    parts = [_format_progress_value("loss", loss)]
    for name, value in metrics.values.items():
        part = _format_progress_value(name, value)
        if part is not None:
            parts.append(part)
    return " | ".join(parts)


def _format_progress_value(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return f"{name}={value:.4g}"


def _build_progress_suffix(learning_rate, info, has_to_stop, stop_message):
    parts = []
    part = _format_linesearch_status(learning_rate, info)
    if part is not None:
        parts.append(part)
    if bool(has_to_stop) and stop_message is not None:
        parts.append(stop_message)
    if len(parts) == 0:
        return None
    return " | ".join(parts)


def _format_linesearch_status(lr, info):
    if info is None:
        return None
    decrease_error = getattr(info, "decrease_error", None)
    curvature_error = getattr(info, "curvature_error", None)
    errors = []
    if _is_linesearch_failure(decrease_error):
        errors.append(("dec", decrease_error))
    if _is_linesearch_failure(curvature_error):
        errors.append(("curv", curvature_error))
    if len(errors) == 0:
        return None
    fields = [_format_linesearch_step(info), _format_linesearch_rate(lr)]
    fields += [_format_linesearch_error(name, value) for name, value in errors]
    return f"ls=fail({', '.join(fields)})"


def _is_linesearch_failure(value):
    if value is None:
        return False
    return float(value) > 0.0


def _format_linesearch_step(info):
    num_steps = getattr(info, "num_linesearch_steps", None)
    return f"n={int(num_steps)}"


def _format_linesearch_rate(learning_rate):
    return f"lr={float(learning_rate):.2e}"


def _format_linesearch_error(name, value):
    return f"{name}={float(value):.2e}"


def _run_progress(
    callback, step_arg, loss, metrics, learning_rate, info, has_to_stop
):
    if callback is None:
        return

    def observe(step_arg, loss, metrics, learning_rate, info, has_to_stop):
        callback(step_arg, loss, metrics, learning_rate, info, has_to_stop)

    args = (step_arg, loss, metrics, learning_rate, info, has_to_stop)
    jax.debug.callback(observe, *args, ordered=True)


def _run_callbacks(callbacks, step_arg, params, loss, metrics):
    if len(callbacks) == 0:
        return

    def observe(step_arg, params, loss, metrics):
        for callback in callbacks:
            callback(step_arg, params, loss, metrics)

    args = (step_arg, params, loss, metrics)
    jax.debug.callback(observe, *args, ordered=True)
