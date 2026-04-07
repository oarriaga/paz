import contextvars
import functools
import inspect
import sys
import time as pytime
from dataclasses import dataclass, field

import jax


_NAMESPACE_RULES = {
    "paz.backend": False,
    "paz.models": True,
    "paz.applications": True,
}
_WRAPPED_FLAG = "__paz_timer_wrapped__"
_EVENT_NAME_FLAG = "__paz_timer_event_name__"

_current_recorder = contextvars.ContextVar("current_recorder", default=None)
_current_depth = contextvars.ContextVar("current_depth", default=0)
_suppressed_event_name = contextvars.ContextVar("suppressed_event_name", default=None)
_wrapped_values = {}


@dataclass
class TimerEvent:
    name: str
    duration_ms: float
    depth: int
    kind: str


@dataclass
class TimerRecorder:
    root_name: str
    policy: str
    events: list = field(default_factory=list)
    total_ms: float = 0.0

    def record(self, name, duration_ms, depth, kind):
        self.events.append(TimerEvent(name, duration_ms, depth, kind))

    def build_report(self, max_depth=1, min_total_ms=0.05, top_k=None):
        summaries = _build_summaries(self.events, max_depth, min_total_ms, top_k)
        width = _report_width(self.root_name, summaries)
        lines = [f"{self.root_name:<{width}} total={_format_duration_ms(self.total_ms)}"]
        for summary in summaries:
            lines.append(_format_summary_line(summary, width))
        return "\n".join(lines)


def _build_summaries(events, max_depth, min_total_ms, top_k):
    summaries_by_name = {}
    for event in events:
        if event.kind == "root" or _is_hidden_depth(event.depth, max_depth):
            continue
        summary = summaries_by_name.get(event.name)
        if summary is None:
            summary = _new_summary(event.name)
            summaries_by_name[event.name] = summary
        _update_summary(summary, event)
    summaries = [summary for summary in summaries_by_name.values()
                 if summary["total_ms"] >= min_total_ms]
    summaries.sort(key=lambda summary: (-summary["total_ms"], summary["name"]))
    return summaries if top_k is None else summaries[:top_k]


def _new_summary(name):
    return {"name": name, "count": 0, "total_ms": 0.0, "max_ms": 0.0}


def _update_summary(summary, event):
    summary["count"] += 1
    summary["total_ms"] += event.duration_ms
    summary["max_ms"] = max(summary["max_ms"], event.duration_ms)


def _is_hidden_depth(depth, max_depth):
    return max_depth is not None and depth > max_depth


def _report_width(root_name, summaries):
    labels = [_display_name(summary["name"]) for summary in summaries]
    return max([len(root_name), *[len(label) for label in labels]])


def _format_summary_line(summary, width):
    average_ms = summary["total_ms"] / summary["count"]
    name = _display_name(summary["name"])
    return (
        f"  {name:<{width}} count={summary['count']} "
        f"total={_format_duration_ms(summary['total_ms'])} "
        f"avg={_format_duration_ms(average_ms)} "
        f"max={_format_duration_ms(summary['max_ms'])}"
    )


def _display_name(name):
    return name.removesuffix(".returned")


def _root_label_from_event(name):
    return _display_name(name).rsplit(".", 1)[-1]


def _format_duration_ms(duration_ms):
    color = _duration_color(duration_ms)
    return f"{color}{duration_ms:.2f} ms\033[0m"


def _duration_color(duration_ms):
    if duration_ms < 50.0:
        return "\033[92m"
    if duration_ms < 200.0:
        return "\033[93m"
    return "\033[91m"


def _is_function_value(value):
    return (
        inspect.isfunction(value)
        or inspect.ismethod(value)
        or isinstance(value, functools.partial)
    )


def _find_namespace_rule(module_name):
    for prefix, wrap_returned in _NAMESPACE_RULES.items():
        if module_name == prefix or module_name.startswith(f"{prefix}."):
            return prefix, wrap_returned


def _build_event_name(module_name, attr_name):
    return f"{module_name.replace('paz.', '', 1)}.{attr_name}"


def _find_root_name(function):
    event_name = getattr(function, _EVENT_NAME_FLAG, None)
    if event_name is not None:
        return _root_label_from_event(event_name)
    function_name = getattr(function, "__name__", None)
    if function_name and function_name != "<lambda>":
        return function_name
    return function.__class__.__name__


def _validate_policy(policy):
    if policy not in {"auto", "jax", "none"}:
        raise ValueError(f"Unknown policy: {policy}")


def _force_readiness(value, policy):
    if policy == "none":
        return value
    if policy == "jax":
        return jax.block_until_ready(value)
    try:
        return jax.block_until_ready(value)
    except Exception:
        return value


def _mark_wrapper(wrapper, event_name):
    setattr(wrapper, _WRAPPED_FLAG, True)
    setattr(wrapper, _EVENT_NAME_FLAG, event_name)


def _wrap_returned_value(value, event_name):
    if not _is_function_value(value):
        return value
    if getattr(value, _WRAPPED_FLAG, False):
        return value
    return _build_wrapper(value, f"{event_name}.returned", "returned_callable")


def _event_depth(kind, parent_depth):
    return parent_depth if kind == "returned_callable" else parent_depth + 1


def _run_timed_call(function, args, kwargs, event_name, kind, recorder):
    parent_depth = _current_depth.get()
    next_depth = _event_depth(kind, parent_depth)
    depth_token = _current_depth.set(next_depth)
    start_time = pytime.perf_counter()
    try:
        return _force_readiness(function(*args, **kwargs), recorder.policy)
    finally:
        duration_ms = (pytime.perf_counter() - start_time) * 1000.0
        recorder.record(event_name, duration_ms, parent_depth + 1, kind)
        _current_depth.reset(depth_token)


def _should_skip_root_event(event_name):
    suppressed_name = _suppressed_event_name.get()
    return suppressed_name == event_name and _current_depth.get() == 0


def _run_wrapped_call(function, args, kwargs, event_name, kind, wrap_returned):
    recorder = _current_recorder.get()
    if recorder is None:
        value = function(*args, **kwargs)
    elif _should_skip_root_event(event_name):
        value = function(*args, **kwargs)
    else:
        value = _run_timed_call(function, args, kwargs, event_name, kind, recorder)
    return _wrap_returned_value(value, event_name) if wrap_returned else value


def _build_wrapper(function, event_name, kind, wrap_returned=False):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return _run_wrapped_call(
            function, args, kwargs, event_name, kind, wrap_returned
        )

    _mark_wrapper(wrapper, event_name)
    return wrapper


def _should_wrap_value(module, module_name, value, prefix):
    if not _is_function_value(value) or getattr(value, _WRAPPED_FLAG, False):
        return False
    value_module = getattr(value, "__module__", "")
    if not value_module.startswith(prefix):
        return False
    return hasattr(module, "__path__") or value_module == module_name


def _wrap_module_values(module_name, prefix, wrap_returned):
    module = sys.modules.get(module_name)
    if module is None:
        return
    for attr_name, value in tuple(vars(module).items()):
        if attr_name.startswith("_"):
            continue
        if not _should_wrap_value(module, module_name, value, prefix):
            continue
        wrapper = _wrapped_values.get(id(value))
        if wrapper is None:
            wrapper = _build_wrapper(
                value, _build_event_name(module_name, attr_name), "function",
                wrap_returned,
            )
            _wrapped_values[id(value)] = wrapper
        setattr(module, attr_name, wrapper)


def _wrap_loaded_namespaces():
    module_rules = []
    for module_name in sys.modules:
        rule = _find_namespace_rule(module_name)
        if rule is not None:
            module_rules.append((module_name.count("."), module_name, *rule))
    module_rules.sort()
    for _, module_name, prefix, wrap_returned in module_rules:
        _wrap_module_values(module_name, prefix, wrap_returned)


def _run_root_call(function, wrapper, policy, max_depth, min_total_ms, top_k,
                   print_summary, store, root_name, args, kwargs):
    recorder = TimerRecorder(root_name, policy)
    wrapper.last_timer = recorder if store else None
    event_name = getattr(function, _EVENT_NAME_FLAG, None)
    start_time = pytime.perf_counter()
    recorder_token = _current_recorder.set(recorder)
    depth_token = _current_depth.set(0)
    event_token = _suppressed_event_name.set(event_name)
    error = None
    try:
        return _force_readiness(function(*args, **kwargs), policy)
    except Exception as exception:
        error = exception
        raise
    finally:
        recorder.total_ms = (pytime.perf_counter() - start_time) * 1000.0
        recorder.record(root_name, recorder.total_ms, 0, "root")
        _current_depth.reset(depth_token)
        _current_recorder.reset(recorder_token)
        _suppressed_event_name.reset(event_token)
        if print_summary and error is None:
            print(recorder.build_report(max_depth, min_total_ms, top_k))


def time(
    function,
    policy="auto",
    max_depth=1,
    min_total_ms=0.05,
    top_k=None,
    print_summary=True,
    store=True,
):
    _validate_policy(policy)
    _wrap_loaded_namespaces()
    root_name = _find_root_name(function)

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        _wrap_loaded_namespaces()
        return _run_root_call(
            function, wrapper, policy, max_depth, min_total_ms, top_k,
            print_summary, store, root_name, args, kwargs,
        )

    wrapper.last_timer = None
    return wrapper


_wrap_loaded_namespaces()
