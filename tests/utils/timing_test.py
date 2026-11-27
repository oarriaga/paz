import sys
import textwrap
import types

import pytest

import paz
import paz.utils as paz_utils
import paz.utils.timing as timing


def make_module(monkeypatch, name, source):
    module = types.ModuleType(name)
    module.__file__ = f"<{name}>"
    module.__package__ = name.rpartition(".")[0]
    exec(textwrap.dedent(source), module.__dict__)
    monkeypatch.setitem(sys.modules, name, module)
    parent_name, _, child_name = name.rpartition(".")
    if parent_name:
        parent = sys.modules[parent_name]
        monkeypatch.setattr(parent, child_name, module, raising=False)
    return module


@pytest.fixture(autouse=True)
def reset_timer_state():
    timing._current_recorder.set(None)
    timing._current_depth.set(0)
    timing._suppressed_event_name.set(None)
    yield
    timing._current_recorder.set(None)
    timing._current_depth.set(0)
    timing._suppressed_event_name.set(None)


def test_time_stores_last_timer_and_root_event():
    def add_one(value):
        return value + 1

    timed = timing.time(add_one, print_summary=False)
    assert timed(2) == 3
    assert timed.last_timer.root_name == "add_one"
    assert timed.last_timer.events[-1].kind == "root"
    assert timed.last_timer.total_ms >= 0.0


def test_paz_exports_utils_time():
    assert paz.time is timing.time
    assert paz_utils.time is timing.time


def test_time_rejects_removed_legacy_arguments():
    with pytest.raises(ValueError):
        timing.time(lambda value: value + 1, False)

    with pytest.raises(TypeError):
        timing.time(lambda value: value + 1, name="adder")


def test_model_factory_wraps_returned_callable_persistently(monkeypatch, capsys):
    module = make_module(
        monkeypatch,
        "paz.models.timing_model_factory",
        """
        def helper(x):
            return x + 1

        def Factory():
            def call(x):
                return helper(x) * 2
            return call
        """,
    )
    monkeypatch.setattr(paz.models, "TimingFactory", module.Factory, raising=False)
    timing._wrap_loaded_namespaces()

    pipeline = paz.models.TimingFactory()
    assert getattr(pipeline, timing._EVENT_NAME_FLAG) == "models.TimingFactory.returned"
    timed = timing.time(pipeline, print_summary=True, min_total_ms=0.0)

    assert timed(2) == 6
    names = [event.name for event in timed.last_timer.events]
    assert "models.TimingFactory.returned" not in names
    assert "models.timing_model_factory.helper" in names

    output = capsys.readouterr().out
    assert output.splitlines()[0].startswith("TimingFactory")
    assert "models.TimingFactory" not in output
    assert "models.timing_model_factory.helper" in output


def test_time_skips_duplicate_root_event_for_wrapped_functions(monkeypatch):
    module = make_module(
        monkeypatch,
        "paz.models.timing_root_skip",
        """
        def helper(x):
            return x + 1

        def Factory():
            def call(x):
                return helper(x)
            return call
        """,
    )
    monkeypatch.setattr(paz.models, "RootSkipFactory", module.Factory, raising=False)
    timing._wrap_loaded_namespaces()

    pipeline = paz.models.RootSkipFactory()
    timed = timing.time(pipeline, print_summary=False)

    assert timed(2) == 3
    names = [event.name for event in timed.last_timer.events]
    assert "models.RootSkipFactory.returned" not in names
    assert "models.timing_root_skip.helper" in names


def test_backend_factory_does_not_wrap_returned_callable(monkeypatch):
    module = make_module(
        monkeypatch,
        "paz.backend.timing_backend_factory",
        """
        def helper(x):
            return x + 1

        def Builder():
            def call(x):
                return helper(x)
            return call
        """,
    )
    monkeypatch.setattr(paz.backend, "TimingBuilder", module.Builder, raising=False)
    timing._wrap_loaded_namespaces()

    pipeline = paz.backend.TimingBuilder()
    timed = timing.time(pipeline, print_summary=False)

    assert timed(2) == 3
    names = [event.name for event in timed.last_timer.events]
    assert "backend.TimingBuilder.returned" not in names
    assert "backend.timing_backend_factory.helper" in names


def test_cross_root_reexport_is_not_wrapped(monkeypatch):
    module = make_module(
        monkeypatch,
        "paz.models.timing_cross_root",
        """
        def Factory():
            return 1
        """,
    )
    original = module.Factory
    monkeypatch.setattr(paz.applications, "CrossRootFactory", module.Factory, raising=False)
    timing._wrap_loaded_namespaces()

    assert paz.applications.CrossRootFactory is original
    assert not getattr(paz.applications.CrossRootFactory, timing._WRAPPED_FLAG, False)


def test_wrapping_is_idempotent(monkeypatch):
    module = make_module(
        monkeypatch,
        "paz.backend.timing_idempotent",
        """
        def leaf(x):
            return x + 1
        """,
    )

    timing._wrap_loaded_namespaces()
    first_wrapper = module.leaf
    timing._wrap_loaded_namespaces()
    assert module.leaf is first_wrapper


def test_callable_objects_are_not_wrapped_as_returned_functions():
    class CallableObject:
        def __call__(self, value):
            return value + 1

    value = CallableObject()
    wrapped = timing._wrap_returned_value(value, "models.Factory")
    assert wrapped is value


def test_policy_controls_readiness(monkeypatch):
    calls = []

    def fake_ready(value):
        calls.append(value)
        if value == "bad":
            raise TypeError("not ready")
        return value

    monkeypatch.setattr(timing.jax, "block_until_ready", fake_ready)

    timed_none = timing.time(lambda: "none", policy="none", print_summary=False)
    assert timed_none() == "none"
    assert calls == []

    timed_jax = timing.time(lambda: "jax", policy="jax", print_summary=False)
    assert timed_jax() == "jax"
    assert calls == ["jax"]

    timed_auto = timing.time(lambda: "bad", policy="auto", print_summary=False)
    assert timed_auto() == "bad"
    assert calls == ["jax", "bad"]


def test_summary_renders_aggregated_rows():
    recorder = timing.TimerRecorder("root", "auto")
    recorder.total_ms = 5.0
    recorder.record("backend.image.crop", 0.10, 1, "function")
    recorder.record("backend.image.crop", 0.20, 1, "function")
    recorder.record("models.Factory.returned", 0.12, 1, "returned_callable")
    recorder.record("backend.boxes.scale", 0.40, 2, "function")

    summary = recorder.build_report(max_depth=1, min_total_ms=0.11, top_k=1)

    assert "root" in summary
    assert "backend.image.crop" in summary
    assert "count=2" in summary
    assert "models.Factory" not in summary
    assert "backend.boxes.scale" not in summary


def test_summary_colors_time_values():
    recorder = timing.TimerRecorder("root", "auto")
    recorder.total_ms = 250.0
    recorder.record("backend.image.crop", 20.0, 1, "function")
    recorder.record("backend.boxes.scale", 120.0, 1, "function")

    summary = recorder.build_report(max_depth=1, min_total_ms=0.0)

    assert "\033[91m250.00 ms\033[0m" in summary
    assert "\033[92m20.00 ms\033[0m" in summary
    assert "\033[93m120.00 ms\033[0m" in summary
