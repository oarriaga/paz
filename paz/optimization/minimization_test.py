import jax
import pytest
import jax.numpy as jp
import optax

from paz.optimization import LBFGS
from paz.optimization import LineSearch
from paz.optimization import MAX_STEPS_REACHED
from paz.optimization import STOP_FN_MET
from paz.optimization import Trace
from paz.optimization import grad_norm_stop
from paz.optimization import loss_stop
from paz.optimization import minimize
from paz.optimization import trim_trace


def quadratic_loss(parameters):
    return jp.sum((parameters - 3.0) ** 2)


def test_minimize_reduces_loss_with_LBFGS():
    parameters = jp.array([8.0, -2.0])
    linesearch = LineSearch(10, "wolfe")
    optimizer = LBFGS(1.0, 5, linesearch)
    status, fitted, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=20,
        stop_fn=grad_norm_stop(1e-4),
    )
    trimmed = trim_trace(history)
    assert status == STOP_FN_MET
    assert isinstance(history, Trace)
    assert len(history.losses) == 20
    assert len(trimmed.losses) == int(jax.device_get(history.stop_step))
    assert jp.allclose(fitted, jp.array([3.0, 3.0]), atol=1e-3)


def test_minimize_reduces_loss_with_adam():
    parameters = jp.array([8.0, -2.0])
    optimizer = optax.adam(1e-1)
    status, fitted, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=200,
        stop_fn=grad_norm_stop(1e-3),
    )
    trimmed = trim_trace(history)
    assert status == STOP_FN_MET
    assert isinstance(history, Trace)
    assert len(trimmed.losses) >= 1
    assert jp.allclose(fitted, jp.array([3.0, 3.0]), atol=5e-2)


def test_minimize_stops_before_update():
    parameters = jp.array([8.0, -2.0])
    optimizer = optax.adam(1e-1)
    stop_fn = lambda step_arg, params, loss, gradients: step_arg == 1
    status, fitted, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=20,
        stop_fn=stop_fn,
    )
    trimmed = trim_trace(history)
    assert status == STOP_FN_MET
    assert len(trimmed.losses) == 1
    assert jp.allclose(fitted, parameters)


def test_minimize_can_stop_on_loss():
    parameters = jp.array([8.0, -2.0])
    optimizer = optax.adam(1e-1)
    status, fitted, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=500,
        stop_fn=loss_stop(1e-2),
    )
    trimmed = trim_trace(history)
    assert status == STOP_FN_MET
    assert len(trimmed.losses) >= 1
    assert quadratic_loss(fitted) < 1e-2


def test_minimize_keeps_history_when_stop_fn_is_not_met():
    parameters = jp.array([8.0, -2.0])
    linesearch = LineSearch(10, "wolfe")
    optimizer = LBFGS(1.0, 5, linesearch)
    status, _, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=2,
        stop_fn=lambda step_arg, params, loss, gradients: False,
    )
    assert isinstance(history, Trace)
    assert status == MAX_STEPS_REACHED
    assert len(history.losses) == 2
    assert history.metrics.trace == {}
    assert len(history.metrics.steps) == 0
    assert history.metrics.arg == 0
    assert history.metrics.period == 1
    assert history.metrics.default == {}
    assert history.metrics.values == {}
    assert history.metrics.step == 0
    assert history.stop_step == 2


def test_minimize_without_stop_fn_reaches_max_steps():
    parameters = jp.array([8.0, -2.0])
    optimizer = optax.adam(1e-1)
    status, _, history = minimize(parameters, quadratic_loss, optimizer, 2)
    assert status == MAX_STEPS_REACHED
    assert history.stop_step == 2


def test_minimize_is_jittable():
    parameters = jp.array([8.0, -2.0])
    linesearch = LineSearch(10, "wolfe")
    optimizer = LBFGS(1.0, 5, linesearch)

    @jax.jit
    def minimize_jit(parameters):
        return minimize(
            parameters,
            quadratic_loss,
            optimizer,
            max_steps=20,
            stop_fn=grad_norm_stop(1e-4),
        )

    status, fitted, history = minimize_jit(parameters)
    assert status == STOP_FN_MET
    assert isinstance(history, Trace)
    assert jp.allclose(fitted, jp.array([3.0, 3.0]), atol=1e-3)


def test_minimize_with_adam_is_jittable():
    parameters = jp.array([8.0, -2.0])
    optimizer = optax.adam(1e-1)

    @jax.jit
    def minimize_with_adam(parameters):
        return minimize(
            parameters,
            quadratic_loss,
            optimizer,
            max_steps=200,
            stop_fn=grad_norm_stop(1e-3),
        )

    status, fitted, history = minimize_with_adam(parameters)
    assert status == STOP_FN_MET
    assert isinstance(history, Trace)
    assert jp.allclose(fitted, jp.array([3.0, 3.0]), atol=5e-2)


def test_minimize_with_metric_cadence_is_jittable():
    parameters = jp.array([8.0, -2.0])
    linesearch = LineSearch(10, "wolfe")
    optimizer = LBFGS(1.0, 5, linesearch)
    metrics = lambda value: {"distance": jp.linalg.norm(value - 3.0)}

    @jax.jit
    def minimize_with_metrics(parameters):
        return minimize(
            parameters,
            quadratic_loss,
            optimizer,
            max_steps=20,
            stop_fn=grad_norm_stop(1e-4),
            metrics=metrics,
            metrics_every=2,
        )

    status, fitted, history = minimize_with_metrics(parameters)
    assert status == STOP_FN_MET
    assert isinstance(history, Trace)
    assert jp.allclose(fitted, jp.array([3.0, 3.0]), atol=1e-3)


def test_minimize_returns_metrics_without_trace():
    parameters = jp.array([8.0, -2.0])
    linesearch = LineSearch(10, "wolfe")
    optimizer = LBFGS(1.0, 5, linesearch)
    metrics = lambda value: {"distance": jp.linalg.norm(value - 3.0)}
    status, fitted, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=20,
        stop_fn=grad_norm_stop(1e-4),
        metrics=metrics,
        metrics_every=2,
    )
    trimmed = trim_trace(history)
    assert status == STOP_FN_MET
    assert isinstance(history, Trace)
    assert history.metrics.steps[0] == 1
    assert history.metrics.period == 2
    assert "distance" in history.metrics.default
    assert "distance" in history.metrics.values
    assert len(trimmed.metrics.trace["distance"]) == len(trimmed.metrics.steps)
    assert jp.allclose(fitted, jp.array([3.0, 3.0]), atol=1e-3)


def test_minimize_with_metrics_without_trace_is_jittable():
    parameters = jp.array([8.0, -2.0])
    linesearch = LineSearch(10, "wolfe")
    optimizer = LBFGS(1.0, 5, linesearch)
    metrics = lambda value: {"distance": jp.linalg.norm(value - 3.0)}

    @jax.jit
    def minimize_with_metrics(parameters):
        return minimize(
            parameters,
            quadratic_loss,
            optimizer,
            max_steps=20,
            stop_fn=grad_norm_stop(1e-4),
            metrics=metrics,
            metrics_every=2,
        )

    status, fitted, history = minimize_with_metrics(parameters)
    assert status == STOP_FN_MET
    assert isinstance(history, Trace)
    assert jp.allclose(fitted, jp.array([3.0, 3.0]), atol=1e-3)


def test_minimize_with_default_verbose_is_silent(capsys):
    parameters = jp.array([8.0, -2.0])
    optimizer = optax.adam(1e-1)
    minimize(parameters, quadratic_loss, optimizer, max_steps=2)
    jax.effects_barrier()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_minimize_with_verbose_prints_loss(capsys):
    parameters = jp.array([8.0, -2.0])
    optimizer = optax.adam(1e-1)
    minimize(parameters, quadratic_loss, optimizer, max_steps=2, verbose=True)
    jax.effects_barrier()
    captured = capsys.readouterr()
    assert "loss=" in captured.out
    assert "minimize " not in captured.out
    assert " | " not in captured.out


def test_minimize_with_verbose_repeats_metrics(capsys):
    parameters = jp.array([8.0, -2.0])
    optimizer = optax.adam(1e-1)
    metrics = lambda value: {
        "distance": jp.linalg.norm(value - 3.0),
        "spread": jp.max(value) - jp.min(value),
    }
    minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=3,
        metrics=metrics,
        metrics_every=2,
        verbose=True,
    )
    jax.effects_barrier()
    captured = capsys.readouterr()
    assert captured.out.count("distance=") == 3
    assert captured.out.count("spread=") == 3
    assert " | distance=" in captured.out
    assert " | spread=" in captured.out


def test_minimize_with_verbose_is_jittable(capsys):
    parameters = jp.array([8.0, -2.0])
    optimizer = optax.adam(1e-1)

    @jax.jit
    def minimize_with_verbose(parameters):
        return minimize(parameters, quadratic_loss, optimizer, 2, verbose=True)

    minimize_with_verbose(parameters)
    jax.effects_barrier()
    captured = capsys.readouterr()
    assert "loss=" in captured.out
    assert "minimize " not in captured.out


def test_minimize_with_verbose_LBFGS_has_no_extra_info(capsys):
    parameters = jp.array([8.0, -2.0])
    linesearch = LineSearch(10, "wolfe")
    optimizer = LBFGS(1.0, 5, linesearch)
    minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=2,
        stop_fn=grad_norm_stop(1e-4),
        verbose=True,
    )
    jax.effects_barrier()
    captured = capsys.readouterr()
    assert "loss=" in captured.out
    assert "Iteration:" not in captured.out
    assert "Value:" not in captured.out
    assert "Gradient norm:" not in captured.out
    assert "optax.scale_by_backtracking_linesearch" not in captured.out
    assert "Backtracking linesearch failed" not in captured.out


def test_minimize_with_verbose_prints_stop_message(capsys):
    parameters = jp.array([8.0, -2.0])
    optimizer = optax.adam(1e-1)
    minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=2,
        stop_fn=loss_stop(100.0),
        verbose=True,
    )
    jax.effects_barrier()
    captured = capsys.readouterr()
    assert "] | stop=loss" in captured.out


def test_minimize_runs_callbacks():
    parameters = jp.array([8.0, -2.0])
    calls = []

    def callback(step_arg, parameters, loss, metrics):
        calls.append((step_arg, parameters, loss, metrics))

    optimizer = optax.adam(1e-1)
    status, _, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=200,
        stop_fn=grad_norm_stop(1e-3),
        callbacks=[callback],
    )
    trimmed = trim_trace(history)
    assert status == STOP_FN_MET
    assert len(calls) == len(trimmed.losses)
    assert calls[0][0] == 1
    assert calls[-1][0] == history.stop_step
    assert calls[0][3].values == {}
    assert calls[0][3].step == 0


def test_minimize_runs_callbacks_with_metrics():
    parameters = jp.array([8.0, -2.0])
    seen_metrics = []
    optimizer = optax.adam(1e-1)
    metrics = lambda value: {"distance": jp.linalg.norm(value - 3.0)}

    def callback(step_arg, parameters, loss, metrics):
        del parameters, loss
        seen_metrics.append((step_arg, metrics))

    status, _, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=200,
        stop_fn=grad_norm_stop(1e-3),
        metrics=metrics,
        metrics_every=2,
        callbacks=[callback],
    )
    assert status == STOP_FN_MET
    assert seen_metrics[0][1].step == 1
    assert seen_metrics[1][1].step == 1
    assert seen_metrics[2][1].step == 3
    assert seen_metrics[-1][0] == history.stop_step


def test_minimize_with_callbacks_is_jittable():
    parameters = jp.array([8.0, -2.0])
    calls = []

    def callback(step_arg, parameters, loss, metrics):
        del parameters, loss, metrics
        calls.append(step_arg)

    optimizer = optax.adam(1e-1)

    @jax.jit
    def minimize_with_callback(parameters):
        return minimize(
            parameters,
            quadratic_loss,
            optimizer,
            max_steps=200,
            stop_fn=grad_norm_stop(1e-3),
            callbacks=[callback],
        )

    status, fitted, history = minimize_with_callback(parameters)
    assert status == STOP_FN_MET
    assert calls[-1] == history.stop_step
    assert jp.allclose(fitted, jp.array([3.0, 3.0]), atol=5e-2)


def test_minimize_rejects_invalid_metrics_every():
    parameters = jp.array([8.0, -2.0])
    linesearch = LineSearch(10, "wolfe")
    optimizer = LBFGS(1.0, 5, linesearch)
    metrics = lambda value: {"distance": jp.linalg.norm(value - 3.0)}
    with pytest.raises(ValueError, match=">= 1"):
        minimize(
            parameters,
            quadratic_loss,
            optimizer,
            max_steps=20,
            stop_fn=grad_norm_stop(1e-4),
            metrics=metrics,
            metrics_every=0,
        )


def test_minimize_rejects_metrics_every_without_metrics():
    parameters = jp.array([8.0, -2.0])
    linesearch = LineSearch(10, "wolfe")
    optimizer = LBFGS(1.0, 5, linesearch)
    with pytest.raises(ValueError, match="requires `metrics`"):
        minimize(
            parameters,
            quadratic_loss,
            optimizer,
            max_steps=20,
            metrics_every=2,
        )
