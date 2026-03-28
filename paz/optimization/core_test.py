import jax
import pytest
import jax.numpy as jp
import optax

from paz.optimization import LBFGS
from paz.optimization import LineSearch
from paz.optimization import MAX_STEPS_REACHED
from paz.optimization import STOP_FN_MET
from paz.optimization import Trace
from paz.optimization import armijo_linesearch
from paz.optimization import grad_norm_stop
from paz.optimization import linesearch
from paz.optimization import loss_stop
from paz.optimization import minimize
from paz.optimization import optimizers
from paz.optimization import trim_trace
from paz.optimization import wolfe_linesearch


def quadratic_loss(parameters):
    return jp.sum((parameters - 3.0) ** 2)


def test_optimization_package_exports_symbols():
    assert callable(LBFGS)
    assert callable(LineSearch)
    assert MAX_STEPS_REACHED == 0
    assert STOP_FN_MET == 1
    assert callable(armijo_linesearch)
    assert callable(grad_norm_stop)
    assert callable(loss_stop)
    assert callable(trim_trace)
    assert callable(minimize)
    assert Trace is not None
    assert callable(wolfe_linesearch)
    assert hasattr(linesearch, "armijo_linesearch")
    assert hasattr(linesearch, "wolfe_linesearch")
    assert hasattr(optimizers, "LBFGS")


def test_LineSearch_returns_wolfe_linesearch():
    transform = LineSearch(5, "wolfe", False)
    assert isinstance(transform, optax.GradientTransformationExtraArgs)


def test_LineSearch_returns_armijo_linesearch():
    transform = LineSearch(5, "armijo", False)
    assert isinstance(transform, optax.GradientTransformationExtraArgs)


def test_LineSearch_rejects_invalid_criterion():
    with pytest.raises(ValueError, match="armijo"):
        LineSearch(5, "invalid", False)


def test_minimize_reduces_loss_with_LBFGS():
    parameters = jp.array([8.0, -2.0])
    linesearch_transform = LineSearch(10, "wolfe", False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)
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
    linesearch_transform = LineSearch(10, "wolfe", False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)
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
    assert history.stop_step == 2


def test_minimize_without_stop_fn_reaches_max_steps():
    parameters = jp.array([8.0, -2.0])
    optimizer = optax.adam(1e-1)
    status, _, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=2,
    )
    assert status == MAX_STEPS_REACHED
    assert history.stop_step == 2


def test_minimize_returns_loss_history():
    parameters = jp.array([8.0, -2.0])
    linesearch_transform = LineSearch(10, "wolfe", False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)
    status, fitted, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=20,
        stop_fn=grad_norm_stop(1e-4),
    )
    trimmed = trim_trace(history)
    num_steps = int(jax.device_get(history.stop_step))
    assert status == STOP_FN_MET
    assert isinstance(history, Trace)
    assert len(history.losses) == 20
    assert len(trimmed.losses) == num_steps
    assert trimmed.losses[-1] <= trimmed.losses[0]
    assert jp.allclose(fitted, jp.array([3.0, 3.0]), atol=1e-3)
    assert trimmed.metrics.trace == {}
    assert len(trimmed.metrics.steps) == 0
    assert trimmed.metrics.arg == 0


def test_trim_trace_trims_sparse_metrics_to_optimization_length():
    parameters = jp.array([8.0, -2.0])
    linesearch_transform = LineSearch(10, "wolfe", False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)
    metrics = lambda value: {"distance": jp.linalg.norm(value - 3.0)}
    status, _, history = minimize(
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
    assert len(history.losses) == 20
    assert len(history.metrics.steps) == 10
    assert history.metrics.steps[0] == 1
    assert history.metrics.arg >= 1
    assert len(trimmed.metrics.trace["distance"]) == len(trimmed.metrics.steps)
    assert trimmed.metrics.trace["distance"][-1] <= trimmed.metrics.trace["distance"][0]
    assert trimmed.metrics.arg == history.metrics.arg
    assert jp.all(trimmed.metrics.steps <= history.stop_step)
    if len(trimmed.metrics.steps) > 1:
        assert jp.all(
            (trimmed.metrics.steps[1:] - trimmed.metrics.steps[:-1]) == 2
        )


def test_minimize_is_jittable():
    parameters = jp.array([8.0, -2.0])
    linesearch_transform = LineSearch(10, "wolfe", False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)

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
    linesearch_transform = LineSearch(10, "wolfe", False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)
    metrics = lambda value: {"distance": jp.linalg.norm(value - 3.0)}

    @jax.jit
    def minimize_with_trace(parameters):
        return minimize(
            parameters,
            quadratic_loss,
            optimizer,
            max_steps=20,
            stop_fn=grad_norm_stop(1e-4),
            metrics=metrics,
            metrics_every=2,
        )

    status, fitted, history = minimize_with_trace(parameters)
    assert status == STOP_FN_MET
    assert isinstance(history, Trace)
    assert jp.allclose(fitted, jp.array([3.0, 3.0]), atol=1e-3)


def test_minimize_returns_metrics_without_trace():
    parameters = jp.array([8.0, -2.0])
    linesearch_transform = LineSearch(10, "wolfe", False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)
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
    assert len(trimmed.metrics.trace["distance"]) == len(trimmed.metrics.steps)
    assert jp.allclose(fitted, jp.array([3.0, 3.0]), atol=1e-3)


def test_minimize_with_metrics_without_trace_is_jittable():
    parameters = jp.array([8.0, -2.0])
    linesearch_transform = LineSearch(10, "wolfe", False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)
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
    assert calls[0][3] == {}


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
    assert seen_metrics[0][1] != {}
    assert seen_metrics[1][1] == {}
    assert seen_metrics[2][1] != {}
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
    linesearch_transform = LineSearch(10, "wolfe", False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)
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
    linesearch_transform = LineSearch(10, "wolfe", False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)
    with pytest.raises(ValueError, match="requires `metrics`"):
        minimize(
            parameters,
            quadratic_loss,
            optimizer,
            max_steps=20,
            metrics_every=2,
        )
