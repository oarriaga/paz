import jax
import pytest
import jax.numpy as jp
import optax

from paz.optimization import LBFGS
from paz.optimization import LineSearch
from paz.optimization import Trace
from paz.optimization import armijo_linesearch
from paz.optimization import linesearch
from paz.optimization import minimize
from paz.optimization import optimizers
from paz.optimization import trim_trace
from paz.optimization import wolfe_linesearch


def quadratic_loss(parameters):
    return jp.sum((parameters - 3.0) ** 2)


def test_optimization_package_exports_symbols():
    assert callable(LBFGS)
    assert callable(LineSearch)
    assert callable(armijo_linesearch)
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


def test_minimize_reduces_loss_with_LBFGS_without_trace():
    parameters = jp.array([8.0, -2.0])
    linesearch_transform = LineSearch(10, "wolfe", False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)
    success, fitted, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=20,
        tolerance=1e-4,
    )
    assert bool(success)
    assert jp.allclose(fitted, jp.array([3.0, 3.0]), atol=1e-3)
    assert history is None


def test_minimize_reduces_loss_with_adam_without_trace():
    parameters = jp.array([8.0, -2.0])
    optimizer = optax.adam(1e-1)
    success, fitted, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=200,
        tolerance=1e-3,
    )
    assert bool(success)
    assert jp.allclose(fitted, jp.array([3.0, 3.0]), atol=5e-2)
    assert history is None


def test_minimize_keeps_history_when_tolerance_is_not_met():
    parameters = jp.array([8.0, -2.0])
    linesearch_transform = LineSearch(10, "wolfe", False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)
    success, _, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=2,
        tolerance=-1.0,
        trace=True,
    )
    assert isinstance(history, Trace)
    assert not bool(success)
    assert len(history.losses) == 2
    assert history.metrics.trace == {}
    assert len(history.metrics.steps) == 0
    assert history.metrics.arg == 0
    assert history.stop_step == 2


def test_minimize_returns_history_when_trace_is_enabled():
    parameters = jp.array([8.0, -2.0])
    linesearch_transform = LineSearch(10, "wolfe", False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)
    success, fitted, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=20,
        tolerance=1e-4,
        trace=True,
    )
    trimmed = trim_trace(history)
    num_steps = int(jax.device_get(history.stop_step))
    assert bool(success)
    assert isinstance(history, Trace)
    assert len(history.losses) == 20
    assert len(trimmed.losses) == num_steps
    assert trimmed.losses[-1] <= trimmed.losses[0]
    assert jp.allclose(fitted, jp.array([3.0, 3.0]), atol=1e-3)
    assert trimmed.metrics.trace == {}
    assert len(trimmed.metrics.steps) == 0
    assert trimmed.metrics.arg == 0


def test_minimize_returns_history_with_adam_trace():
    parameters = jp.array([8.0, -2.0])
    optimizer = optax.adam(1e-1)
    success, fitted, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=200,
        tolerance=1e-3,
        trace=True,
    )
    trimmed = trim_trace(history)
    assert bool(success)
    assert isinstance(history, Trace)
    assert len(trimmed.losses) >= 1
    assert trimmed.losses[-1] <= trimmed.losses[0]
    assert jp.allclose(fitted, jp.array([3.0, 3.0]), atol=5e-2)


def test_trim_trace_trims_sparse_metrics_to_optimization_length():
    parameters = jp.array([8.0, -2.0])
    linesearch_transform = LineSearch(10, "wolfe", False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)
    metrics = lambda value: {"distance": jp.linalg.norm(value - 3.0)}
    success, _, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=20,
        tolerance=1e-4,
        metrics=metrics,
        metrics_every=2,
        trace=True,
    )
    trimmed = trim_trace(history)
    assert bool(success)
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


def test_minimize_with_trace_is_jittable():
    parameters = jp.array([8.0, -2.0])
    linesearch_transform = LineSearch(10, "wolfe", False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)

    @jax.jit
    def minimize_with_trace(parameters):
        return minimize(
            parameters,
            quadratic_loss,
            optimizer,
            max_steps=20,
            tolerance=1e-4,
            trace=True,
        )

    success, fitted, history = minimize_with_trace(parameters)
    assert bool(success)
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
            tolerance=1e-3,
        )

    success, fitted, history = minimize_with_adam(parameters)
    assert bool(success)
    assert history is None
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
            tolerance=1e-4,
            metrics=metrics,
            metrics_every=2,
            trace=True,
        )

    success, fitted, history = minimize_with_trace(parameters)
    assert bool(success)
    assert isinstance(history, Trace)
    assert jp.allclose(fitted, jp.array([3.0, 3.0]), atol=1e-3)


def test_minimize_returns_metrics_without_trace():
    parameters = jp.array([8.0, -2.0])
    linesearch_transform = LineSearch(10, "wolfe", False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)
    metrics = lambda value: {"distance": jp.linalg.norm(value - 3.0)}
    success, fitted, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=20,
        tolerance=1e-4,
        metrics=metrics,
        metrics_every=2,
    )
    trimmed = trim_trace(history)
    assert bool(success)
    assert isinstance(history, Trace)
    assert history.losses is None
    assert history.parameters is None
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
            tolerance=1e-4,
            metrics=metrics,
            metrics_every=2,
        )

    success, fitted, history = minimize_with_metrics(parameters)
    assert bool(success)
    assert isinstance(history, Trace)
    assert history.losses is None
    assert history.parameters is None
    assert jp.allclose(fitted, jp.array([3.0, 3.0]), atol=1e-3)


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
            tolerance=1e-4,
            metrics=metrics,
            metrics_every=0,
            trace=True,
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
            tolerance=1e-4,
            metrics_every=2,
            trace=True,
        )
