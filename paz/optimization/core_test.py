import pytest
import jax.numpy as jp
import optax

from paz.optimization import LBFGS
from paz.optimization import LineSearch
from paz.optimization import linesearch
from paz.optimization import minimize
from paz.optimization import optimizers


def quadratic_loss(parameters):
    return jp.sum((parameters - 3.0) ** 2)


def test_optimization_package_exports_symbols():
    assert callable(LBFGS)
    assert callable(LineSearch)
    assert callable(minimize)
    assert hasattr(linesearch, "zoom_linesearch")
    assert hasattr(optimizers, "LBFGS")


def test_LineSearch_returns_zoom_linesearch():
    transform = LineSearch(5, True, False)
    assert isinstance(transform, optax.GradientTransformationExtraArgs)


def test_minimize_reduces_loss_with_LBFGS_without_trace():
    parameters = jp.array([8.0, -2.0])
    linesearch_transform = LineSearch(10, True, False)
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


def test_minimize_keeps_history_when_tolerance_is_not_met():
    parameters = jp.array([8.0, -2.0])
    linesearch_transform = LineSearch(10, True, False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)
    success, _, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=2,
        tolerance=-1.0,
        trace=True,
    )
    losses, _, metrics = history
    assert not bool(success)
    assert len(losses) == 2
    assert metrics == {}


def test_minimize_returns_history_when_trace_is_enabled():
    parameters = jp.array([8.0, -2.0])
    linesearch_transform = LineSearch(10, True, False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)
    success, fitted, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=20,
        tolerance=1e-4,
        trace=True,
    )
    losses, _, metrics = history
    assert bool(success)
    assert losses[-1] <= losses[0]
    assert jp.allclose(fitted, jp.array([3.0, 3.0]), atol=1e-3)
    assert metrics == {}


def test_minimize_trims_metrics_to_optimization_length():
    parameters = jp.array([8.0, -2.0])
    linesearch_transform = LineSearch(10, True, False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)
    metrics = lambda value: {"distance": jp.linalg.norm(value - 3.0)}
    success, _, history = minimize(
        parameters,
        quadratic_loss,
        optimizer,
        max_steps=20,
        tolerance=1e-4,
        metrics=metrics,
        trace=True,
    )
    losses, _, metric_history = history
    assert bool(success)
    assert len(metric_history["distance"]) == len(losses)
    assert metric_history["distance"][-1] <= metric_history["distance"][0]


def test_minimize_rejects_metrics_without_trace():
    parameters = jp.array([8.0, -2.0])
    linesearch_transform = LineSearch(10, True, False)
    optimizer = LBFGS(1.0, 5, False, linesearch_transform)
    metrics = lambda value: {"distance": jp.linalg.norm(value - 3.0)}
    with pytest.raises(ValueError, match="trace=True"):
        minimize(
            parameters,
            quadratic_loss,
            optimizer,
            max_steps=20,
            tolerance=1e-4,
            metrics=metrics,
        )
