import jax
import jax.numpy as jp

from paz.optimization import LBFGS
from paz.optimization import LineSearch
from paz.optimization import STOP_FN_MET
from paz.optimization import Trace
from paz.optimization import grad_norm_stop
from paz.optimization import minimize
from paz.optimization import trim_trace


def quadratic_loss(parameters):
    return jp.sum((parameters - 3.0) ** 2)


def test_minimize_returns_loss_history():
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
    assert trimmed.metrics.period == 1
    assert trimmed.metrics.default == {}
    assert trimmed.metrics.values == {}
    assert trimmed.metrics.step == 0


def test_trim_trace_trims_sparse_metrics_to_optimization_length():
    parameters = jp.array([8.0, -2.0])
    linesearch = LineSearch(10, "wolfe")
    optimizer = LBFGS(1.0, 5, linesearch)
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
    assert history.metrics.period == 2
    assert "distance" in history.metrics.default
    assert "distance" in history.metrics.values
    assert len(trimmed.metrics.trace["distance"]) == len(trimmed.metrics.steps)
    assert trimmed.metrics.trace["distance"][-1] <= trimmed.metrics.trace["distance"][0]
    assert trimmed.metrics.arg == history.metrics.arg
    assert trimmed.metrics.period == history.metrics.period
    assert jp.all(trimmed.metrics.steps <= history.stop_step)
    assert trimmed.metrics.step == history.metrics.step
    if len(trimmed.metrics.steps) > 1:
        diffs = trimmed.metrics.steps[1:] - trimmed.metrics.steps[:-1]
        assert jp.all(diffs == 2)
