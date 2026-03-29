import jax.numpy as jp

from paz.optimization import MAX_STEPS_REACHED
from paz.optimization import STOP_FN_MET
from paz.optimization import grad_norm_stop
from paz.optimization import loss_stop


def test_stopping_exports_symbols():
    assert MAX_STEPS_REACHED == 0
    assert STOP_FN_MET == 1
    assert callable(grad_norm_stop)
    assert callable(loss_stop)


def test_grad_norm_stop_uses_gradient_norm():
    stop_fn = grad_norm_stop(1e-3)
    gradients = jp.array([1e-4, 1e-4])
    assert stop_fn(1, jp.array([0.0]), 0.0, gradients)


def test_loss_stop_uses_loss_value():
    stop_fn = loss_stop(1e-3)
    assert stop_fn(1, jp.array([0.0]), 1e-4, jp.array([1.0]))
