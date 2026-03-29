import paz

from paz.optimization import LBFGS
from paz.optimization import LineSearch
from paz.optimization import grad_norm_stop
from paz.optimization import loss_stop
from paz.optimization import minimize
from paz.optimization import optimizers


def test_optimizers_package_exports_symbols():
    assert callable(LBFGS)
    assert hasattr(optimizers, "LBFGS")


def test_root_package_exports_optimization_symbols():
    assert paz.optimizers.LBFGS is LBFGS
    assert paz.optimizers.LineSearch is LineSearch
    assert paz.minimize is minimize
    assert paz.optimization.loss_stop is loss_stop
    assert paz.optimization.grad_norm_stop is grad_norm_stop
