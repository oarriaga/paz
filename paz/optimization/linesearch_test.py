import optax
import pytest

from paz.optimization import LineSearch
from paz.optimization import armijo_linesearch
from paz.optimization import linesearch
from paz.optimization import wolfe_linesearch


def test_linesearch_package_exports_symbols():
    assert callable(LineSearch)
    assert callable(armijo_linesearch)
    assert callable(wolfe_linesearch)
    assert hasattr(linesearch, "armijo_linesearch")
    assert hasattr(linesearch, "wolfe_linesearch")


def test_LineSearch_returns_wolfe_linesearch():
    transform = LineSearch(5, "wolfe", False)
    assert isinstance(transform, optax.GradientTransformationExtraArgs)


def test_LineSearch_returns_armijo_linesearch():
    transform = LineSearch(5, "armijo", False)
    assert isinstance(transform, optax.GradientTransformationExtraArgs)


def test_LineSearch_rejects_invalid_criterion():
    with pytest.raises(ValueError, match="armijo"):
        LineSearch(5, "invalid", False)
