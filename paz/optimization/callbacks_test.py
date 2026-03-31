from types import SimpleNamespace

from paz.optimization.callbacks import _build_progress_description
from paz.optimization.callbacks import _build_progress_suffix


class Metrics:
    def __init__(self, values):
        self.values = values


def test_build_progress_description_adds_compact_armijo_failure():
    metrics = Metrics({"depth": 0.15, "masks": 0.05})
    description = _build_progress_description(0.2, metrics)
    assert description == "loss=0.2 | depth=0.15 | masks=0.05"


def test_build_progress_suffix_adds_compact_armijo_failure():
    info = SimpleNamespace(num_linesearch_steps=50, decrease_error=1e-8)
    suffix = _build_progress_suffix(1e-4, info, False, None)
    assert suffix == "ls=fail(n=50, lr=1.00e-04, dec=1.00e-08)"


def test_build_progress_suffix_adds_compact_wolfe_failure():
    info = SimpleNamespace(
        num_linesearch_steps=10,
        decrease_error=0.0,
        curvature_error=1e-6,
    )
    suffix = _build_progress_suffix(5e-3, info, False, None)
    assert suffix == "ls=fail(n=10, lr=5.00e-03, curv=1.00e-06)"


def test_build_progress_suffix_adds_stop_message():
    suffix = _build_progress_suffix(None, None, True, "stop=grad_norm")
    assert suffix == "stop=grad_norm"


def test_build_progress_suffix_skips_status_on_success():
    info = SimpleNamespace(
        num_linesearch_steps=3,
        decrease_error=0.0,
        curvature_error=0.0,
    )
    suffix = _build_progress_suffix(1e-2, info, False, None)
    assert suffix is None
