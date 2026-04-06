import gc
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import pytest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


collect_ignore = ["keras_hub"]


@pytest.fixture
def clear_keras_session():
    keras.backend.clear_session()
    gc.collect()
    yield
    keras.backend.clear_session()
    gc.collect()
