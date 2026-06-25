import os
import sys


os.environ.setdefault("KERAS_BACKEND", "jax")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
