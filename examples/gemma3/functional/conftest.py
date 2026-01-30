import os
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from examples.gemma3.functional.keras_hub_utils import ensure_keras_hub


os.environ.setdefault("KERAS_BACKEND", "jax")
ensure_keras_hub()
