import os
import sys
from pathlib import Path

os.environ["KERAS_BACKEND"] = "jax"

# The torch reference modules are test-only helpers. tests/ has no packages,
# so their folders go on the path to keep them importable by name.
for folder in ("layers", "models"):
    sys.path.insert(0, str(Path(__file__).parent / folder))
