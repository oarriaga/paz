import os
import sys
from pathlib import Path

import pytest

os.environ["KERAS_BACKEND"] = "jax"

# The torch reference modules are test-only helpers. tests/ has no packages,
# so their folders go on the path to keep them importable by name.
for folder in ("layers", "models"):
    sys.path.insert(0, str(Path(__file__).parent / folder))


@pytest.fixture(scope="session")
def torch_weights_path():
    """Locates an official DINOv3 ``.pth`` file, or skips the test."""
    def locate(filename):
        directory = os.environ.get("PAZ_DINOV3_TORCH_WEIGHTS")
        if directory is None:
            reason = "set PAZ_DINOV3_TORCH_WEIGHTS to the DINOv3 .pth folder"
            pytest.skip(reason)
        path = os.path.join(directory, filename)
        if not os.path.isfile(path):
            pytest.skip(f"PyTorch weight file not found: {path}")
        return path
    return locate
