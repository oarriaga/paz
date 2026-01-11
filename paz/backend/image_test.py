import tempfile
from pathlib import Path

import jax.numpy as jp

import paz


def test_write_accepts_path_object():
    image = jp.full((4, 4, 3), 128, dtype=jp.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "image.png"
        paz.image.write(filepath, image)
        assert filepath.is_file()


def test_load_accepts_path_object():
    image = jp.full((4, 4, 3), 200, dtype=jp.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "image.png"
        paz.image.write(str(filepath), image)
        loaded = paz.image.load(filepath)
        assert loaded.shape == image.shape
