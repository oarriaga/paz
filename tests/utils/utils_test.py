import tempfile
from pathlib import Path

import jax.numpy as jp
import pytest

import paz


# assert_snapshot


@pytest.fixture
def temp_snapshot_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_assert_snapshot_raises_on_missing_golden(temp_snapshot_dir):
    filepath = temp_snapshot_dir / "missing.npy"
    with pytest.raises(FileNotFoundError):
        paz.assert_snapshot(jp.zeros(3), str(filepath))


def test_assert_snapshot_does_not_write_missing_golden(temp_snapshot_dir):
    filepath = temp_snapshot_dir / "missing.npy"
    with pytest.raises(FileNotFoundError):
        paz.assert_snapshot(jp.zeros(3), str(filepath))
    assert not filepath.exists()


def test_assert_snapshot_update_writes_golden(temp_snapshot_dir):
    filepath = temp_snapshot_dir / "golden.npy"
    paz.assert_snapshot(jp.arange(3.0), str(filepath), update=True)
    assert filepath.exists()


def test_assert_snapshot_matches_written_golden(temp_snapshot_dir):
    filepath = temp_snapshot_dir / "golden.npy"
    paz.assert_snapshot(jp.arange(3.0), str(filepath), update=True)
    paz.assert_snapshot(jp.arange(3.0), str(filepath))


def test_assert_snapshot_raises_on_mismatch(temp_snapshot_dir):
    filepath = temp_snapshot_dir / "golden.npy"
    paz.assert_snapshot(jp.arange(3.0), str(filepath), update=True)
    with pytest.raises(AssertionError):
        paz.assert_snapshot(jp.ones(3), str(filepath))
