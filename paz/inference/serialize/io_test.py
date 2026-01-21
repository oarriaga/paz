import jax.numpy as jp

from paz.inference.serialize.io import (
    _build_manifest,
    _read_arrays,
    _read_json,
    _read_safe,
    _to_jax,
    _write_arrays,
    _write_json,
    _write_safe,
)


def test_build_manifest_fields():
    manifest = _build_manifest("Thing", 2)
    assert manifest["object_type"] == "Thing"
    assert manifest["schema_version"] == 2
    assert "jax_version" in manifest
    assert "tfp_version" in manifest
    assert "python_version" in manifest


def test_write_read_json(tmp_path):
    path = tmp_path / "payload.json"
    payload = {"a": 1, "b": "c"}
    _write_json(path, payload)
    loaded = _read_json(path)
    assert loaded == payload


def test_write_read_arrays(tmp_path):
    path = tmp_path / "arrays.npz"
    arrays = {"x": jp.array([1.0, 2.0])}
    _write_arrays(path, arrays)
    loaded = _read_arrays(path, None)
    assert jp.allclose(loaded["x"], arrays["x"])


def test_write_read_safe(tmp_path):
    path = tmp_path / "safe"
    manifest = {"object_type": "X"}
    payload = {"value": 3}
    arrays = {"x": jp.array([0.5])}
    _write_safe(path, manifest, payload, arrays, overwrite=False)
    loaded_manifest, loaded_payload, loaded_arrays = _read_safe(path, None)
    assert loaded_manifest == manifest
    assert loaded_payload == payload
    assert jp.allclose(loaded_arrays["x"], arrays["x"])


def test_to_jax_returns_array():
    value = _to_jax([1, 2, 3], None)
    assert isinstance(value, jp.ndarray)
    assert jp.allclose(value, jp.array([1, 2, 3]))
