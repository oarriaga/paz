import os

from .io import _read_safe, _write_safe
from .serde import _find_serde, _serde_for_type


def save(obj, path, format="paz", overwrite=False):
    if format != "paz":
        raise ValueError(f"Unknown format '{format}'")
    serde = _find_serde(obj)
    manifest, payload, arrays = serde.to_spec(obj)
    _write_safe(path, manifest, payload, arrays, overwrite)


def load(path, format=None, device=None):
    if format is None:
        format = _infer_format(path)
    if format != "paz":
        raise ValueError(f"Unknown format '{format}'")
    manifest, payload, arrays = _read_safe(path, device)
    serde = _serde_for_type(manifest["object_type"])
    return serde.from_spec(manifest, payload, arrays)


def _infer_format(path):
    manifest = os.path.join(path, "manifest.json")
    if os.path.exists(manifest):
        return "paz"
    raise ValueError(f"Unable to infer format for '{path}'.")
