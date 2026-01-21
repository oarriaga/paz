import json
import os
import shutil
import sys
from datetime import datetime, timezone

import jax
import jax.numpy as jp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

PAZ_FORMAT_VERSION = "1.0"


def _build_manifest(object_type, schema_version):
    return {
        "paz_format_version": PAZ_FORMAT_VERSION,
        "object_type": object_type,
        "schema_version": schema_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paz_version": None,
        "jax_version": jax.__version__,
        "tfp_version": tfp.__version__,
        "python_version": sys.version.split()[0],
    }


def _write_safe(path, manifest, payload, arrays, overwrite):
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f"Path '{path}' already exists.")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    _write_json(os.path.join(path, "manifest.json"), manifest)
    _write_json(os.path.join(path, "payload.json"), payload)
    _write_arrays(os.path.join(path, "arrays.npz"), arrays)


def _read_safe(path, device):
    manifest = _read_json(os.path.join(path, "manifest.json"))
    payload = _read_json(os.path.join(path, "payload.json"))
    arrays = _read_arrays(os.path.join(path, "arrays.npz"), device)
    return manifest, payload, arrays


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_arrays(path, arrays):
    np_arrays = {name: np.asarray(value) for name, value in arrays.items()}
    np.savez(path, **np_arrays)


def _read_arrays(path, device):
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: _to_jax(data[name], device) for name in data}
    return arrays


def _to_jax(value, device):
    value = jp.asarray(value)
    return jax.device_put(value, device) if device is not None else value
