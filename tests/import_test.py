import json
import os
import subprocess
import sys


def read_backend_after_importing_paz(keras_home, environment):
    keras_config = json.dumps({"backend": "tensorflow"})
    (keras_home / "keras.json").write_text(keras_config)
    environment = {**environment, "KERAS_HOME": str(keras_home)}
    code = "import paz, keras; print(keras.backend.backend())"
    command = [sys.executable, "-c", code]
    kwargs = dict(env=environment, capture_output=True, text=True, check=True)
    return subprocess.run(command, **kwargs).stdout.split()[-1]


def test_import_defaults_keras_backend_to_jax(tmp_path):
    environment = dict(os.environ)
    environment.pop("KERAS_BACKEND", None)
    assert read_backend_after_importing_paz(tmp_path, environment) == "jax"


def test_import_keeps_explicit_jax_backend(tmp_path):
    environment = {**os.environ, "KERAS_BACKEND": "jax"}
    assert read_backend_after_importing_paz(tmp_path, environment) == "jax"
