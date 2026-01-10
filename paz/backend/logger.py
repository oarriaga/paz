"""
DEPRECATED: This module is deprecated and will be removed in a future version.

Please use the new specialized modules instead:
- paz.directory (make, make_timestamped, find_latest)
- paz.file (write_json, write_weights, load_csv, load_latest)
- paz.message (warn)

For now, all functions remain accessible through paz.logger for backward
compatibility, but you should migrate to the new modules.

Migration guide:
    paz.logger.make_directory → paz.directory.make
    paz.logger.make_timestamped_directory → paz.directory.make_timestamped
    paz.logger.find_path → paz.directory.find_latest
    paz.logger.write_dictionary → paz.file.write_json
    paz.logger.write_weights → paz.file.write_weights
    paz.logger.load_csv → paz.file.load_csv
    paz.logger.load_latest → paz.file.load_latest
    paz.logger.warn → paz.message.warn
"""
import os
import json
import warnings

import keras
import jax

from paz.backend.directory import make as make_directory
from paz.backend.directory import make_timestamped as make_timestamped_directory
from paz.backend.directory import find_latest as find_path
from paz.backend.file import write_json as write_dictionary
from paz.backend.file import write_weights, load_latest, load_csv
from paz.backend.message import warn

warnings.warn(
    "paz.logger is deprecated and will be removed in a future version. "
    "Please use paz.directory, paz.file, or paz.message instead. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)


# def setup(args, model=None, label=None, root="experiments"):
#     labels = [item for item in [model, label] if item]
#     root = make_timestamped_directory(root, "_".join(labels))
#     write_dictionary(args.__dict__, root, "parameters.json")
#     keras.utils.set_random_seed(args.seed)
#     key = jax.random.PRNGKey(args.seed)
#     return root, key


def setup(args):
    defaults = {"model": None, "label": None, "root": "log", "seed": 777}
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
            warn(f"`{key}` not found in `args`. Using default `{value}`.")

    keras.utils.set_random_seed(args.seed)
    labels = [item for item in [args.model, args.label] if item]
    experiment_name = "_".join(labels)
    experiment_root = make_timestamped_directory(args.root, experiment_name)
    filepath = os.path.join(experiment_root, "parameters.json")
    write_dictionary(args.__dict__, filepath)
    return experiment_root, jax.random.PRNGKey(args.seed)
