import json
import pickle
import tempfile
from collections import namedtuple
from pathlib import Path

import jax.numpy as jp

from paz import pytree


Sample = namedtuple("Sample", ["array", "value", "items"])


def test_to_pickle_saves_and_loads_pytree():
    tree = Sample(jp.array([1, 2, 3]), 7, {"a": jp.array([[1.0, 2.0]])})
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "tree.pkl"
        pytree.to_pickle(tree, filepath)
        loaded = pickle.load(open(filepath, "rb"))

    assert isinstance(loaded, Sample)
    assert jp.array_equal(loaded.array, tree.array)
    assert loaded.value == tree.value
    assert jp.array_equal(loaded.items["a"], tree.items["a"])


def test_to_json_saves_serializable_pytree():
    tree = Sample(jp.array([1, 2]), 3, {"a": jp.array([[1.0, 2.0]])})
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "tree.json"
        pytree.to_json(tree, filepath)
        loaded = json.load(open(filepath, "r"))

    assert loaded == {"array": [1, 2], "value": 3, "items": {"a": [[1.0, 2.0]]}}
