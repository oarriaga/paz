import hashlib
import os
import sys
from collections import namedtuple

import jax
import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "slam"))

import synthetic  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "tests", "slam", "data")
BASELINE_NPZ = os.path.join(DATA_DIR, "vslam_baseline_v1.npz")
BASELINE_JSON = os.path.join(DATA_DIR, "vslam_baseline_v1.json")

Fixtures = namedtuple(
    "Fixtures", ["two_view_clean", "two_view_noisy", "pnp", "bundle",
                 "stereo"])

SEEDS = {"two_view_clean": 1, "two_view_noisy": 2, "pnp": 3,
         "bundle": 7, "stereo": 4}


def build_fixtures():
    clean = synthetic.build_two_view_scene(
        jax.random.PRNGKey(SEEDS["two_view_clean"]), 120, 0.0, 0.0)
    noisy = synthetic.build_two_view_scene(
        jax.random.PRNGKey(SEEDS["two_view_noisy"]), 150, 0.5, 0.15)
    pnp = synthetic.build_pnp_scene(
        jax.random.PRNGKey(SEEDS["pnp"]), 150, 0.5, 0.15)
    bundle = synthetic.build_bundle_adjustment_scene(
        jax.random.PRNGKey(SEEDS["bundle"]), 6, 200, 0.5, 0.03, 0.05)
    stereo = synthetic.build_stereo_sequence(
        jax.random.PRNGKey(SEEDS["stereo"]), 40, 500, 0.5, 0.10)
    return Fixtures(clean, noisy, pnp, bundle, stereo)


def compute_checksums(fixtures):
    checksums = {}
    for name, fixture in zip(Fixtures._fields, fixtures):
        digest = hashlib.sha256()
        for array in fixture:
            digest.update(np.ascontiguousarray(array).tobytes())
        checksums[name] = digest.hexdigest()
    return checksums


def to_arrays(fixtures):
    arrays = {}
    for name, fixture in zip(Fixtures._fields, fixtures):
        for field, array in zip(type(fixture)._fields, fixture):
            arrays[f"{name}.{field}"] = np.asarray(array)
    return arrays
