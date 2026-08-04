"""Opt-in numeric parity check against the real released GEAR-WBC weights.

Skipped unless GEAR_WBC_RELEASE_DIR points at the released policy directory
holding GR00T-WholeBodyControl-Balance.onnx and
GR00T-WholeBodyControl-Walk.onnx (decoupled_wbc/sim2mujoco/resources/robots/
g1/policy in GR00T-WholeBodyControl, with git-lfs pulled).
"""
import os
from pathlib import Path

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")

from paz.models.foundation.gear_wbc.conversion import port_weights
from paz.models.foundation.gear_wbc.model import OBSERVATION_DIM

RELEASE_DIR = os.environ.get("GEAR_WBC_RELEASE_DIR")
RELEASE_NAMES = ("GR00T-WholeBodyControl-Balance.onnx",
                 "GR00T-WholeBodyControl-Walk.onnx")


def release_files_exist():
    if not RELEASE_DIR:
        return False
    release_dir = Path(RELEASE_DIR)
    return all((release_dir / name).exists() for name in RELEASE_NAMES)


pytestmark = pytest.mark.skipif(
    not release_files_exist(),
    reason="GEAR_WBC_RELEASE_DIR not set to a real release directory")


@pytest.mark.parametrize("release_name", RELEASE_NAMES)
def test_actor_matches_onnx_runtime(release_name):
    onnx_path = Path(RELEASE_DIR) / release_name
    actor = port_weights(onnx_path)
    session = ort.InferenceSession(str(onnx_path))
    x = np.random.default_rng(0).normal(size=(4, OBSERVATION_DIM))
    x = x.astype("float32")
    onnx_action = session.run(None, {"input": x})[0]
    paz_action = np.array(actor(x, training=False))
    assert np.abs(onnx_action - paz_action).max() < 1e-4


def test_balance_and_walk_are_distinct_experts():
    actors = [port_weights(Path(RELEASE_DIR) / n) for n in RELEASE_NAMES]
    x = np.random.default_rng(1).normal(size=(1, OBSERVATION_DIM))
    x = x.astype("float32")
    actions = [np.array(actor(x, training=False)) for actor in actors]
    assert not np.allclose(actions[0], actions[1])
