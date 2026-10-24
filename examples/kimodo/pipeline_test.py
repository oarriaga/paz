"""End-to-end smoke test: text embedding -> Kimodo -> SONIC in MuJoCo.

Opt in with ``KIMODO_PIPELINE_TEST=1``; it downloads the Kimodo and SONIC
assets (or reads ``KIMODO_WEIGHTS_DIR``) and runs about a minute.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("KIMODO_PIPELINE_TEST") is None,
    reason="set KIMODO_PIPELINE_TEST=1")

DEMO = Path(__file__).with_name("mujoco_demo.py")


def read_final_pelvis(stdout):
    match = re.search(r"pelvis_xyz=\[([^\]]+)\]", stdout)
    assert match is not None, stdout
    return np.fromstring(match.group(1), sep=",")


def test_text_to_motion_to_sonic(tmp_path):
    command = [sys.executable, str(DEMO), "A person walks forward.",
               "--duration", "3", "--headless", "--output-dir", str(tmp_path)]
    weights_dir = os.environ.get("KIMODO_WEIGHTS_DIR")
    if weights_dir is not None:
        command.extend(["--models-path", weights_dir])
    result = subprocess.run(command, check=True, cwd=DEMO.parent,
                            capture_output=True, text=True)
    qpos = np.loadtxt(tmp_path / "qpos.csv", delimiter=",")
    assert qpos.shape == (90, 36)
    assert np.isfinite(qpos).all()
    clip = tmp_path / "clips" / "generated"
    joint_pos = np.loadtxt(clip / "joint_pos.csv", delimiter=",", skiprows=1)
    assert joint_pos.shape == (149, 29)
    pelvis = read_final_pelvis(result.stdout)
    print(f"reference travel {qpos[-1, 0] - qpos[0, 0]:.2f} m, "
          f"robot pelvis {pelvis.round(2).tolist()}")
    assert pelvis[2] > 0.5, "the robot fell while tracking the motion"
    assert pelvis[0] > 1.0, "the robot did not follow the walk forward"
