import sys
from pathlib import Path

import numpy as np
import pytest

from paz.models.foundation.sonic.layout import POLICY_FROM_SCENE

sys.path.insert(0, str(Path(__file__).parents[1] / "sonic"))
mujoco = pytest.importorskip("mujoco")
from simulation import build_encoder_obs  # noqa: E402
from simulation import compute_heading_state  # noqa: E402
from simulation import load_motion_clip  # noqa: E402
from simulation_test import build_release_layout  # noqa: E402

import sonic_clip  # noqa: E402


def build_qpos(num_frames=31):
    times = np.arange(num_frames, dtype=np.float32) / sonic_clip.KIMODO_FPS
    height = np.full_like(times, 0.8)
    position = np.stack([times * 0.5, np.zeros_like(times), height], -1)
    yaw = times * 0.3
    root = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], -1)
    angles = np.arange(29, dtype=np.float32)[None] * 0.01 + times[:, None]
    return np.concatenate([position, root, angles], -1).astype(np.float32)


def test_resampling_keeps_duration_and_linear_motion():
    qpos = sonic_clip.resample_qpos(build_qpos(), 30.0, 50.0)
    assert qpos.shape == (51, 36)
    assert np.allclose(qpos[-1, :3], build_qpos()[-1, :3], atol=1e-6)
    times = np.arange(51) / 50.0
    assert np.allclose(qpos[:, 0], times * 0.5, atol=1e-6)
    assert np.allclose(qpos[:, 7], times, atol=1e-6)


def test_resampling_slerps_the_root_yaw():
    qpos = sonic_clip.resample_qpos(build_qpos(), 30.0, 50.0)
    yaw = 2.0 * np.arctan2(qpos[:, 6], qpos[:, 3])
    assert np.allclose(yaw, np.arange(51) / 50.0 * 0.3, atol=1e-5)
    assert np.allclose(np.linalg.norm(qpos[:, 3:7], axis=-1), 1.0, atol=1e-6)


def test_written_clip_loads_into_sonic(tmp_path):
    qpos = sonic_clip.resample_qpos(build_qpos(), 30.0, 50.0)
    folder = sonic_clip.write_clip(tmp_path / "walk", qpos, 50.0)
    clip = load_motion_clip(folder)
    assert clip.num_frames == 51
    assert np.allclose(clip.joint_pos, qpos[:, 7:][:, POLICY_FROM_SCENE])
    assert np.allclose(clip.joint_vel[0], 50.0 * (qpos[1, 7:] - qpos[0, 7:])[
        POLICY_FROM_SCENE], atol=1e-4)
    assert clip.body_indices.tolist() == [0]
    assert np.allclose(clip.body_quat[:, 0], qpos[:, 3:7], atol=1e-6)
    layout = build_release_layout()
    base_quat = np.asarray([1.0, 0.0, 0.0, 0.0], np.float32)
    heading = compute_heading_state(base_quat, clip.body_quat[0, 0])
    obs = build_encoder_obs(layout, "g1", clip, 3, True, base_quat, heading)
    assert obs.shape == (1, 1762)
    assert np.isfinite(obs).all()
