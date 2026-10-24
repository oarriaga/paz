"""Turns Kimodo MuJoCo ``qpos`` trajectories into SONIC reference clips.

SONIC consumes 50 Hz clips with joint angles in its policy order, joint
velocities, and the root pose as body slot 0. Kimodo generates 30 Hz
``qpos`` rows in MuJoCo scene order, so the trajectory is resampled
(linear for positions and angles, slerp for the root quaternion) and
reordered before being written in the clip-folder format that
``examples/sonic/simulation.py`` loads.
"""
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from paz import quaternion
from paz.models.foundation.sonic.layout import POLICY_FROM_SCENE

KIMODO_FPS = 30.0
SONIC_FPS = 50.0


def resample_qpos(qpos, source_fps, target_fps):
    """Resamples ``(T, 36)`` qpos rows onto the target frame rate."""
    source_times = np.arange(len(qpos)) / source_fps
    num_frames = int(np.floor(source_times[-1] * target_fps)) + 1
    times = np.arange(num_frames) / target_fps
    position = interpolate(times, source_times, qpos[:, :3])
    root_quaternion = slerp_sequence(times, source_times, qpos[:, 3:7])
    angles = interpolate(times, source_times, qpos[:, 7:])
    return np.concatenate([position, root_quaternion, angles], axis=-1)


def interpolate(times, source_times, values):
    columns = [np.interp(times, source_times, column) for column in values.T]
    return np.stack(columns, axis=-1).astype(np.float32)


def slerp_sequence(times, source_times, quaternions):
    index = np.searchsorted(source_times, times, side="right") - 1
    index = np.clip(index, 0, len(source_times) - 2)
    step = source_times[index + 1] - source_times[index]
    ratio = (times - source_times[index]) / step
    args = (jp.asarray(quaternions[index]), jp.asarray(quaternions[index + 1]),
            jp.asarray(ratio, dtype=jp.float32))
    return np.asarray(jax.vmap(quaternion.slerp)(*args), dtype=np.float32)


def compute_velocities(values, fps):
    velocities = fps * np.diff(values, axis=0)
    return np.concatenate([velocities, velocities[-1:]], axis=0)


def write_clip(directory, qpos, fps):
    """Writes a SONIC clip folder with the root as the only tracked body."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joint_pos = qpos[:, 7:][:, POLICY_FROM_SCENE]
    write_csv(directory / "joint_pos.csv", joint_pos, "joint")
    write_csv(directory / "joint_vel.csv", compute_velocities(joint_pos, fps),
              "joint")
    write_csv(directory / "body_pos.csv", qpos[:, :3], "body_0")
    write_csv(directory / "body_quat.csv", qpos[:, 3:7], "body_0")
    (directory / "metadata.txt").write_text("Body part indexes: [0]\n")
    return directory


def write_csv(path, values, prefix):
    names = [f"{prefix}_{column}" for column in range(values.shape[1])]
    header = ",".join(names)
    np.savetxt(path, values, delimiter=",", header=header, comments="")
