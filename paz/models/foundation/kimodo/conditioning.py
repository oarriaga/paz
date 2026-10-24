"""Kinematic constraints as masked feature observations.

Mirrors the reference ``create_conditions`` for the Kimodo representation:
a root path pins the planar smooth-root position (and optionally the
heading), and a full-body keyframe pins every joint position relative to
that root. Constraints accumulate into a ``Condition`` of unnormalized
``(T, 417)`` observations and a boolean mask; ``normalize`` prepares the
pair for ``sampling.sample``.
"""
from collections import namedtuple

import jax
import jax.numpy as jp

from paz.models.foundation.kimodo import features, skeleton

Condition = namedtuple("Condition", "observed mask")
ROOT_XZ = (0, 2)
ROOT_Y = (1,)
HEADING = (3, 4)
POSITIONS = tuple(range(5, 5 + 3 * skeleton.NUM_JOINTS))


def build_empty(num_frames):
    shape = (num_frames, features.FEATURE_DIM)
    return Condition(jp.zeros(shape, "float32"), jp.zeros(shape, bool))


def add_root_path(condition, frames, positions_xz):
    """Pins the smooth root ``(x, z)`` at ``frames`` to ``positions_xz``."""
    return set_columns(condition, frames, ROOT_XZ, positions_xz)


def add_heading(condition, frames, angles):
    direction = jp.stack([jp.cos(angles), jp.sin(angles)], axis=-1)
    return set_columns(condition, frames, HEADING, direction)


def add_keyframes(condition, frames, joint_positions):
    """Pins full-body poses given as world joint positions ``(K, 34, 3)``.

    As in the reference, the smooth root follows the pelvis at those frames
    and the heading is read from the hips.
    """
    root = joint_positions[:, 0]
    condition = add_root_path(condition, frames, root[:, jp.array(ROOT_XZ)])
    condition = set_columns(condition, frames, ROOT_Y, root[:, 1:2])
    heading = jax.vmap(skeleton.compute_heading_angle)(joint_positions)
    condition = add_heading(condition, frames, heading)
    planar_root = root * jp.array([1.0, 0.0, 1.0])
    local = (joint_positions - planar_root[:, None]).reshape(len(frames), -1)
    return set_columns(condition, frames, POSITIONS, local)


def set_columns(condition, frames, columns, values):
    rows = jp.asarray(frames)[:, None]
    columns = jp.asarray(columns)[None, :]
    observed = condition.observed.at[rows, columns].set(values)
    mask = condition.mask.at[rows, columns].set(True)
    return Condition(observed, mask)


def normalize(condition, stats):
    """Normalized observations and a float mask ready for the sampler."""
    normalization = features.motion_normalization(stats)
    observed = features.normalize(condition.observed, normalization)
    return Condition(observed, condition.mask.astype("float32"))
