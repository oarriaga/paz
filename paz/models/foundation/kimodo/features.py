"""Kimodo motion features: block layout, normalization and decoding.

A frame is a 417 vector: smooth root position (3), heading cos/sin (2),
joint positions relative to the smooth root (34 x 3), global joint
rotations as 6D (34 x 6), joint velocities (34 x 3) and foot contacts (4).
The first five values are the global root block, the rest the body block.
"""
from collections import namedtuple
from itertools import accumulate

import jax
import jax.numpy as jp

from paz import SO3
from paz.models.foundation.kimodo import skeleton

FPS = 30.0
NUM_JOINTS = skeleton.NUM_JOINTS
BLOCK_SIZES = (3, 2, 3 * NUM_JOINTS, 6 * NUM_JOINTS, 3 * NUM_JOINTS, 4)
BLOCK_BOUNDS = tuple(accumulate(BLOCK_SIZES))[:-1]
FEATURE_DIM = sum(BLOCK_SIZES)
GLOBAL_ROOT_DIM = 5
LOCAL_ROOT_DIM = 4
BODY_DIM = FEATURE_DIM - GLOBAL_ROOT_DIM

Normalization = namedtuple("Normalization", "mean stdv")
MotionStats = namedtuple("MotionStats", "global_root local_root body")
MotionFeatures = namedtuple(
    "MotionFeatures",
    "smooth_root_positions heading local_joint_positions rotations_6d "
    "velocities foot_contacts",
)
Motion = namedtuple(
    "Motion",
    "local_rotations global_rotations joint_positions root_positions "
    "smooth_root_positions foot_contacts heading",
)


def build_normalization(mean, std):
    # The reference adds the epsilon inside the square root.
    stdv = jp.sqrt(jp.asarray(std) ** 2 + 1e-5)
    return Normalization(jp.asarray(mean), stdv)


def build_stats(arrays):
    blocks = ("global_root", "local_root", "body")
    normalizations = []
    for block in blocks:
        mean, std = arrays[f"{block}_mean"], arrays[f"{block}_std"]
        normalizations.append(build_normalization(mean, std))
    return MotionStats(*normalizations)


def motion_normalization(stats):
    mean = jp.concatenate([stats.global_root.mean, stats.body.mean])
    stdv = jp.concatenate([stats.global_root.stdv, stats.body.stdv])
    return Normalization(mean, stdv)


def normalize(features, normalization):
    return (features - normalization.mean) / normalization.stdv


def unnormalize(features, normalization):
    return features * normalization.stdv + normalization.mean


def split(features):
    """Unpacks ``(..., 417)`` features into per-block arrays."""
    blocks = jp.split(features, BLOCK_BOUNDS, axis=-1)
    leading = features.shape[:-1]
    blocks[2] = blocks[2].reshape(*leading, NUM_JOINTS, 3)
    blocks[3] = blocks[3].reshape(*leading, NUM_JOINTS, 6)
    blocks[4] = blocks[4].reshape(*leading, NUM_JOINTS, 3)
    return MotionFeatures(*blocks)


def decode(features, stats):
    """Turns normalized ``(T, 417)`` features into articulated motion."""
    features = unnormalize(features, motion_normalization(stats))
    blocks = split(features)
    global_rotations = jax.vmap(jax.vmap(SO3.from_6d))(blocks.rotations_6d)
    local_rotations = jax.vmap(skeleton.to_local_rotations)(global_rotations)
    planar = blocks.smooth_root_positions * jp.array([1.0, 0.0, 1.0])
    root_positions = blocks.local_joint_positions[:, 0] + planar
    compute_positions = jax.vmap(skeleton.compute_joint_positions)
    joint_positions = compute_positions(global_rotations, root_positions)
    args = (local_rotations, global_rotations, joint_positions, root_positions,
            blocks.smooth_root_positions, blocks.foot_contacts > 0.5,
            blocks.heading)
    return Motion(*args)
