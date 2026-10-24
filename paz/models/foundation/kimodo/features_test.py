import jax
import jax.numpy as jp

from paz import SO3
from paz.models.foundation.kimodo import features
from paz.models.foundation.kimodo import skeleton


def build_unit_stats():
    blocks = ("global_root", "local_root", "body")
    dims = (features.GLOBAL_ROOT_DIM, features.LOCAL_ROOT_DIM,
            features.BODY_DIM)
    arrays = {}
    for block, dim in zip(blocks, dims):
        arrays[f"{block}_mean"] = jp.zeros(dim)
        arrays[f"{block}_std"] = jp.ones(dim)
    return features.build_stats(arrays)


def build_neutral_features(num_frames, smooth_root):
    identity_6d = SO3.to_6d(jp.eye(3))
    rotations = jp.tile(identity_6d, (num_frames, features.NUM_JOINTS, 1))
    local = jp.tile(skeleton.NEUTRAL_JOINTS, (num_frames, 1, 1))
    local = local.at[:, :, 1].add(smooth_root[1])
    blocks = [
        jp.tile(smooth_root, (num_frames, 1)),
        jp.tile(jp.array([1.0, 0.0]), (num_frames, 1)),
        local.reshape(num_frames, -1),
        rotations.reshape(num_frames, -1),
        jp.zeros((num_frames, 3 * features.NUM_JOINTS)),
        jp.ones((num_frames, 4)),
    ]
    return jp.concatenate(blocks, axis=-1)


def test_feature_dim_matches_the_released_layout():
    assert features.FEATURE_DIM == 417
    assert features.BODY_DIM == 412


def test_split_restores_block_shapes():
    blocks = features.split(jp.arange(2 * 417.0).reshape(2, 417))
    assert blocks.smooth_root_positions.shape == (2, 3)
    assert blocks.heading.shape == (2, 2)
    assert blocks.local_joint_positions.shape == (2, 34, 3)
    assert blocks.rotations_6d.shape == (2, 34, 6)
    assert blocks.velocities.shape == (2, 34, 3)
    assert blocks.foot_contacts.shape == (2, 4)
    assert blocks.heading[0, 0] == 3.0


def test_normalize_round_trip_uses_epsilon_adjusted_stdv():
    normalization = features.build_normalization(jp.array([1.0]),
                                                 jp.array([0.0]))
    assert jp.allclose(normalization.stdv, jp.sqrt(1e-5))
    x = jp.array([[3.0]])
    recovered = features.unnormalize(features.normalize(x, normalization),
                                     normalization)
    assert jp.allclose(recovered, x, atol=1e-5)


def test_decode_neutral_pose():
    smooth_root = jp.array([1.0, 0.8, -2.0])
    motion = features.decode(build_neutral_features(3, smooth_root),
                             build_unit_stats())
    assert jp.allclose(motion.root_positions, smooth_root, atol=1e-6)
    expected = skeleton.NEUTRAL_JOINTS + smooth_root
    assert jp.allclose(motion.joint_positions[1], expected, atol=1e-5)
    assert jp.allclose(motion.local_rotations, jp.eye(3), atol=1e-6)
    assert bool(motion.foot_contacts.all())


def test_decode_is_jittable():
    stats = build_unit_stats()
    decode = jax.jit(lambda x: features.decode(x, stats))
    motion = decode(build_neutral_features(4, jp.zeros(3)))
    assert motion.joint_positions.shape == (4, 34, 3)
