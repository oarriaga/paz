import jax
import jax.numpy as jp

from paz import SO3
from paz.models.foundation.kimodo import skeleton


def sample_rotations(seed):
    keys = jax.random.split(jax.random.PRNGKey(seed), skeleton.NUM_JOINTS)
    return jax.vmap(SO3.sample)(keys)


def test_parents_precede_children():
    assert skeleton.PARENTS[0] == -1
    assert all(p < j for j, p in enumerate(skeleton.PARENTS) if j > 0)
    assert skeleton.NEUTRAL_JOINTS.shape == (skeleton.NUM_JOINTS, 3)


def test_identity_rotations_reproduce_neutral_pose():
    identities = jp.tile(jp.eye(3), (skeleton.NUM_JOINTS, 1, 1))
    root = jp.array([1.0, 0.8, -2.0])
    positions = skeleton.compute_joint_positions(identities, root)
    assert jp.allclose(positions, skeleton.NEUTRAL_JOINTS + root, atol=1e-6)


def test_local_and_global_rotations_invert_each_other():
    local = sample_rotations(0)
    global_rotations = skeleton.to_global_rotations(local)
    recovered = skeleton.to_local_rotations(global_rotations)
    assert jp.allclose(recovered, local, atol=1e-5)


def test_heading_angle_follows_root_yaw():
    identities = jp.tile(jp.eye(3), (skeleton.NUM_JOINTS, 1, 1))
    neutral = skeleton.compute_joint_positions(identities, jp.zeros(3))
    assert jp.allclose(skeleton.compute_heading_angle(neutral), 0.0)
    turned = jp.tile(SO3.rotation_y(0.7), (skeleton.NUM_JOINTS, 1, 1))
    positions = skeleton.compute_joint_positions(turned, jp.zeros(3))
    assert jp.allclose(skeleton.compute_heading_angle(positions), 0.7,
                       atol=1e-5)


def test_joint_positions_vmap_over_frames():
    rotations = jp.stack([sample_rotations(1), sample_rotations(2)])
    roots = jp.zeros((2, 3))
    positions = jax.vmap(skeleton.compute_joint_positions)(rotations, roots)
    assert positions.shape == (2, skeleton.NUM_JOINTS, 3)
    assert jp.isfinite(positions).all()
