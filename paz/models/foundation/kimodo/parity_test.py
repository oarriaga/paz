"""Numerical parity against fixtures captured from NVIDIA's torch code.

Run ``reference.py`` once in a torch environment, convert the checkpoint
with ``conversion.py``, then::

    KIMODO_FIXTURES=~/.keras/paz/models/kimodo/reference_fixtures.npz \\
    KIMODO_WEIGHTS_DIR=~/.keras/paz/models/kimodo \\
    pytest paz/models/foundation/kimodo/parity_test.py -s
"""
import os
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest

from paz.models.foundation.kimodo import conditioning
from paz.models.foundation.kimodo import diffusion, features, sampling
from paz.models.foundation.kimodo import model as denoiser_model
from paz.models.foundation.kimodo import mujoco as export
from paz.models.foundation.kimodo import skeleton
from paz.models.foundation.kimodo.pretrained import KimodoG1RP

FIXTURES = os.environ.get("KIMODO_FIXTURES")
WEIGHTS_DIR = os.environ.get("KIMODO_WEIGHTS_DIR")
pytestmark = pytest.mark.skipif(
    FIXTURES is None or WEIGHTS_DIR is None,
    reason="set KIMODO_FIXTURES and KIMODO_WEIGHTS_DIR")


@pytest.fixture(scope="module", autouse=True)
def full_precision():
    # TF32 matmuls on GPU inflate the 32-layer error to about 1e-2.
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_default_matmul_precision", None)


@pytest.fixture(scope="module")
def fixtures():
    return dict(np.load(Path(FIXTURES).expanduser()))


@pytest.fixture(scope="module")
def kimodo():
    return KimodoG1RP(models_path=Path(WEIGHTS_DIR).expanduser())


def report(name, actual, expected, tolerance):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    print(f"{name}: max abs error {error:.2e}")
    assert error < tolerance, f"{name} error {error} exceeds {tolerance}"


def report_relative(name, actual, expected, tolerance):
    scale = np.maximum(np.abs(np.asarray(expected)), 1.0)
    error = np.abs(np.asarray(actual) - np.asarray(expected)) / scale
    print(f"{name}: max relative error {float(error.max()):.2e}")
    assert error.max() < tolerance


def test_stats_and_skeleton_match(fixtures, kimodo):
    for block in ("global_root", "local_root", "body"):
        normalization = getattr(kimodo.stats, block)
        assert np.allclose(normalization.mean, fixtures[f"{block}_mean"])
        expected = np.sqrt(fixtures[f"{block}_std"] ** 2 + 1e-5)
        assert np.allclose(normalization.stdv, expected)
    report("neutral joints", skeleton.NEUTRAL_JOINTS,
           fixtures["neutral_joints"], 1e-6)
    assert list(skeleton.PARENTS) == fixtures["joint_parents"].tolist()


def test_mujoco_tables_match(fixtures):
    report("rotation offsets", export.ROTATION_OFFSETS,
           fixtures["mujoco_rotation_offsets"], 1e-6)
    report("hinge axes", export.HINGE_AXIS_VECTORS,
           fixtures["mujoco_hinge_axes"], 1e-6)
    hinge_joints = fixtures["mujoco_hinge_joints"].tolist()
    assert list(export.HINGE_INDICES) == hinge_joints


@pytest.mark.parametrize("num_steps", [3, 100])
def test_schedule_matches(fixtures, num_steps):
    schedule = diffusion.build_schedule(num_steps)
    prefix = f"schedule_{num_steps}"
    assert schedule.base_timesteps.tolist() == (
        fixtures[f"{prefix}_base_timesteps"].tolist())
    report_relative(f"{prefix} sqrt_recip", schedule.sqrt_recip_alphas,
                    fixtures[f"{prefix}_sqrt_recip"], 1e-5)
    report_relative(f"{prefix} sqrt_recipm1", schedule.sqrt_recipm1_alphas,
                    fixtures[f"{prefix}_sqrt_recipm1"], 1e-5)
    report(f"{prefix} alphas_prev", schedule.alphas_prev,
           fixtures[f"{prefix}_alphas_prev"], 1e-6)


def build_inputs(fixtures, mask, observed):
    return [fixtures["noisy"], mask, observed, fixtures["text"],
            fixtures["timesteps"].astype("int32"), fixtures["heading"]]


def test_denoiser_forward_matches(fixtures, kimodo):
    zeros = np.zeros_like(fixtures["noisy"])
    output = kimodo.denoiser(build_inputs(fixtures, zeros, zeros))
    report("denoiser output", output, fixtures["denoiser_output"], 5e-4)
    report("root stage", output[..., :5], fixtures["root_output"], 5e-4)


def test_masked_denoiser_forward_matches(fixtures, kimodo):
    inputs = build_inputs(fixtures, fixtures["mask"], fixtures["observed"])
    output = kimodo.denoiser(inputs)
    report("masked output", output, fixtures["denoiser_output_masked"], 5e-4)


def test_local_root_matches(fixtures, kimodo):
    local = denoiser_model.convert_root_to_local(
        fixtures["root_output"], kimodo.stats, features.FPS)
    report("local root", local, fixtures["local_root"], 1e-4)


def test_three_step_sampling_matches(fixtures, kimodo):
    config = sampling.SamplingConfig(60, 3, 2.0, 2.0, 0.0)
    observed, mask = sampling.build_empty_constraints(2, 60)
    text = jp.asarray(fixtures["text"])
    heading = fixtures["heading"]
    sampled = []
    for sample in range(2):
        config = config._replace(heading_angle=float(heading[sample]))
        noise = jp.asarray(fixtures["sampling_noise"][sample:sample + 1])
        args = (kimodo.denoiser, text[sample:sample + 1], observed[:1],
                mask[:1], config)
        sampled.append(sampling.denoise_from_noise(noise, *args))
    report("3-step sample", jp.concatenate(sampled), fixtures["sampled"], 2e-3)


def test_decode_matches(fixtures, kimodo):
    decode = jax.vmap(lambda x: features.decode(x, kimodo.stats))
    motion = decode(jp.asarray(fixtures["sampled"]))
    report("joint positions", motion.joint_positions,
           fixtures["decoded_joint_positions"], 1e-4)
    report("local rotations", motion.local_rotations,
           fixtures["decoded_local_rotations"], 1e-4)
    report("root positions", motion.root_positions,
           fixtures["decoded_root_positions"], 1e-5)
    assert np.array_equal(np.asarray(motion.foot_contacts),
                          fixtures["decoded_foot_contacts"])
    qpos = jax.vmap(jax.vmap(export.to_qpos))(motion.local_rotations,
                                              motion.root_positions)
    report("qpos", qpos, fixtures["qpos"], 1e-4)


def test_conditioning_matches(fixtures, kimodo):
    condition = conditioning.build_empty(60)
    condition = conditioning.add_root_path(
        condition, fixtures["condition_frames"], fixtures["condition_xz"])
    condition = conditioning.add_heading(
        condition, fixtures["condition_frames"], fixtures["condition_angles"])
    condition = conditioning.add_keyframes(
        condition, fixtures["keyframe_frames"], fixtures["keyframe_positions"])
    condition = conditioning.normalize(condition, kimodo.stats)
    assert np.array_equal(np.asarray(condition.mask) > 0,
                          fixtures["condition_mask"])
    report("conditioning", condition.observed, fixtures["condition_observed"],
           1e-5)
