"""Captures reference outputs from NVIDIA's torch implementation.

Development tool for the parity tests. It needs the upstream ``kimodo``
package and torch, runs on CPU in float32 with the fused attention fast
path disabled, and writes one ``.npz`` that the ``*_test.py`` files read
through ``KIMODO_FIXTURES``.

    python -m paz.models.foundation.kimodo.reference --output fixtures.npz
"""
import argparse

import numpy as np
import torch

from kimodo.constraints import FullBodyConstraintSet, Root2DConstraintSet
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.model.load_model import load_model

BATCH, NUM_FRAMES, TEXT_DIM = 2, 60, 4096


def capture(output_path):
    torch.manual_seed(0)
    torch.backends.mha.set_fastpath_enabled(False)
    model = load_model("kimodo-g1-rp", device="cpu", text_encoder=object())
    model = model.float().eval()
    fixtures = capture_denoiser(model)
    fixtures.update(capture_schedules(model))
    fixtures.update(capture_sampling(model, fixtures))
    fixtures.update(capture_skeleton(model))
    fixtures.update(capture_conditioning(model, fixtures))
    np.savez(output_path, **fixtures)


def capture_denoiser(model):
    denoiser = model.denoiser.model
    motion_rep = denoiser.motion_rep
    noisy = torch.randn(BATCH, NUM_FRAMES, motion_rep.motion_rep_dim)
    text = torch.randn(BATCH, 1, TEXT_DIM)
    timesteps = torch.tensor([3, 500])
    heading = torch.tensor([0.0, 0.4])
    pad_mask = torch.ones(BATCH, NUM_FRAMES, dtype=torch.bool)
    text_mask = torch.ones(BATCH, 1, dtype=torch.bool)
    mask = (torch.rand_like(noisy) < 0.2).float()
    observed = torch.randn_like(noisy) * mask
    with torch.inference_mode():
        output = denoiser(noisy, pad_mask, text, text_mask, timesteps,
                          heading, None, None)
        masked_output = denoiser(noisy, pad_mask, text, text_mask, timesteps,
                                 heading, mask, observed)
        root_input = torch.cat([noisy, torch.zeros_like(noisy)], -1)
        root = denoiser.root_model(root_input, pad_mask, text, text_mask,
                                   timesteps, heading)
        lengths = pad_mask.sum(-1)
        local_root = motion_rep.global_root_to_local_root(
            root, normalized=True, lengths=lengths)
    return {
        "noisy": noisy, "text": text, "timesteps": timesteps,
        "heading": heading, "mask": mask, "observed": observed,
        "denoiser_output": output, "denoiser_output_masked": masked_output,
        "root_output": root, "local_root": local_root,
        "global_root_mean": motion_rep.global_root_stats.mean,
        "global_root_std": motion_rep.global_root_stats.std,
        "local_root_mean": motion_rep.local_root_stats.mean,
        "local_root_std": motion_rep.local_root_stats.std,
        "body_mean": motion_rep.body_stats.mean,
        "body_std": motion_rep.body_stats.std,
    }


def capture_schedules(model):
    fixtures = {}
    for num_steps in (3, 100):
        use_timesteps, map_tensor = model.diffusion.space_timesteps(num_steps)
        model.diffusion.calc_diffusion_vars(use_timesteps)
        diffusion = model.diffusion
        prefix = f"schedule_{num_steps}"
        fixtures[f"{prefix}_base_timesteps"] = map_tensor[:num_steps]
        fixtures[f"{prefix}_sqrt_recip"] = (
            diffusion.sqrt_recip_alphas_cumprod[:num_steps])
        fixtures[f"{prefix}_sqrt_recipm1"] = (
            diffusion.sqrt_recipm1_alphas_cumprod[:num_steps])
        fixtures[f"{prefix}_alphas_prev"] = (
            diffusion.alphas_cumprod_prev[:num_steps])
    return fixtures


def capture_sampling(model, fixtures):
    motion_rep = model.motion_rep
    shape = (BATCH, NUM_FRAMES, motion_rep.motion_rep_dim)
    torch.manual_seed(1)
    noise = torch.randn(shape)
    torch.manual_seed(1)
    pad_mask = torch.ones(BATCH, NUM_FRAMES, dtype=torch.bool)
    text_mask = torch.ones(BATCH, 1, dtype=torch.bool)
    with torch.inference_mode():
        sampled = model._generate(
            None, NUM_FRAMES, num_denoising_steps=3, pad_mask=pad_mask,
            first_heading_angle=torch.tensor([0.0, 0.4]), motion_mask=None,
            observed_motion=None, cfg_weight=[2.0, 2.0],
            text_feat=torch.as_tensor(fixtures["text"]),
            text_pad_mask=text_mask, cfg_type="separated",
            progress_bar=lambda x: x)
        decoded = motion_rep.inverse(sampled, is_normalized=True)
        converter = MujocoQposConverter(model.skeleton)
        qpos = converter.to_qpos(decoded["local_rot_mats"],
                                 decoded["root_positions"])
    return {
        "sampling_noise": noise, "sampled": sampled, "qpos": qpos,
        "decoded_local_rotations": decoded["local_rot_mats"],
        "decoded_global_rotations": decoded["global_rot_mats"],
        "decoded_joint_positions": decoded["posed_joints"],
        "decoded_root_positions": decoded["root_positions"],
        "decoded_foot_contacts": decoded["foot_contacts"],
    }


def capture_skeleton(model):
    converter = MujocoQposConverter(model.skeleton)
    return {
        "neutral_joints": model.skeleton.neutral_joints,
        "joint_parents": model.skeleton.joint_parents,
        "mujoco_rotation_offsets": converter._rot_offsets_f2q,
        "mujoco_hinge_axes": converter._mujoco_joint_axis_values_f2q_space,
        "mujoco_hinge_joints": converter._mujoco_indices_to_kimodo_indices,
    }


def capture_conditioning(model, fixtures):
    skeleton = model.skeleton
    frames = torch.tensor([0, 10, 25])
    positions_xz = torch.tensor([[0.0, 0.0], [0.5, 0.2], [1.0, 0.9]])
    angles = torch.tensor([0.0, 0.3, 0.6])
    heading = torch.stack([torch.cos(angles), torch.sin(angles)], -1)
    root_path = Root2DConstraintSet(skeleton, frames, positions_xz,
                                    global_root_heading=heading)
    keyframes = torch.tensor([40, 59])
    joint_positions = torch.as_tensor(fixtures["decoded_joint_positions"])
    joint_positions = joint_positions[0, keyframes]
    rotations = torch.as_tensor(fixtures["decoded_global_rotations"])
    body = FullBodyConstraintSet(skeleton, keyframes, joint_positions,
                                 rotations[0, keyframes])
    observed, mask = model.motion_rep.create_conditions_from_constraints(
        [root_path, body], NUM_FRAMES, to_normalize=True, device="cpu")
    return {
        "condition_frames": frames, "condition_xz": positions_xz,
        "condition_angles": angles, "keyframe_frames": keyframes,
        "keyframe_positions": joint_positions,
        "condition_observed": observed, "condition_mask": mask,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    capture(parser.parse_args().output)
