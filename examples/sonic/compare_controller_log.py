"""Replay original C++ SONIC controller logs through PAZ tick by tick."""

import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax
import numpy as np

from paz.models.foundation.sonic.conversion import port_sonic_weights
from paz.models.foundation.sonic.layout import load_release_observation_layout

from simulation import build_encoder_obs
from simulation import build_history_buffer
from simulation import build_history_entry_from_state
from simulation import build_policy_tail
from simulation import compute_heading_state
from simulation import load_motion_clip


def load_signal(logs_dir, name):
    path = logs_dir / f"{name}.csv"
    values = np.loadtxt(path, delimiter=",", skiprows=1)
    return values[:, 5:].astype(np.float32)


def load_policy_input(path):
    values = np.genfromtxt(path, delimiter=",")
    if values.ndim == 1:
        values = values[None, :]
    if np.isnan(values[:, -1]).all():
        values = values[:, :-1]
    return values.astype(np.float32)


def load_motion_name(path):
    with path.open(newline="", encoding="utf-8") as log_file:
        names = [row["motion_name"] for row in csv.DictReader(log_file)]
    unique_names = set(names)
    if len(unique_names) != 1:
        raise ValueError(
            "A parity log must contain exactly one motion, got "
            f"{sorted(unique_names)}")
    return names[0]


def print_result(name, errors, tolerance):
    tick = int(np.argmax(errors))
    maximum = float(errors[tick])
    result = "PASS" if maximum <= tolerance else "FAIL"
    print(
        f"{name:16} max={maximum:.9g} tick={tick:4d} "
        f"tolerance={tolerance:.3g} {result}")
    return maximum <= tolerance


if __name__ == "__main__":
    repositories = Path(__file__).resolve().parents[3]
    default_root = repositories / "GR00T-WholeBodyControl"
    default_root = default_root / "gear_sonic_deploy"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonic-root", type=Path, default=default_root)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--policy-input-logfile", type=Path, required=True)
    parser.add_argument("--tail-tolerance", type=float, default=1e-4)
    parser.add_argument("--token-tolerance", type=float, default=1e-6)
    parser.add_argument("--action-tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    sonic_root = args.sonic_root.expanduser().resolve()
    logs_dir = args.logs_dir.expanduser().resolve()
    release_dir = sonic_root / "policy" / "release"
    layout = load_release_observation_layout(
        release_dir / "observation_config.yaml")
    encoder, decoder, _ = port_sonic_weights(
        layout,
        release_dir / "model_encoder.onnx",
        release_dir / "model_decoder.onnx",
    )

    q = load_signal(logs_dir, "q")
    dq = load_signal(logs_dir, "dq")
    action = load_signal(logs_dir, "action")
    base_quat = load_signal(logs_dir, "base_quat")
    base_ang_vel = load_signal(logs_dir, "base_ang_vel")
    cpp_tokens = load_signal(logs_dir, "token_state")
    playing = load_signal(logs_dir, "motion_playing")[:, 0].astype(bool)
    encoder_modes = load_signal(logs_dir, "encoder_mode")[:, 0].astype(int)
    cpp_policy_input = load_policy_input(args.policy_input_logfile)
    row_counts = {
        len(values) for values in (
            q, dq, action, base_quat, base_ang_vel, cpp_tokens,
            playing, encoder_modes, cpp_policy_input)
    }
    if len(row_counts) != 1:
        parser.error(f"C++ log row counts disagree: {sorted(row_counts)}")
    unique_modes = set(encoder_modes.tolist())
    if len(unique_modes) != 1:
        parser.error(
            f"A parity log must contain one mode, got {unique_modes}")
    mode_id = unique_modes.pop()
    mode_names = {0: "g1", 1: "teleop", 2: "smpl"}
    if mode_id not in mode_names:
        parser.error(f"Unsupported encoder mode ID in log: {mode_id}")
    mode_name = mode_names[mode_id]

    motion_name = load_motion_name(logs_dir / "motion_name.csv")
    clip = load_motion_clip(
        sonic_root / "reference" / "example" / motion_name)
    heading = compute_heading_state(base_quat[0], clip.body_quat[0, 0])
    history = build_history_buffer()
    frame = 0
    encoder_observations = []
    policy_tails = []
    for tick in range(len(q)):
        history.append(build_history_entry_from_state(
            base_quat[tick], base_ang_vel[tick], q[tick], dq[tick],
            action[tick]))
        policy_tails.append(build_policy_tail(layout, history)[0])
        encoder_observations.append(build_encoder_obs(
            layout, mode_name, clip, frame, playing[tick], base_quat[tick],
            heading)[0])
        if playing[tick]:
            frame += 1
            if frame >= clip.num_frames:
                frame = 0

    encoder_observations = np.asarray(
        encoder_observations, dtype=np.float32)
    policy_tails = np.asarray(policy_tails, dtype=np.float32)
    encoder_fn = jax.jit(encoder)
    decoder_fn = jax.jit(decoder)
    paz_tokens = np.concatenate([
        np.asarray(encoder_fn(observation[None, :]))
        for observation in encoder_observations
    ])
    paz_actions = np.concatenate([
        np.asarray(decoder_fn(policy_input[None, :]))
        for policy_input in cpp_policy_input
    ])
    tail_errors = np.max(
        np.abs(policy_tails - cpp_policy_input[:, 64:]), axis=1)
    token_errors = np.max(np.abs(paz_tokens - cpp_tokens), axis=1)
    action_errors = np.max(
        np.abs(paz_actions[:-1] - action[1:]), axis=1)

    print(f"Compared {len(q)} {mode_name} ticks for {motion_name}")
    passed = [
        print_result("policy history", tail_errors, args.tail_tolerance),
        print_result("encoder token", token_errors, args.token_tolerance),
        print_result("decoder action", action_errors, args.action_tolerance),
    ]
    if not all(passed):
        sys.exit(1)
