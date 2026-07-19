"""Interactive MuJoCo demonstration of the PAZ SONIC actor."""

import argparse
import os
from pathlib import Path
import time

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax
import mujoco
import mujoco.viewer
import numpy as np

from paz.models.foundation.sonic.conversion import port_weights
from paz.models.foundation.sonic.layout import compute_encoder_input_dim
from paz.models.foundation.sonic.layout import compute_policy_tail_dim
from paz.models.foundation.sonic.layout import load_release_observation_layout

from simulation import DEFAULT_ANGLES
from simulation import NUM_JOINTS
from simulation import apply_support_force
from simulation import build_encoder_obs
from simulation import build_history_buffer
from simulation import build_history_entry_from_state
from simulation import build_policy_tail
from simulation import check_mode_available
from simulation import clamp_frame
from simulation import clear_viewer_markers
from simulation import compute_action_targets
from simulation import compute_hand_pd_torque
from simulation import compute_heading_state
from simulation import compute_pd_torque
from simulation import compute_reference_markers
from simulation import find_joint_addresses
from simulation import load_motion_set
from simulation import update_viewer_markers


INIT_SECONDS = 3.0
CONTROL_WARMUP_SECONDS = 2.0


def interpolate_targets(start, end, ratio):
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    return start * (np.float32(1.0) - ratio) + end * ratio


def find_motion_index(clips, requested):
    try:
        return int(requested) % len(clips)
    except ValueError:
        pass
    for index, clip in enumerate(clips):
        if requested.lower() in clip.name.lower():
            return index
    raise ValueError(f"No motion name contains {requested!r}")


def print_controls():
    print("\nSONIC / PAZ MuJoCo controls")
    print("  space       pause or resume")
    print("  [ / ]       previous or next motion")
    print("  , / .       previous or next frame while paused")
    print("  1 / 2 / 3   G1, 3-point teleop, or SMPL encoder")
    print("  g           toggle the original elastic band")
    print("  v           hide or show reference targets")
    print("  r           reset the current motion")
    print("  h           print these controls")
    print("  Esc         close the viewer\n")


def update_hud(viewer, state):
    if viewer is None or not hasattr(viewer, "set_texts"):
        return
    clip = state["clip"]
    support = "elastic band on" if state["support_enabled"] else "free"
    playback = "playing" if state["play"] else "paused"
    labels = "SONIC / PAZ\nmode\nmotion\nframe\ncontroller\nsupport"
    values = (
        "Keras + JAX\n"
        f"{state['mode']}\n"
        f"{clip.name}\n"
        f"{state['frame'] + 1} / {clip.num_frames}\n"
        f"{state['phase']} / {playback}\n"
        f"{support}"
    )
    controls = "space play  [ ] motion  1 2 3 mode  g assist  h help"
    texts = [
        (None, mujoco.mjtGridPos.mjGRID_TOPLEFT, labels, values),
        (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, controls, ""),
    ]
    viewer.set_texts(texts)


def sleep_to_rate(last_time, timestep):
    next_time = last_time + timestep
    now = time.perf_counter()
    if next_time > now:
        time.sleep(next_time - now)
        return next_time
    return now


def close_viewer(viewer):
    viewer.close()
    # launch_passive owns a daemon render thread. Give it time to destroy its
    # GLFW resources before Python unloads MuJoCo during a scripted shutdown.
    time.sleep(0.25)


if __name__ == "__main__":
    repositories = Path(__file__).resolve().parents[3]
    sibling_deploy = repositories / "GR00T-WholeBodyControl"
    sibling_deploy = sibling_deploy / "gear_sonic_deploy"
    default_root = os.environ.get("SONIC_DEPLOY_DIR", sibling_deploy)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sonic-root",
        type=Path,
        default=default_root,
        help="gear_sonic_deploy directory (or set SONIC_DEPLOY_DIR)",
    )
    parser.add_argument("--mode", choices=("g1", "teleop", "smpl"),
                        default="g1")
    parser.add_argument(
        "--motion",
        default="0",
        help="zero-based motion index or a unique part of its name",
    )
    parser.add_argument("--sim-dt", type=float, default=0.005)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    args = parser.parse_args()

    sonic_root = args.sonic_root.expanduser().resolve()
    release_dir = sonic_root / "policy" / "release"
    scene_path = sonic_root.parent / "gear_sonic" / "data"
    scene_path = scene_path / "robot_model/model_data/g1/scene_43dof.xml"
    motion_dir = sonic_root / "reference" / "example"
    required_paths = (
        release_dir / "observation_config.yaml",
        release_dir / "model_encoder.onnx",
        release_dir / "model_decoder.onnx",
        scene_path,
        motion_dir,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        parser.error(
            "SONIC deployment assets are missing: " + ", ".join(missing))
    if args.control_dt < args.sim_dt:
        parser.error("--control-dt must be at least --sim-dt")

    print(f"Loading PAZ SONIC weights from {release_dir}")
    config_path = release_dir / "observation_config.yaml"
    layout = load_release_observation_layout(config_path)
    _, _, actor = port_weights(
        layout,
        release_dir / "model_encoder.onnx",
        release_dir / "model_decoder.onnx",
    )
    clips = load_motion_set(motion_dir)
    motion_index = find_motion_index(clips, args.motion)
    available, reason = check_mode_available(clips[motion_index], args.mode)
    if not available:
        parser.error(f"Cannot start in {args.mode} mode: {reason}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    model.opt.timestep = args.sim_dt
    pelvis_id = model.body("pelvis").id
    joints = find_joint_addresses(model)
    if joints.hand_qpos.size != 14:
        parser.error(
            "The original SONIC simulator scene must contain 14 hand joints")
    policy_steps = max(1, int(round(args.control_dt / args.sim_dt)))
    action_fn = jax.jit(
        lambda encoder_obs, policy_tail: actor(
            {
                "encoder_obs": encoder_obs,
                "policy_obs_tail": policy_tail,
            },
            training=False,
        )
    )
    print("Compiling the PAZ actor for JAX (first launch can take a moment)")
    empty_encoder = np.zeros(
        (1, compute_encoder_input_dim(layout)), dtype=np.float32)
    empty_tail = np.zeros(
        (1, compute_policy_tail_dim(layout)), dtype=np.float32)
    np.asarray(action_fn(empty_encoder, empty_tail))

    state = {
        "clip": clips[motion_index],
        "motion_index": motion_index,
        "mode": args.mode,
        "frame": 0,
        "play": not args.start_paused,
        "pending_reset": True,
        "pending_frame": 0,
        "history": build_history_buffer(),
        "last_action": np.zeros(NUM_JOINTS, np.float32),
        "q_target": np.zeros(NUM_JOINTS, np.float32),
        "startup_q": np.zeros(NUM_JOINTS, np.float32),
        "startup_step": 0,
        "control_tick": 0,
        "phase": "resetting",
        "heading": None,
        "support_enabled": True,
        "auto_release": True,
        "show_markers": True,
    }
    init_steps = max(1, int(round(INIT_SECONDS / args.control_dt)))
    warmup_steps = max(
        1, int(round(CONTROL_WARMUP_SECONDS / args.control_dt)))

    def reset_state():
        mujoco.mj_resetData(model, data)
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(model, data)
        state["frame"] = 0
        state["history"] = build_history_buffer()
        state["last_action"] = np.zeros(NUM_JOINTS, np.float32)
        joint_q = data.qpos[joints.body_qpos]
        state["q_target"] = joint_q.astype(np.float32)
        state["startup_q"] = state["q_target"].copy()
        state["startup_step"] = 0
        state["control_tick"] = 0
        state["phase"] = "initializing"
        state["support_enabled"] = True
        state["auto_release"] = True
        state["heading"] = None

    def select_motion(offset):
        new_index = (state["motion_index"] + offset) % len(clips)
        new_clip = clips[new_index]
        available, reason = check_mode_available(new_clip, state["mode"])
        if not available:
            print(f"{new_clip.name}: {state['mode']} unavailable ({reason})")
            return
        state["motion_index"] = new_index
        state["clip"] = new_clip
        state["pending_reset"] = True
        print(f"Motion: {new_clip.name}")

    def select_mode(mode_name):
        available, reason = check_mode_available(state["clip"], mode_name)
        if not available:
            print(f"{mode_name} mode unavailable: {reason}")
            return
        state["mode"] = mode_name
        state["pending_reset"] = True
        print(f"Encoder mode: {mode_name}")

    def key_callback(keycode):
        try:
            key = chr(keycode).lower()
        except ValueError:
            key = ""
        if key == " ":
            state["play"] = not state["play"]
        elif key == "[":
            select_motion(-1)
        elif key == "]":
            select_motion(1)
        elif key == ",":
            state["play"] = False
            state["pending_frame"] = -1
        elif key == ".":
            state["play"] = False
            state["pending_frame"] = 1
        elif key == "1":
            select_mode("g1")
        elif key == "2":
            select_mode("teleop")
        elif key == "3":
            select_mode("smpl")
        elif key == "g":
            state["support_enabled"] = not state["support_enabled"]
            state["auto_release"] = False
        elif key == "v":
            state["show_markers"] = not state["show_markers"]
        elif key == "r":
            state["pending_reset"] = True
        elif key == "h":
            print_controls()

    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=key_callback,
            show_left_ui=False,
            show_right_ui=False,
        )
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat = np.asarray([0.0, 0.0, 0.8])

    print_controls()
    reset_state()
    state["pending_reset"] = False
    update_hud(viewer, state)
    max_steps = args.steps
    if args.headless and max_steps is None:
        max_steps = 1500
    realtime = not args.headless and not args.no_realtime
    last_time = time.perf_counter()
    step = 0
    try:
        while True:
            if max_steps is not None and step >= max_steps:
                break
            if viewer is not None and not viewer.is_running():
                break
            if state["pending_reset"]:
                reset_state()
                state["pending_reset"] = False
                state["pending_frame"] = 0
            if step % policy_steps == 0:
                if state["pending_frame"]:
                    frame = state["frame"] + state["pending_frame"]
                    state["frame"] = clamp_frame(frame, state["clip"])
                    state["pending_frame"] = 0
                    state["heading"] = compute_heading_state(
                        data.qpos[3:7],
                        state["clip"].body_quat[state["frame"], 0],
                    )
                if state["startup_step"] < init_steps:
                    state["startup_step"] += 1
                    ratio = np.float32(
                        state["startup_step"] / init_steps)
                    state["q_target"] = interpolate_targets(
                        state["startup_q"], DEFAULT_ANGLES, ratio)
                    state["phase"] = "initializing"
                else:
                    joint_q = data.qpos[joints.body_qpos]
                    joint_dq = data.qvel[joints.body_dof]
                    if state["heading"] is None:
                        state["heading"] = compute_heading_state(
                            data.qpos[3:7],
                            state["clip"].body_quat[state["frame"], 0],
                        )
                    entry = build_history_entry_from_state(
                        data.qpos[3:7], data.qvel[3:6], joint_q,
                        joint_dq, state["last_action"])
                    state["history"].append(entry)
                    warming_up = state["control_tick"] < warmup_steps
                    play_for_obs = state["play"] and not warming_up
                    encoder_obs = build_encoder_obs(
                        layout, state["mode"], state["clip"],
                        state["frame"], play_for_obs, data.qpos[3:7],
                        state["heading"],
                    )
                    policy_tail = build_policy_tail(
                        layout, state["history"])
                    action = np.asarray(
                        action_fn(encoder_obs, policy_tail))[0]
                    action = action.astype(np.float32)
                    state["q_target"] = compute_action_targets(action)
                    state["last_action"] = action
                    state["control_tick"] += 1
                    if warming_up:
                        state["phase"] = "policy warmup"
                        if state["control_tick"] == warmup_steps:
                            if state["auto_release"]:
                                state["support_enabled"] = False
                    elif state["play"]:
                        state["phase"] = "running"
                        if state["frame"] < state["clip"].num_frames - 1:
                            state["frame"] += 1
                        else:
                            state["play"] = False
                    else:
                        state["phase"] = "paused"
                if viewer is not None:
                    if state["show_markers"]:
                        points = compute_reference_markers(
                            state["mode"], state["clip"], state["frame"],
                            data.qpos[:3], data.qpos[3:7])
                        update_viewer_markers(
                            viewer, points, state["mode"])
                    else:
                        clear_viewer_markers(viewer)
                    update_hud(viewer, state)

            apply_support_force(
                model, data, pelvis_id, float(state["support_enabled"]))
            joint_q = data.qpos[joints.body_qpos]
            joint_dq = data.qvel[joints.body_dof]
            data.ctrl[joints.body_actuator] = compute_pd_torque(
                state["q_target"], joint_q, joint_dq)
            hand_q = data.qpos[joints.hand_qpos]
            hand_dq = data.qvel[joints.hand_dof]
            data.ctrl[joints.hand_actuator] = compute_hand_pd_torque(
                hand_q, hand_dq)
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                raise RuntimeError("MuJoCo state became non-finite")
            if viewer is not None and step % policy_steps == 0:
                viewer.sync()
            if realtime:
                last_time = sleep_to_rate(last_time, args.sim_dt)
            step += 1
    finally:
        if viewer is not None:
            close_viewer(viewer)

    print(
        f"Completed {step} simulation steps; mode={state['mode']}, "
        f"motion={state['clip'].name}, pelvis_z={data.qpos[2]:.3f}"
    )
