"""Interactive MuJoCo demonstration of the PAZ GEAR-WBC controller."""

import argparse
import os
from pathlib import Path
import time

os.environ.setdefault("KERAS_BACKEND", "jax")

import mujoco
import mujoco.viewer
import numpy as np

from paz.models import GearWBC
from paz.models.foundation.gear_wbc.model import ACTION_DIM

from simulation import CONTROL_DECIMATION
from simulation import LOWER_BODY_ANGLES
from simulation import SIMULATION_STEP
from simulation import build_command
from simulation import build_history
from simulation import build_observation_frame
from simulation import build_plant
from simulation import compute_action
from simulation import compute_control
from simulation import compute_target_angles
from simulation import update_history

LINEAR_STEP = 0.2
ANGULAR_STEP = 0.2
HEIGHT_STEP = 0.1
ORIENTATION_STEP = np.deg2rad(10)

VELOCITY_KEYS = "w", "s", "a", "d", "q", "e", "z"
HEIGHT_KEYS = "1", "2"
ORIENTATION_KEYS = "3", "4", "5", "6", "7", "8"
ORIENTATION_AXES = {"3": 0, "4": 0, "5": 1, "6": 1, "7": 2, "8": 2}

CONTROLS = """GEAR-WBC / PAZ controls
  w / s   forward velocity
  a / d   lateral velocity
  q / e   yaw rate
  z       stop
  1 / 2   base height
  3 / 4   torso roll
  5 / 6   torso pitch
  7 / 8   torso yaw
"""


def apply_key(command, key):
    if key in VELOCITY_KEYS:
        command = apply_velocity_key(command, key)
    elif key in HEIGHT_KEYS:
        command = apply_height_key(command, key)
    elif key in ORIENTATION_KEYS:
        command = apply_orientation_key(command, key)
    return command


def apply_velocity_key(command, key):
    velocity = command.velocity.copy()
    if key == "w":
        velocity[0] = velocity[0] + LINEAR_STEP
    elif key == "s":
        velocity[0] = velocity[0] - LINEAR_STEP
    elif key == "a":
        velocity[1] = velocity[1] + LINEAR_STEP
    elif key == "d":
        velocity[1] = velocity[1] - LINEAR_STEP
    elif key == "q":
        velocity[2] = velocity[2] + ANGULAR_STEP
    elif key == "e":
        velocity[2] = velocity[2] - ANGULAR_STEP
    else:
        velocity = np.zeros(3, "float32")
    return command._replace(velocity=velocity)


def apply_height_key(command, key):
    if key == "1":
        height = command.height + HEIGHT_STEP
    else:
        height = command.height - HEIGHT_STEP
    return command._replace(height=height)


def apply_orientation_key(command, key):
    orientation = command.orientation.copy()
    axis = ORIENTATION_AXES[key]
    if key in ("3", "5", "7"):
        orientation[axis] = orientation[axis] - ORIENTATION_STEP
    else:
        orientation[axis] = orientation[axis] + ORIENTATION_STEP
    return command._replace(orientation=orientation)


def describe(command):
    velocity = np.round(command.velocity, 2)
    orientation = np.round(np.rad2deg(command.orientation), 1)
    height = round(command.height, 2)
    return f"velocity {velocity}  height {height}  rpy_deg {orientation}"


def sleep_to_rate(last_time, timestep):
    next_time = last_time + timestep
    now = time.perf_counter()
    if next_time > now:
        time.sleep(next_time - now)
        rate_time = next_time
    else:
        rate_time = now
    return rate_time


def build_viewer(model, data, key_callback):
    launch = mujoco.viewer.launch_passive
    viewer = launch(model, data, key_callback=key_callback)
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -20
    viewer.cam.distance = 3.0
    viewer.cam.lookat = np.asarray([0.0, 0.0, 0.8])
    return viewer


if __name__ == "__main__":
    repositories = Path(__file__).resolve().parents[3]
    sibling = repositories / "GR00T-WholeBodyControl" / "decoupled_wbc"
    sibling = sibling / "sim2mujoco" / "resources" / "robots" / "g1"
    default_scene_dir = os.environ.get("GEAR_WBC_SCENE_DIR", sibling)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=default_scene_dir,
                        help="g1 directory holding g1_gear_wbc.xml")
    parser.add_argument("--steps", type=int, default=0,
                        help="stop after this many simulation steps")
    parser.add_argument("--headless", action="store_true")
    arguments = parser.parse_args()

    scene_path = Path(arguments.scene_dir) / "g1_gear_wbc.xml"
    if not scene_path.exists():
        raise SystemExit(f"Missing MuJoCo scene: {scene_path}")

    print(CONTROLS)
    print(f"Loading PAZ GEAR-WBC weights and scene from {scene_path}")
    models = GearWBC(weights="pretrained")
    model, data = build_plant(scene_path)

    command = build_command()
    history = build_history()
    action = np.zeros(ACTION_DIM, "float32")
    target_angles = LOWER_BODY_ANGLES.copy()

    def key_callback(keycode):
        global command
        command = apply_key(command, chr(keycode).lower())
        print(describe(command))

    if arguments.headless:
        viewer = None
    else:
        viewer = build_viewer(model, data, key_callback)

    step, last_time = 0, time.perf_counter()
    while arguments.steps == 0 or step < arguments.steps:
        if viewer is not None and not viewer.is_running():
            break
        data.ctrl[:] = compute_control(data, target_angles)
        mujoco.mj_step(model, data)
        step = step + 1
        if step % CONTROL_DECIMATION == 0:
            frame = build_observation_frame(data, command, action)
            observation = update_history(history, frame)
            action = compute_action(models, observation, command)
            target_angles = compute_target_angles(action)
        if viewer is not None:
            viewer.sync()
            last_time = sleep_to_rate(last_time, SIMULATION_STEP)

    print(f"Ran {step} steps. Final base height {data.qpos[2]:.3f} m")
    if viewer is not None:
        viewer.close()
        time.sleep(0.25)
