"""Interactive MuJoCo demonstration of the PAZ GEAR-WBC controller."""

import argparse
import os
from pathlib import Path
import signal
import time

os.environ.setdefault("KERAS_BACKEND", "jax")

# One 15-joint actor at batch one is bound by launch latency, not by
# throughput, so the CPU runs it faster per call than a GPU and leaves the
# GPU free. Set JAX_PLATFORMS to override.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import mujoco
import mujoco.viewer
import numpy as np

from paz.models import GearWBC
from paz.models.foundation.gear_wbc.model import ACTION_DIM

from controller import CONTROLS
from controller import build_pad
from controller import read_command
from simulation import CONTROL_DECIMATION
from simulation import LOWER_BODY_ANGLES
from simulation import SIMULATION_STEP
from simulation import build_command
from simulation import build_history
from simulation import build_observation_frame
from simulation import build_plant
from simulation import compile_actors
from simulation import compute_action
from simulation import compute_control
from simulation import compute_target_angles
from simulation import update_history

STATUS_STEPS = 100


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


def restore_interrupt():
    # launch_passive keeps SIGINT for itself, so Ctrl-C never reaches Python
    # and the loop runs until the process is killed. Take the signal back.
    signal.signal(signal.SIGINT, signal.default_int_handler)


def build_viewer(model, data):
    # The camera tracks the pelvis. Left at the origin it loses the robot
    # after a couple of metres, which reads as a frozen scene.
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = pelvis
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -20
    viewer.cam.distance = 3.0
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
    actors = compile_actors(GearWBC(weights="pretrained"))
    model, data = build_plant(scene_path)
    pad = build_pad()

    command = build_command()
    history = build_history()
    action = np.zeros(ACTION_DIM, "float32")
    target_angles = LOWER_BODY_ANGLES.copy()

    if arguments.headless:
        viewer = None
    else:
        viewer = build_viewer(model, data)
        restore_interrupt()

    control_period = SIMULATION_STEP * CONTROL_DECIMATION
    step, last_time = 0, time.perf_counter()
    try:
        while arguments.steps == 0 or step < arguments.steps:
            if viewer is not None and not viewer.is_running():
                break
            data.ctrl[:] = compute_control(data, target_angles)
            mujoco.mj_step(model, data)
            step = step + 1
            if step % CONTROL_DECIMATION == 0:
                command = read_command(pad)
                frame = build_observation_frame(data, command, action)
                observation = update_history(history, frame)
                action = compute_action(actors, observation, command)
                target_angles = compute_target_angles(action)
                if viewer is not None:
                    viewer.sync()
                    last_time = sleep_to_rate(last_time, control_period)
            if step % STATUS_STEPS == 0:
                print(describe(command), end="\r", flush=True)
    except KeyboardInterrupt:
        pass

    print(f"\nRan {step} steps. Final base height {data.qpos[2]:.3f} m")
    if viewer is not None:
        viewer.close()
        time.sleep(0.25)
