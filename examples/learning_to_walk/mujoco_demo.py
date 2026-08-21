"""Interactive MuJoCo demonstration of the unitree_rl_lab G1 policy."""

import argparse
import os
from pathlib import Path
import signal
import time

os.environ.setdefault("KERAS_BACKEND", "jax")

# One 29-joint actor at batch one is bound by launch latency, not by
# throughput, so the CPU runs it faster per call than a GPU and leaves the
# GPU free. Set JAX_PLATFORMS to override.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import mujoco
import mujoco.viewer
import numpy as np

from controller import CONTROLS
from controller import find_pad
from controller import read_command
from controller import read_push
from policy import compile_actor
from policy import find_latest_checkpoint
from policy import load_actor
from simulation import CONTROL_DECIMATION
from simulation import CONTROL_FREQUENCY
from simulation import PUSH_DURATION
from simulation import PUSH_FORCE
from simulation import SIMULATION_FREQUENCY
from simulation import SIMULATION_STEP
from simulation import TRAINED_PUSH_SPEED
from simulation import apply_push
from simulation import build_flat_plant
from simulation import build_rollout
from simulation import build_terrain_plant
from simulation import compute_control
from simulation import compute_push_speed
from simulation import compute_tilt
from simulation import draw_push
from simulation import find_torso
from simulation import has_fallen
from simulation import randomize_plant
from simulation import sample_push
from simulation import sample_push_step
from simulation import update_rollout
from terrain import TERRAIN_NAMES
from terrain import build_terrain
from terrain import describe_terrain

# A shove lasts 0.2 s, so a slower status line misses most of them.
STATUS_STEPS = 20


def describe_rates():
    return (f"Control {CONTROL_FREQUENCY:.0f} Hz over physics at"
            f" {SIMULATION_FREQUENCY:.0f} Hz, decimation {CONTROL_DECIMATION}")


def describe_push(force, speed):
    ratio = speed / TRAINED_PUSH_SPEED
    if ratio > 1.0:
        note = f"OUT OF DISTRIBUTION: {ratio:.2f}x the 1.0 m/s it trained on"
    else:
        note = f"inside the {TRAINED_PUSH_SPEED} m/s kick it trained on"
    shove = f"{force:.0f} N for {PUSH_DURATION} s, worth {speed:.2f} m/s"
    return f"Pushes shove the torso at {shove}\n  {note}"


def describe(data, command, torso):
    velocity = np.round(command, 2)
    tilt = round(float(np.rad2deg(compute_tilt(data))), 1)
    push = np.linalg.norm(data.xfrc_applied[torso, 0:3])
    return (f"command {velocity}  height {data.qpos[2]:.2f}"
            f"  tilt {tilt} deg  push {push:5.0f} N")


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
    example_dir = Path(__file__).resolve().parent
    repositories = example_dir.parents[2]
    sibling = repositories / "unitree_mujoco" / "unitree_robots" / "g1"
    default_scene = os.environ.get("WALK_SCENE_DIR", sibling)
    experiment = example_dir / "unitree_g1_29dof_velocity_robust"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=default_scene,
                        help="g1 directory holding scene_29dof.xml")
    parser.add_argument("--checkpoint", type=Path,
                        help="rsl_rl model_*.pt, newest run by default")
    parser.add_argument("--terrain", default="flat", choices=TERRAIN_NAMES,
                        help="ground the robot walks over")
    parser.add_argument("--difficulty", type=float, default=1.0,
                        help="terrain height or grade, 1 is the tallest"
                             " trained; above 1 is out of distribution")
    parser.add_argument("--seed", type=int, default=0,
                        help="terrain layout and push schedule")
    parser.add_argument("--push-force", type=float, default=PUSH_FORCE,
                        help="torso shove in newtons, held for 0.2 s")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="forward velocity held with no pad, in m/s")
    parser.add_argument("--steps", type=int, default=0,
                        help="stop after this many simulation steps")
    parser.add_argument("--headless", action="store_true")
    arguments = parser.parse_args()

    scene_path = Path(arguments.scene_dir) / "scene_29dof.xml"
    if not scene_path.exists():
        raise SystemExit(f"Missing MuJoCo scene: {scene_path}")

    checkpoint = arguments.checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint(experiment)

    if arguments.headless:
        pad = None
    else:
        pad = find_pad()
    if pad is None:
        print(f"No pad. Holding {arguments.speed} m/s forward.")
    else:
        print(CONTROLS)
    print(f"Loading {checkpoint} and scene {scene_path}")
    actor = compile_actor(load_actor(checkpoint))
    if arguments.terrain == "flat":
        plant = build_flat_plant(scene_path)
    else:
        ground = arguments.terrain, arguments.seed, arguments.difficulty
        plant = build_terrain_plant(scene_path, build_terrain(*ground))

    model, data = plant.model, plant.data
    torso = find_torso(model)
    push_speed = compute_push_speed(model, arguments.push_force)
    print(describe_rates())
    print("Terrain " + describe_terrain(arguments.terrain,
                                        arguments.difficulty))
    print(describe_push(arguments.push_force, push_speed))

    rng = np.random.default_rng(arguments.seed)
    command = np.array([arguments.speed, 0.0, 0.0], "float32")
    rollout = build_rollout(data, command)
    push_step, release_step = sample_push_step(rng, 0), -1
    pushes, falls = 0, 0

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
            if step >= push_step:
                apply_push(data, torso, sample_push(rng, arguments.push_force))
                release_step = step + round(PUSH_DURATION / SIMULATION_STEP)
                push_step = sample_push_step(rng, release_step)
                pushes = pushes + 1
            if step == release_step:
                apply_push(data, torso, np.zeros(3))
            data.ctrl[:] = compute_control(data, rollout.targets)
            mujoco.mj_step(model, data)
            step = step + 1
            if step % CONTROL_DECIMATION == 0:
                if pad is not None:
                    command = read_command(pad)
                    if read_push(pad) and step > release_step:
                        push_step = step
                if has_fallen(data):
                    # Left standing, the gains keep driving a collapsed
                    # robot until MuJoCo blows up. Put it back on its feet.
                    randomize_plant(model, data, plant.height, rng)
                    rollout = build_rollout(data, command)
                    falls = falls + 1
                else:
                    rollout = update_rollout(rollout, actor, data, command)
                if viewer is not None:
                    draw_push(viewer.user_scn, data, torso)
                    viewer.sync()
                    last_time = sleep_to_rate(last_time, control_period)
            if step % STATUS_STEPS == 0:
                print(describe(data, command, torso), end="\r", flush=True)
    except KeyboardInterrupt:
        pass

    distance = float(np.linalg.norm(data.qpos[:2]))
    print(f"\nRan {step} steps, took {pushes} pushes of"
          f" {arguments.push_force:.0f} N and fell {falls} times."
          f" Travelled {distance:.2f} m since the last spawn."
          f" Final base height {data.qpos[2]:.3f} m,"
          f" tilt {np.rad2deg(compute_tilt(data)):.1f} deg")
    if viewer is not None:
        viewer.close()
        time.sleep(0.25)
