"""Headless robustness sweep for the unitree_rl_lab G1 velocity policy.

Holds one velocity command over every terrain and seed, shoving the torso on
the training push schedule, and reports how often the robot goes down.
"""

import argparse
from collections import namedtuple
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import mujoco
import numpy as np

from policy import compile_actor
from policy import find_latest_checkpoint
from policy import load_actor
from simulation import CONTROL_DECIMATION
from simulation import CONTROL_FREQUENCY
from simulation import PUSH_DURATION
from simulation import SIMULATION_FREQUENCY
from simulation import SIMULATION_STEP
from simulation import apply_push
from simulation import build_flat_plant
from simulation import build_rollout
from simulation import build_terrain_plant
from simulation import compute_control
from simulation import compute_push_speed
from simulation import find_torso
from simulation import has_fallen
from simulation import randomize_plant
from simulation import sample_push
from simulation import sample_push_step
from simulation import update_rollout
from terrain import TERRAIN_NAMES
from terrain import build_terrain
from terrain import describe_terrain

Result = namedtuple("Result", "distance steps fell")


def build_plant(scene_path, name, seed, difficulty):
    if name == "flat":
        plant = build_flat_plant(scene_path)
    else:
        terrain = build_terrain(name, seed, difficulty)
        plant = build_terrain_plant(scene_path, terrain)
    return plant


def run_rollout(plant, actor, command, steps, force, seed):
    # The rollout ends at the fall the way the trained episode does. Left
    # running, the gains keep driving a collapsed robot until MuJoCo blows up.
    data, rng = plant.data, np.random.default_rng(seed)
    randomize_plant(plant.model, data, plant.height, rng)
    torso = find_torso(plant.model)
    rollout = build_rollout(data, command)
    push_step, release_step, step = sample_push_step(rng, 0), -1, 0
    fallen = False
    while step < steps and not fallen:
        if step >= push_step:
            apply_push(data, torso, sample_push(rng, force))
            release_step = step + round(PUSH_DURATION / SIMULATION_STEP)
            push_step = sample_push_step(rng, release_step)
        if step == release_step:
            apply_push(data, torso, np.zeros(3))
        data.ctrl[:] = compute_control(data, rollout.targets)
        mujoco.mj_step(plant.model, data)
        step = step + 1
        if step % CONTROL_DECIMATION == 0:
            rollout = update_rollout(rollout, actor, data, command)
            fallen = has_fallen(data)
    distance = float(np.linalg.norm(data.qpos[:2]))
    return Result(distance, step, fallen)


def report(name, results, steps):
    falls = sum(result.fell for result in results)
    distances = [result.distance for result in results]
    survived = [result.steps * SIMULATION_STEP for result in results]
    print(f"| {name} | {falls}/{len(results)} | {np.median(distances):.2f} m"
          f" | {np.median(survived):.1f} s |")


if __name__ == "__main__":
    example_dir = Path(__file__).resolve().parent
    repositories = example_dir.parents[2]
    sibling = repositories / "unitree_mujoco" / "unitree_robots" / "g1"
    default_scene = os.environ.get("WALK_SCENE_DIR", sibling)
    experiment = example_dir / "unitree_g1_29dof_velocity_robust"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=default_scene)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--terrains", nargs="+", default=TERRAIN_NAMES,
                        choices=TERRAIN_NAMES)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--difficulty", type=float, default=1.0,
                        help="1 is the tallest trained; above 1 is out"
                             " of distribution")
    parser.add_argument("--push-force", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=2000)
    arguments = parser.parse_args()

    scene_path = Path(arguments.scene_dir) / "scene_29dof.xml"
    if not scene_path.exists():
        raise SystemExit(f"Missing MuJoCo scene: {scene_path}")
    checkpoint = arguments.checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint(experiment)
    actor = compile_actor(load_actor(checkpoint))
    command = np.array([arguments.speed, 0.0, 0.0], "float32")

    seconds = arguments.steps * SIMULATION_STEP
    reference = build_plant(scene_path, "flat", 0, 1.0)
    speed = compute_push_speed(reference.model, arguments.push_force)
    print(f"{checkpoint.parent.name}/{checkpoint.name}, {seconds:.0f} s at"
          f" {arguments.speed} m/s, {arguments.push_force:.0f} N pushes"
          f" worth {speed:.2f} m/s")
    print(f"Control {CONTROL_FREQUENCY:.0f} Hz over physics at"
          f" {SIMULATION_FREQUENCY:.0f} Hz, decimation {CONTROL_DECIMATION}")
    for name in arguments.terrains:
        print("Terrain " + describe_terrain(name, arguments.difficulty))
    print("| Terrain | Falls | Median distance | Median time up |")
    print("| --- | --- | --- | --- |")
    for name in arguments.terrains:
        results = []
        for seed in range(arguments.seeds):
            ground = scene_path, name, seed, arguments.difficulty
            plant = build_plant(*ground)
            run = actor, command, arguments.steps, arguments.push_force, seed
            results.append(run_rollout(plant, *run))
        report(name, results, arguments.steps)
