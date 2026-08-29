import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax
import jax.numpy as jp
import keras
import mujoco
import numpy as np
from jax import random as jr

import checkpoint
import paz
from paz.backend import video
from robots.g1 import G1DoF29, build_reward_indices
from simulation import robust
from simulation.common import Command
from terrain import build as build_terrain
from world import build as build_world, build_mjmodel


def load_actor(directory):
    iteration = checkpoint.find_latest_iteration(Path(directory))
    return keras.models.load_model(Path(directory) / f"actor_{iteration:06d}.keras")  # fmt: skip


def build_command(forward, sideways, turn):
    values = jp.full((1,), forward), jp.full((1,), sideways)
    return Command(*values, jp.full((1,), turn))


def hold_command(state, command):
    # keep the demanded command and freeze its resample counter so the
    # observation history never sees a randomly sampled command
    counters = state.counters._replace(command=jp.zeros(1, jp.int32))
    return state._replace(command=command, counters=counters)


def render_frames(mjmodel, positions, width, height, distance=3.0):
    mjmodel.vis.global_.offwidth = width
    mjmodel.vis.global_.offheight = height
    data = mujoco.MjData(mjmodel)
    renderer = mujoco.Renderer(mjmodel, height, width)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = distance, 90, -20
    frames = []
    for qpos in positions:
        data.qpos[:] = qpos
        mujoco.mj_forward(mjmodel, data)
        camera.lookat[:] = qpos[:3]
        renderer.update_scene(data, camera)
        frames.append(renderer.render())
    renderer.close()
    return frames


def write_video(frames, directory, filepath, fps):
    paths = []
    for frame_arg, frame in enumerate(frames):
        path = str(Path(directory) / f"frame_{frame_arg:05d}.png")
        paz.image.write(path, frame)
        paths.append(path)
    video.from_paths(paths, filepath, fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", default="warp")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--forward", type=float, default=0.5)
    parser.add_argument("--sideways", type=float, default=0.0)
    parser.add_argument("--turn", type=float, default=0.0)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--filepath", default="walk.mp4")
    args = parser.parse_args()
    terrain = build_terrain(args.seed)
    robot = G1DoF29()
    device = jax.local_devices()[0]
    world = build_world(robot, terrain, args.backend, 1, device)
    indices = build_reward_indices(robot)
    max_level = world.terrain.origins.shape[0] - 1
    actor = load_actor(args.checkpoint)
    reset = robust.build_batch_reset(world, world.dynamics, None)
    step_args = world, world.dynamics, None, indices, max_level
    step = robust.build_batch_step(*step_args)
    max_speed = jp.asarray(1.0)
    command = build_command(args.forward, args.sideways, args.turn)
    levels = jp.full((1,), args.level, dtype=jp.int32)
    key = jax.random.key(args.seed)
    state = jax.jit(reset)(key, levels, max_speed)
    state = hold_command(state, command)
    step = jax.jit(step)
    # a few warmup steps flush the reset command from the history
    for _ in range(5):
        key, step_key = jr.split(key)
        action = actor(list(state.history.actor), training=False)
        state, _ = step(step_key, state, action, max_speed)
        state = hold_command(state, command)
    positions = [np.asarray(state.physics_state.qpos[0])]
    for step_arg in range(args.num_steps):
        key, step_key = jr.split(key)
        action = actor(list(state.history.actor), training=False)
        state, transition = step(step_key, state, action, max_speed)
        state = hold_command(state, command)
        positions.append(np.asarray(state.physics_state.qpos[0]))
        if bool(transition.done[0]):
            print(f"episode ended at step {step_arg}")
            break
    mjmodel = build_mjmodel(robot, terrain)
    frames = render_frames(mjmodel, positions, args.width, args.height)
    frame_directory = Path(args.filepath).with_suffix("")
    frame_directory.mkdir(parents=True, exist_ok=True)
    write_video(frames, frame_directory, args.filepath, fps=50)
    print(f"wrote {args.filepath} with {len(frames)} frames")
