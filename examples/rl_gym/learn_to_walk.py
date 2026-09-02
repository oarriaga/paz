import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
# the warp physics allocator lives outside XLA's pool and its solver
# workspace dominates device memory, so XLA must allocate on demand
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jp
from jax import random as jr

from functools import partial

import curriculum
import log
import paz
import ppo
from networks import Optimizer, PPO, compute_shapes, snapshot_parameters
from rollout import build_collect
from robots.g1 import G1DoF29, build_reward_indices
from simulation import robust
from terrain import build as build_terrain
from world import build as build_world


def build_batch_reset(world):
    axes = 0, None, 0, None, None, None
    batched = jax.vmap(robust.reset, in_axes=axes)
    template, origins = world.physics_template, world.terrain.origins

    def reset(key, level, max_speed):
        keys = jr.split(key, level.shape[0])
        args = keys, world.dynamics, level, max_speed, template, origins
        return batched(*args)

    return reset


def build_batch_step(world, robot, indices, max_level):
    kwargs = dict(robot=robot, indices=indices, max_level=max_level)
    kwargs["tile_size"] = world.terrain.tile_size
    axes = 0, None, 0, 0, None
    batched = jax.vmap(partial(robust.step, **kwargs), in_axes=axes)

    def step(key, state, action, max_speed):
        keys = jr.split(key, action.shape[0])
        return batched(keys, world.dynamics, state, action, max_speed)

    return step


def sample_levels(key, num_envs, max_level):
    return jax.random.randint(key, (num_envs,), 0, max_level + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--backend", default="warp")
    parser.add_argument("--max_level", type=int, default=8)
    parser.add_argument("--initial_max_speed", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument("--num_steps", type=int, default=24)
    parser.add_argument("--log_interval", type=int, default=10)
    # mujoco warp budgets contacts (naconmax) for the whole batch and
    # constraint rows (njmax) for one environment
    parser.add_argument("--num_contacts", type=int, default=32)
    parser.add_argument("--num_constraints", type=int, default=256)
    parser.add_argument("--root", default="experiments")
    args = parser.parse_args()
    terrain = build_terrain(args.seed)
    robot = G1DoF29()
    budget = args.num_contacts, args.num_constraints
    world_args = robot, terrain, args.backend, args.num_envs, *budget
    world = build_world(*world_args)
    indices = build_reward_indices(robot)
    num_levels = world.terrain.origins.shape[0]
    max_level = min(args.max_level, num_levels - 1)
    reset = build_batch_reset(world)
    step = build_batch_step(world, robot, indices, max_level)
    level_key = jax.random.key(args.seed + 1)
    levels = sample_levels(level_key, args.num_envs, max_level)
    max_speed = jp.asarray(args.initial_max_speed)
    state = jax.jit(reset)(jax.random.key(args.seed), levels, max_speed)
    actor_shapes = compute_shapes(state.history.actor)
    critic_shapes = compute_shapes(state.history.critic)
    actor, critic, stdv = PPO(actor_shapes, critic_shapes)
    optimizer_args = actor, critic, stdv, args.learning_rate
    optimizer, optimizer_state = Optimizer(*optimizer_args)
    update = ppo.build_update(actor, critic, optimizer)
    collect = jax.jit(build_collect(actor, critic, reset, step, args.num_steps))  # fmt: skip
    parameters = snapshot_parameters(actor, critic, stdv)
    learning_rate = jp.asarray(args.learning_rate)
    training = ppo.TrainingState(parameters, optimizer_state, learning_rate)
    rollout_key = jax.random.key(args.seed + 2)
    root = paz.directory.make_timestamped(args.root, "walk")
    paz.file.write_json(args.__dict__, Path(root) / "parameters.json")
    log_file, writer = log.open_log(root)
    print(f"walking on {args.num_envs} environments")
    print(f"writing to {root}")
    print(f"terrain {world.terrain.origins.shape} tiles")
    print(f"actor parameters {actor.count_params()}")
    print(f"critic parameters {critic.count_params()}")
    started = time.perf_counter()
    try:
        for iteration in range(1, args.num_iterations + 1):
            collect_args = state, training.parameters, rollout_key, max_speed
            state, rollout_key, experience, metrics = collect(*collect_args)
            update_key = jax.random.key(args.seed + iteration)
            training, update_metrics = update(update_key, training, experience)
            speed_args = max_speed, metrics.terms[0], iteration
            max_speed = curriculum.update_max_speed(*speed_args)
            if iteration % args.log_interval == 0:
                steps = iteration * args.num_envs * args.num_steps
                elapsed = time.perf_counter() - started
                row_args = iteration, metrics, update_metrics, max_speed
                row = log.build_row(*row_args, steps, elapsed)
                log.write_row(writer, log_file, row)
                log.print_row(row)
    finally:
        log_file.close()
