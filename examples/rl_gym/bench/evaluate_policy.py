"""Run a policy (ours or a converted Isaac checkpoint) through our full
training environment and report what training would have seen: episode
length, per-term rewards in both our and Isaac's units, termination
causes, push-to-death timing, per-terrain survival, divergence samples.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jp
import keras
import numpy as np
from jax import random as jr

import checkpoint
import randomize
from networks import Parameters, read_stdv
from rewards import ROBUST_WEIGHTS
from rollout import select_done
from robots.g1 import G1DoF29, build_reward_indices
from simulation import robust
from simulation.common import compute_gravity, compute_tilt
from terrain import build as build_terrain, TERRAIN_COUNTS
from world import build as build_world

ISAAC_TO_SDK = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])  # fmt: skip
TERM_NAMES = ["track_lin_vel_xy", "track_ang_vel_z", "alive", "base_linear_velocity", "base_angular_velocity", "joint_vel", "joint_acc", "action_rate", "dof_pos_limits", "energy", "joint_deviation_arms", "joint_deviation_waists", "joint_deviation_legs", "flat_orientation_l2", "base_height", "gait", "feet_slide", "feet_clearance", "undesired_contacts"]  # fmt: skip
assert len(TERM_NAMES) == len(ROBUST_WEIGHTS)
JOINT_FIELDS = ("joint_positions", "joint_velocities", "action")
NO_PUSH = 10**8


def load_keras_policy(directory, iteration):
    if iteration is None:
        iteration = checkpoint.find_latest_iteration(directory)
    actor = checkpoint.load_actor(directory, iteration)
    layers = []
    for layer in actor.layers:
        if isinstance(layer, keras.layers.Dense):
            kernel, bias = layer.get_weights()
            layers.append((jp.asarray(kernel), jp.asarray(bias)))
    stdv = read_stdv(Parameters(checkpoint.load_stdv(directory, iteration), None, None))  # fmt: skip
    return layers, stdv, np.arange(29)


def load_isaac_policy(path):
    arrays = np.load(path)
    layers = []
    for slot in (0, 2, 4, 6):
        kernel = jp.asarray(arrays[f"actor.{slot}.weight"].T)
        layers.append((kernel, jp.asarray(arrays[f"actor.{slot}.bias"])))
    return layers, jp.asarray(arrays["std"]), ISAAC_TO_SDK


def call_mlp(layers, x):
    for kernel, bias in layers[:-1]:
        x = jax.nn.elu(x @ kernel + bias)
    kernel, bias = layers[-1]
    return x @ kernel + bias


def flatten_history(history, permutation):
    flat = []
    for name, term in zip(history._fields, history):
        if name in JOINT_FIELDS:
            term = term[..., permutation]
        flat.append(term.reshape(term.shape[0], -1))
    return jp.concatenate(flat, axis=-1)


def disable_pushes(state):
    push = jp.full_like(state.counters.push, NO_PUSH)
    return state._replace(counters=state.counters._replace(push=push))


def build_runner(step, reset, layers, stdv, permutation, max_speed, pushes, stochastic):  # fmt: skip
    inverse = np.argsort(permutation)

    def advance(carry, _):
        state, key = carry
        keys = jr.split(key, 4)
        observation = flatten_history(state.history.actor, permutation)
        mean = call_mlp(layers, observation)
        noise = jr.normal(keys[1], mean.shape) * stdv * stochastic
        action = (mean + noise)[:, inverse]
        next_state, transition = step(keys[2], state, action, max_speed)
        fresh = reset(keys[3], transition.level, state.tile.column, max_speed)  # fmt: skip
        if not pushes:
            fresh = disable_pushes(fresh)
        quaternions = next_state.physics_state.qpos[:, 3:7]
        gravity = jax.vmap(compute_gravity)(quaternions)
        tilt = jax.vmap(compute_tilt)(gravity)
        pushed = next_state.counters.push != state.counters.push
        low = next_state.physics_state.qpos[:, 2] < 0.2
        first_diverged = jp.argmax(transition.diverged)
        sample = (state.physics_state.qpos[first_diverged], state.physics_state.qvel[first_diverged], action[first_diverged], first_diverged)  # fmt: skip
        record = (transition.terms, transition.done, transition.timeout, transition.diverged, low, tilt > 0.8, pushed, next_state.counters.episode, state.tile.column, jp.abs(action).max(axis=-1), sample)  # fmt: skip
        state = select_done(transition.done, fresh, next_state)
        return (state, keys[0]), record

    def run(state, key, num_steps):
        return jax.lax.scan(advance, (state, key), None, length=num_steps)

    return jax.jit(run, static_argnums=2)


def terrain_type(column):
    names = []
    for name, count in zip(TERRAIN_COUNTS._fields, TERRAIN_COUNTS):
        names.extend([name] * count)
    return names[column]


def summarize(records, length_isaac=None):
    terms, done, timeout, diverged, low, tilted, pushed, episode, column, action_max, sample = records  # fmt: skip
    terms, done = np.asarray(terms), np.asarray(done).astype(bool)
    num_steps, num_envs = done.shape
    completed = done.sum()
    length = done.size / max(completed, 1)
    print(f"steps {done.size}  episodes {completed}  length {length:.1f}")
    fallen = done & ~np.asarray(timeout).astype(bool) & ~np.asarray(diverged).astype(bool)  # fmt: skip
    low, tilted = np.asarray(low), np.asarray(tilted)
    print(f"  timeouts {np.asarray(timeout).sum()/max(completed,1):.3f}  diverged {np.asarray(diverged).sum()/max(completed,1):.4f}  falls {fallen.sum()/max(completed,1):.3f}  (low {np.sum(fallen & low)/max(completed,1):.3f}, tilted {np.sum(fallen & tilted)/max(completed,1):.3f})")  # fmt: skip
    print(f"  diverged steps {np.asarray(diverged).sum()} of {done.size} ({np.asarray(diverged).mean()*100:.4f}%)")  # fmt: skip
    print(f"  mean |action|max {np.asarray(action_max).mean():.2f}")
    reward = (terms * np.asarray(ROBUST_WEIGHTS)).sum(-1) * 0.02
    print(f"  per-step reward {reward.mean():.5f}  return/episode ~{reward.mean()*length:.3f}")  # fmt: skip
    flat_terms = terms.reshape(-1, terms.shape[-1])
    weighted = flat_terms * np.asarray(ROBUST_WEIGHTS) * 0.02
    extremes = np.percentile(reward, [0.01, 1, 50, 99, 99.99])
    print(f"  per-step reward percentiles 0.01/1/50/99/99.99: {extremes}")
    worst = np.argmin(weighted.sum(1))
    print("  worst step contributions: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(TERM_NAMES, weighted[worst]) if abs(v) > 0.01))  # fmt: skip
    means = flat_terms.mean(0)
    print(f"  {'term':24s} {'ours/step':>10s} {'isaac-units':>12s}")
    for name, mean, weight in zip(TERM_NAMES, means, np.asarray(ROBUST_WEIGHTS)):  # fmt: skip
        isaac_units = weight * mean * length * 0.02 / 20.0
        print(f"  {name:24s} {mean:10.4f} {isaac_units:12.4f}")
    summarize_push_deaths(fallen, np.asarray(pushed), np.asarray(episode))
    summarize_terrain(done, fallen, np.asarray(column))
    summarize_death_steps(fallen, np.asarray(episode))


def summarize_push_deaths(fallen, pushed, episode):
    # for each fall, control steps since the last push in the same episode
    num_steps, num_envs = fallen.shape
    since = np.full(num_envs, -1)
    gaps = []
    for step in range(num_steps):
        since = np.where(episode[step] <= 1, -1, since)
        since = np.where(since >= 0, since + 1, since)
        since = np.where(pushed[step], 0, since)
        gaps.extend(since[fallen[step]].tolist())
    gaps = np.array(gaps)
    if gaps.size:
        near = np.mean((gaps >= 0) & (gaps <= 50))
        print(f"  falls within 1 s after a push: {near:.3f}  (pushed at all before falling: {np.mean(gaps >= 0):.3f}, pushes total {pushed.sum()})")  # fmt: skip


def summarize_terrain(done, fallen, column):
    print("  per terrain type: length / falls")
    types = np.array([terrain_type(c) for c in range(column.max() + 1)])
    for name in TERRAIN_COUNTS._fields:
        mask = np.isin(column, np.where(types == name)[0])
        completed = done[mask].sum()
        print(f"    {name:15s} length {mask.sum()/max(completed,1):7.1f}  fall share {fallen[mask].sum()/max(completed,1):.3f}")  # fmt: skip


def summarize_death_steps(fallen, episode):
    steps = episode[fallen]
    if steps.size:
        edges = [0, 25, 50, 100, 150, 200, 300, 400, 500, 750, 1001]
        hist, _ = np.histogram(steps, edges)
        cells = " ".join(f"{a}-{b}:{h/steps.size:.2f}" for a, b, h in zip(edges[:-1], edges[1:], hist))  # fmt: skip
        print(f"  fall step histogram {cells}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keras", default=None)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--isaac", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--level_uniform", type=int, default=0)
    parser.add_argument("--max_speed", type=float, default=0.1)
    parser.add_argument("--randomize", type=int, default=1)
    parser.add_argument("--pushes", type=int, default=1)
    parser.add_argument("--stochastic", type=int, default=1)
    parser.add_argument("--save", default=None)
    # terrain columns as a half-open range: flat 0:5, rough 5:13, slope
    # 13:15, inverted 15:17, boxes 17:21
    parser.add_argument("--columns", default="0:21")
    args = parser.parse_args()
    if args.isaac:
        layers, stdv, permutation = load_isaac_policy(args.isaac)
    else:
        layers, stdv, permutation = load_keras_policy(args.keras, args.iteration)  # fmt: skip
    print(f"policy stdv mean {float(stdv.mean()):.3f}")
    terrain = build_terrain(args.seed)
    robot = G1DoF29()
    world = build_world(robot, terrain, "warp", args.num_envs)
    indices = build_reward_indices(robot)
    max_level = world.terrain.origins.shape[0] - 1
    keys = jr.split(jr.key(args.seed), 6)
    if args.randomize:
        random_keys = jr.split(keys[0], args.num_envs)
        torso_arg = robot.bodies.torso_link.arg
        randomize_args = random_keys, world.dynamics, robot.num_actuators, torso_arg  # fmt: skip
        dynamics, axes = randomize.physics(*randomize_args)
    else:
        dynamics, axes = world.dynamics, None
    reset = robust.build_batch_reset(world, dynamics, axes)
    step = robust.build_batch_step(world, dynamics, axes, indices, max_level)
    levels = jp.full((args.num_envs,), args.level, dtype=jp.int32)
    if args.level_uniform:
        levels = jr.randint(keys[5], (args.num_envs,), 0, args.level + 1)
    first_column, last_column = map(int, args.columns.split(":"))
    columns = jr.randint(keys[1], (args.num_envs,), first_column, last_column)  # fmt: skip
    max_speed = jp.asarray(args.max_speed)
    state = jax.jit(reset)(keys[2], levels, columns, max_speed)
    if not args.pushes:
        state = disable_pushes(state)
    runner_args = step, reset, layers, stdv, permutation, max_speed, args.pushes, float(args.stochastic)  # fmt: skip
    run = build_runner(*runner_args)
    # a warm-up pass so the synchronized first episodes do not bias stats
    (state, key), _ = run(state, keys[4], 200)
    (state, key), records = run(state, key, args.num_steps)
    records = jax.tree.map(np.asarray, records)
    summarize(records)
    if args.save:
        terms, done, timeout, diverged, low, tilted, pushed, episode, column, action_max, sample = records  # fmt: skip
        np.savez(args.save, terms=terms, done=done, timeout=timeout, diverged=diverged, low=low, tilted=tilted, pushed=pushed, episode=episode, column=column, sample_qpos=sample[0], sample_qvel=sample[1], sample_action=sample[2], sample_env=sample[3])  # fmt: skip
        print(f"saved {args.save}")
