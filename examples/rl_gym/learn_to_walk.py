import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
# the warp physics allocator lives outside XLA's pool and its solver
# workspace dominates device memory, so XLA must allocate on demand
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import distributed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--backend", default="warp")
    parser.add_argument("--max_level", type=int, default=8)
    parser.add_argument("--initial_max_speed", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--entropy_coefficient", type=float, default=0.01)
    # the reference trains without empirical normalization; a fresh
    # normalizer is the identity, so skipping its updates disables it
    parser.add_argument("--normalize_observations", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument("--num_steps", type=int, default=24)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--load", default=None)
    # mujoco warp budgets contacts (naconmax) for the whole batch and
    # constraint rows (njmax) for one environment
    parser.add_argument("--num_contacts", type=int, default=32)
    parser.add_argument("--num_constraints", type=int, default=256)
    # multi-GPU runs launch one process per GPU, each simulating its own
    # environments; gradients are averaged globally inside the update
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--process_id", type=int, default=0)
    parser.add_argument("--coordinator", default="localhost:9911")
    parser.add_argument("--root", default="experiments")
    args = parser.parse_args()
    if args.num_processes > 1:
        initialize_args = args.coordinator, args.num_processes, args.process_id  # fmt: skip
        distributed.initialize(*initialize_args)
    # these imports build jax arrays at module scope, so in a multi-GPU
    # run they must come after the distributed initialization
    import jax
    import jax.numpy as jp
    import keras
    from jax import random as jr

    import checkpoint
    import curriculum
    import log
    import paz
    import ppo
    import randomize
    from networks import Optimizer, PPO, compute_shapes, snapshot_parameters
    from rollout import build_collect, build_normalizers
    from robots.g1 import G1DoF29, build_reward_indices
    from simulation import robust
    from simulation.common import decorrelate_counters
    from terrain import build as build_terrain
    from world import build as build_world

    is_leader = jax.process_index() == 0
    num_envs = args.num_envs * jax.process_count()
    device = jax.local_devices()[0]
    terrain = build_terrain(args.seed)
    robot = G1DoF29()
    budget = args.num_contacts, args.num_constraints
    world_args = robot, terrain, args.backend, args.num_envs, device
    world = build_world(*world_args, *budget)
    indices = build_reward_indices(robot)
    num_levels = world.terrain.origins.shape[0]
    max_level = min(args.max_level, num_levels - 1)
    environment_key = jr.fold_in(jax.random.key(args.seed), args.process_id)
    environment_keys = jr.split(environment_key, 5)
    random_keys = jr.split(environment_keys[0], args.num_envs)
    torso_arg = robot.bodies.torso_link.arg
    randomize_args = random_keys, world.dynamics, robot.num_actuators, torso_arg  # fmt: skip
    dynamics, dynamics_axes = randomize.physics(*randomize_args)
    reset = robust.build_batch_reset(world, dynamics, dynamics_axes)
    step_args = world, dynamics, dynamics_axes, indices, max_level
    step = robust.build_batch_step(*step_args)
    levels = jax.random.randint(environment_keys[1], (args.num_envs,), 0, max_level + 1)  # fmt: skip
    max_speed = jp.asarray(args.initial_max_speed)
    state = jax.jit(reset)(environment_keys[2], levels, max_speed)
    state = jax.jit(decorrelate_counters)(environment_keys[3], state)
    rollout_key = environment_keys[4]
    normalizers = build_normalizers(state.history)
    keras.utils.set_random_seed(args.seed)
    actor_shapes = compute_shapes(state.history.actor)
    critic_shapes = compute_shapes(state.history.critic)
    actor, critic, stdv = PPO(actor_shapes, critic_shapes)
    optimizer_args = actor, critic, stdv, args.learning_rate
    optimizer, optimizer_state = Optimizer(*optimizer_args)
    learning_rate, start_iteration = args.learning_rate, 0
    if args.load:
        loaded = checkpoint.load(args.load, actor, critic)
        stdv.assign(jp.asarray(loaded.stdv))
        optimizer_state = jax.tree.map(jp.asarray, loaded.optimizer_state)
        learning_rate, start_iteration = loaded.learning_rate, loaded.iteration  # fmt: skip
        max_speed = jp.asarray(loaded.max_speed)
        normalizers = checkpoint.restore_normalizers(loaded, normalizers)
        # redo the initial reset so the first commands follow the restored
        # speed curriculum, with keys advanced past the ones the original
        # run already consumed
        environment_keys = jr.split(jr.fold_in(environment_key, start_iteration), 5)  # fmt: skip
        state = jax.jit(reset)(environment_keys[2], levels, max_speed)
        state = jax.jit(decorrelate_counters)(environment_keys[3], state)
        rollout_key = environment_keys[4]
    mesh = distributed.build_mesh()
    update_args = actor, critic, optimizer, mesh.devices.size
    update = ppo.build_update(*update_args, entropy_weight=args.entropy_coefficient)  # fmt: skip
    collect = jax.jit(build_collect(actor, critic, reset, step, args.num_steps))  # fmt: skip
    parameters = snapshot_parameters(actor, critic, stdv)
    learning_rate = jp.asarray(learning_rate)
    training = ppo.TrainingState(parameters, optimizer_state, learning_rate)
    training = distributed.replicate(mesh, training)
    if is_leader:
        root = paz.directory.make_timestamped(args.root, "walk")
        paz.file.write_json(args.__dict__, Path(root) / "parameters.json")
        log_file, writer = log.open_log(root)
        print(f"walking on {num_envs} environments")
        print(f"writing to {root}")
        print(f"terrain {world.terrain.origins.shape} tiles")
        print(f"actor parameters {actor.count_params()}")
        print(f"critic parameters {critic.count_params()}")
    started = time.perf_counter()
    try:
        for iteration in range(start_iteration + 1, args.num_iterations + 1):
            parameters = distributed.localize(training.parameters)
            collect_args = state, parameters, normalizers, rollout_key, max_speed  # fmt: skip
            outputs = collect(*collect_args)
            state, rollout_key, experience, updated_normalizers, metrics = outputs  # fmt: skip
            if args.normalize_observations:
                normalizers = updated_normalizers
            experience = distributed.shard_experience(mesh, experience)
            training, update_metrics = update(args.seed + iteration, training, experience)  # fmt: skip
            tracking = distributed.global_mean(mesh, metrics.terms[0])
            episode_length = distributed.global_mean(mesh, metrics.episode_length)  # fmt: skip
            speed_args = max_speed, tracking, episode_length, iteration
            max_speed = curriculum.update_max_speed(*speed_args, args.num_steps)  # fmt: skip
            save_now = iteration % args.save_interval == 0
            if is_leader and save_now:
                save_args = Path(root) / "checkpoints", iteration
                checkpoint.save(*save_args, actor, critic, training, max_speed, normalizers)  # fmt: skip
            if is_leader and iteration % args.log_interval == 0:
                iterations = iteration - start_iteration
                steps = iterations * num_envs * args.num_steps
                elapsed = time.perf_counter() - started
                row_args = iteration, metrics, update_metrics, max_speed
                row = log.build_row(*row_args, steps, elapsed)
                log.write_row(writer, log_file, row)
                log.print_row(row)
    finally:
        if is_leader:
            log_file.close()
