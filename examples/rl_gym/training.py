from functools import partial
import jax
import jax.numpy as jp

def train(seed, sim_backend, num_envs, num_steps, actor, critic, stdv, optimizer, optimizer_state, build_environment, conditioned, max_speed=0.1):  # fmt: skip
    parameters = snapshot_parameters(actor, critic, stdv)
    learning_rate = jp.asarray(optimizer_state[1])
    _, model, reset, step = build_environment(sim_backend, num_envs, seed)  # fmt: skip
    initialize = build_initialize(model, reset)
    keys = build_environment_keys(seed, num_envs)
    levels = build_initial_levels(seed, num_envs)
    models, state = initialize(keys[0], keys[1], levels, max_speed)
    devices = jax.local_devices()
    training_state = parameters, optimizer_state, learning_rate
    replicate = partial(replicate_for_devices, num_devices=len(devices))
    training_state = jax.tree.map(replicate, training_state)
    rollout_keys = jax.random.split(jax.random.key(arguments.seed + 2), len(devices))  # fmt: skip
    collect = build_collect(actor, critic, reset, step)
    update_algorithm = (update_distributed_fixed if conditioned else update_distributed)  # fmt: skip
    update = build_update(actor, critic, optimizer, update_algorithm)

    for step_arg in range(num_steps):
        collect_args = (models, state, training_state[0], rollout_keys, max_speed)  # fmt: skip
        outputs = collect(*collect_args)
        state, rollout_keys, batch, metrics = outputs
        update_keys = jax.random.split(jax.random.key(seed + step_arg), len(devices))  # fmt: skip
        training_state, update_metrics = update(training_state, batch, update_keys)  # fmt: skip
        tracking = float(jp.mean(metrics[2][..., 0]))
        command_args = max_speed, tracking, step_arg
        max_speed = update_max_speed(*command_args)


def build_initialize(model, reset):

    @partial(jax.pmap, in_axes=(0, 0, 0, None), axis_name="devices")
    def initialize(random_keys, reset_keys, levels, max_speed):
        models, model_axes = randomize_model(random_keys, model)
        reset_batch = jax.vmap(reset, in_axes=(model_axes, 0, 0, None))
        state = reset_batch(models, reset_keys, levels, max_speed)
        return models, state

    return initialize
