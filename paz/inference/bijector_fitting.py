import jax
import jax.numpy as jp
import optax
from tensorflow_probability.substrates import jax as tfp

from paz.inference import progress as progress_module

tfd = tfp.distributions


def fit_bijector(
    source_distribution,
    target_distribution,
    initial_bijector,
    key,
    num_samples=10_000,
    optimizer=None,
    num_steps=1000,
    print=True,
):
    target_samples = target_distribution.sample(num_samples, seed=key)
    params, treedef = jax.tree_util.tree_flatten(initial_bijector)
    if optimizer is None:
        optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    def loss_fn(current_params):
        bijector = jax.tree_util.tree_unflatten(treedef, current_params)
        transformed = tfd.TransformedDistribution(
            distribution=source_distribution, bijector=bijector
        )
        return -jp.mean(transformed.log_prob(target_samples))

    @jax.jit
    def update_step(current_params, current_state):
        loss, grads = jax.value_and_grad(loss_fn)(current_params)
        updates, new_state = optimizer.update(grads, current_state, current_params)
        new_params = optax.apply_updates(current_params, updates)
        return new_params, new_state, loss

    start_time = progress_module.now()
    losses = []
    current_params = params
    current_state = opt_state
    for step in range(num_steps):
        current_params, current_state, loss = update_step(current_params, current_state)
        losses.append(float(loss))
        if print:
            progress_module.draw_bar(
                step + 1, num_steps, start_time, "fit bijector", 30
            )
    if print:
        progress_module.move_to_next_line()
    optimized_bijector = jax.tree_util.tree_unflatten(treedef, current_params)
    return optimized_bijector, losses
