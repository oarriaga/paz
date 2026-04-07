import jax
import jax.numpy as jp
import optax
from tensorflow_probability.substrates import jax as tfp

import paz.utils.progressbar as progressbar

tfd = tfp.distributions


def fit_bijector(source_distribution, target_distribution, initial_bijector, key, num_samples=10_000, optimizer=None, num_steps=1000, verbose=True):  # fmt: skip

    target_samples = target_distribution.sample(num_samples, seed=key)
    parameters, treedef = jax.tree_util.tree_flatten(initial_bijector)
    if optimizer is None:
        optimizer = optax.adam(1e-3)
    state = optimizer.init(parameters)

    def loss_fn(parameters):
        bijector = jax.tree_util.tree_unflatten(treedef, parameters)
        kwargs = {"distribution": source_distribution, "bijector": bijector}
        transformed = tfd.TransformedDistribution(**kwargs)
        return -jp.mean(transformed.log_prob(target_samples))

    @jax.jit
    def update_step(parameters, state):
        loss, grads = jax.value_and_grad(loss_fn)(parameters)
        updates, new_state = optimizer.update(grads, state, parameters)
        new_parameters = optax.apply_updates(parameters, updates)
        return new_parameters, new_state, loss

    start_time = progressbar.start() if verbose else None
    losses = []
    for arg in range(num_steps):
        parameters, state, loss = update_step(parameters, state)
        losses.append(float(loss))
        if verbose:
            progressbar.draw(arg + 1, num_steps, start_time, "fit bijector", 30)
    if verbose:
        progressbar.newline()
    optimized_bijector = jax.tree_util.tree_unflatten(treedef, parameters)
    return optimized_bijector, losses
